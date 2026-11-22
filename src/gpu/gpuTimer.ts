/**
 * Measures GPU time per pass with timestamp queries. Readback is asynchronous,
 * so results arrive a few frames late, but the CPU never waits on the GPU.
 */

const BYTES_PER_TIMESTAMP = 8;
const NANOSECONDS_PER_MILLISECOND = 1e6;
// Enough buffers to cover a few frames of CPU-GPU latency. When all are in
// flight a frame is simply not sampled, rather than stalling the pipeline.
// 足够覆盖几帧的 CPU-GPU 延迟；全部占用时这一帧不采样，而不是让流水线停下来等。
const READBACK_BUFFER_COUNT = 3;

export class GpuTimer<PassName extends string> {
  private readonly passNames: readonly PassName[];
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly writes: ReadonlyMap<PassName, GPUComputePassTimestampWrites>;
  private readonly freeReadbacks: GPUBuffer[] = [];
  private pendingReadback: GPUBuffer | null = null;
  private readonly sumsMs: Float64Array;
  private sampleCount = 0;

  /** The device must have been created with the 'timestamp-query' feature. */
  constructor(device: GPUDevice, passNames: readonly PassName[]) {
    if (
      passNames.length === 0 ||
      new Set(passNames).size !== passNames.length
    ) {
      throw new Error('GpuTimer needs at least one pass and unique names.');
    }
    this.passNames = passNames;
    const queryCount = 2 * passNames.length;
    this.querySet = device.createQuerySet({
      label: 'pass timestamps',
      type: 'timestamp',
      count: queryCount,
    });
    const byteSize = queryCount * BYTES_PER_TIMESTAMP;
    this.resolveBuffer = device.createBuffer({
      label: 'timestamp resolve',
      size: byteSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    for (let i = 0; i < READBACK_BUFFER_COUNT; i++) {
      this.freeReadbacks.push(
        device.createBuffer({
          label: `timestamp readback ${i}`,
          size: byteSize,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
      );
    }
    this.writes = new Map(
      passNames.map((name, i) => [
        name,
        {
          querySet: this.querySet,
          beginningOfPassWriteIndex: 2 * i,
          endOfPassWriteIndex: 2 * i + 1,
        },
      ]),
    );
    this.sumsMs = new Float64Array(passNames.length);
  }

  /** Works for both compute and render pass descriptors. */
  timestampWrites(pass: PassName): GPUComputePassTimestampWrites {
    const writes = this.writes.get(pass);
    if (!writes) throw new Error(`Unknown timed pass "${pass}".`);
    return writes;
  }

  /** Records the copy of this frame's timestamps; call before `finish()`. */
  resolve(encoder: GPUCommandEncoder): void {
    if (this.pendingReadback) {
      throw new Error('resolve() called twice without startReadback().');
    }
    const readback = this.freeReadbacks.pop();
    if (!readback) return;
    encoder.resolveQuerySet(
      this.querySet,
      0,
      2 * this.passNames.length,
      this.resolveBuffer,
      0,
    );
    encoder.copyBufferToBuffer(
      this.resolveBuffer,
      0,
      readback,
      0,
      this.resolveBuffer.size,
    );
    this.pendingReadback = readback;
  }

  /** Call after submitting the encoder passed to `resolve()`. */
  startReadback(): void {
    const readback = this.pendingReadback;
    if (!readback) return;
    this.pendingReadback = null;
    readback.mapAsync(GPUMapMode.READ).then(
      () => {
        try {
          this.accumulate(new BigUint64Array(readback.getMappedRange()));
        } finally {
          readback.unmap();
          this.freeReadbacks.push(readback);
        }
      },
      (error: unknown) => {
        // Only fails when the device is lost; the buffer is not reused then.
        // 只有设备丢失时才会失败，此时这个 buffer 不再复用。
        console.warn('GPU timestamp readback failed.', error);
      },
    );
  }

  /** Mean milliseconds per pass since the last call, or null if no samples. */
  takeAverages(): ReadonlyMap<PassName, number> | null {
    if (this.sampleCount === 0) return null;
    const averages = new Map(
      this.passNames.map((name, i) => [
        name,
        this.sumsMs[i] / this.sampleCount,
      ]),
    );
    this.sumsMs.fill(0);
    this.sampleCount = 0;
    return averages;
  }

  private accumulate(timestamps: BigUint64Array): void {
    const durations = this.passNames.map(
      (_, i) => timestamps[2 * i + 1] - timestamps[2 * i],
    );
    // Implementations may return zero or non-monotonic values (e.g. across a
    // GPU clock change); such frames are dropped instead of skewing the mean.
    // 实现可能返回 0 或非单调的时间戳（比如 GPU 频率切换时），这种帧直接丢弃，避免拉偏平均值。
    if (durations.some(duration => duration <= 0n)) return;
    durations.forEach((duration, i) => {
      this.sumsMs[i] += Number(duration) / NANOSECONDS_PER_MILLISECOND;
    });
    this.sampleCount++;
  }
}
