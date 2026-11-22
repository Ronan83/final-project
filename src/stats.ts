const SAMPLE_WINDOW_FRAMES = 30;

/** Source of per-pass GPU times, e.g. GpuTimer. */
export interface GpuTimingSource {
  takeAverages(): ReadonlyMap<string, number> | null;
}

/** Frame-time overlay averaged over a fixed window of frames. */
export class FrameStats {
  private readonly element: HTMLDivElement;
  private readonly gpuTiming: GpuTimingSource | null;
  private gpuLine: string;
  private accumulatedMs = 0;
  private frameCount = 0;

  /** `gpuTiming` is null when the device cannot measure GPU time. */
  constructor(parent: HTMLElement, gpuTiming: GpuTimingSource | null) {
    this.element = document.createElement('div');
    this.element.className = 'frame-stats';
    parent.appendChild(this.element);
    this.gpuTiming = gpuTiming;
    this.gpuLine = gpuTiming
      ? 'GPU timing: waiting'
      : 'GPU timing: unavailable';
  }

  record(frameMs: number, renderSize: readonly [number, number]): void {
    if (frameMs <= 0) return;
    this.accumulatedMs += frameMs;
    this.frameCount++;
    if (this.frameCount < SAMPLE_WINDOW_FRAMES) return;

    const averageMs = this.accumulatedMs / this.frameCount;
    const gpuAverages = this.gpuTiming?.takeAverages();
    if (gpuAverages) this.gpuLine = formatGpuLine(gpuAverages);
    this.element.textContent =
      `${(1000 / averageMs).toFixed(1)} FPS | ${averageMs.toFixed(1)} ms | ` +
      `${renderSize[0]}x${renderSize[1]}\n${this.gpuLine}`;
    this.accumulatedMs = 0;
    this.frameCount = 0;
  }
}

function formatGpuLine(averages: ReadonlyMap<string, number>): string {
  const passes = [...averages].map(([name, ms]) => `${name} ${ms.toFixed(2)}`);
  return `GPU ms | ${passes.join(' | ')}`;
}
