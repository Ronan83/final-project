# Performance log / 性能记录

GPU: NVIDIA GeForce RTX 5070 Laptop · Chrome (WebGPU) · "benchmark view" camera

| Date | Version | Render scale | Resolution | FPS | Frame time |
|---|---|---|---|---|---|
| 2026-11-09 | Baseline: procedural rocks (6-octave 3D FBM, 3×3 cell search) | 0.5 | 1280×673 | 27.5 | 36.4 ms |
| 2026-11-09 | Baseline: procedural rocks (6-octave 3D FBM, 3×3 cell search) | 1.0 | 2560×1347 | 8.0 | 124.2 ms |
| 2026-11-18 | Baseline v2: 4 rock species + random yaw, strictly conservative far-field bound | 0.5 | 1280×673 | 15.8 | 63.1 ms |
| 2026-11-18 | Baseline v2: 4 rock species + random yaw, strictly conservative far-field bound | 1.0 | 2561×1347 | 4.6 | 219.4 ms |

## GPU pass timing (timestamp queries) / GPU 各 pass 耗时

Measured per pass with WebGPU timestamp queries, averaged over 30 frames.
用 WebGPU timestamp query 逐 pass 测量，取 30 帧平均。

| Date | Rock SDF | Render scale | Resolution | scene (ms) | blit (ms) | FPS |
|---|---|---|---|---|---|---|
| 2026-11-22 | Procedural (teacher, 6-octave fbm3) | 0.5 | 1280×673 | 58.65 | 0.07 | 17.0 |
| 2026-11-22 | Neural baseline (39→64→64→1 SiLU MLP, scalar loops, storage-buffer weights) | 0.5 | 1280×673 | 64.77 | 0.07 | 15.4 |
