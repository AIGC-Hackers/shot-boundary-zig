# Deferred Work Log

- 2026-04-13: `decode_video_rgb24` shells out to `ffmpeg` to match the official Python reference's default vsync/scaler behavior. If this project needs library-only embedding without an `ffmpeg` binary, add an explicit frame-duplication policy before removing the command dependency.
- 2026-04-14: Linux CPU/CUDA remains unevaluated. Do not assume the macOS MLX result transfers; rerun the same runtime candidate gate on Linux before choosing ORT/CUDA/TensorRT/libtorch or another backend.
- 2026-04-14: MLX-C is now managed through gitignored `externals/` by the Zig build. Before CI or packaging, make sure CMake is available and decide whether the MLX-C build cache should be persisted.
