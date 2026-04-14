# Deferred Work Log

- 2026-04-13: Python/Rust alignment passes on `assets/333.mp4`, but Rust Candle CPU inference is slower than TensorFlow. This is historical context only; do not spend more macOS effort on Candle unless it is needed for debugging.
- 2026-04-13: `decode_video_rgb24` shells out to `ffmpeg` to match the official Python reference's default vsync/scaler behavior. If this project needs library-only embedding without an `ffmpeg` binary, add an explicit frame-duplication policy before removing the command dependency.
- 2026-04-14: Linux CPU/CUDA remains unevaluated. Do not assume the macOS MLX result transfers; rerun the same runtime candidate gate on Linux before choosing ORT/CUDA/TensorRT/libtorch or another backend.
- 2026-04-14: Zig MLX-C now passes the full macOS candidate gate and the direct Rust MLX comparison gate. Before deleting `rust/`, check for external scripts that still call Rust-only CLI options or old report filenames.
- 2026-04-14: MLX-C is now managed through gitignored `externals/` by the Zig build. Before CI or packaging, make sure CMake is available and decide whether the MLX-C build cache should be persisted.
- 2026-04-14: `scripts/evaluate_runtime_candidate.py` still exposes the migration-era `--baseline-rust` option. Generalize that baseline before documenting Linux ONNX benchmark commands.
