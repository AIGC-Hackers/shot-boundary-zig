# Model Artifacts

Model binaries in this directory are tracked with Git LFS.

Mainline artifacts:

- `autoshot.safetensors` for macOS MLX-C
- `autoshot.onnx` for Linux ONNX Runtime
- `autoshot.safetensors.manifest.json` and `autoshot.onnx.manifest.json` export metadata

The upstream checkpoint `ckpt_0_200_0.pth` (sha256: `3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6`) is NOT checked into this repo. Download it from the AutoShot Google Drive/Baidu folder into `models/` when you need to re-export artifacts.

Explicit fallback/comparison artifact:

- `transnetv2.safetensors`

AutoShot remains the default runtime model. On macOS, TransNetV2 is selectable with `--model transnetv2` and loads from `{prefix}/models/transnetv2.safetensors` when `--weights` is omitted.

There is no committed `transnetv2.onnx` artifact in this checkout yet. Linux can still use an explicit TransNetV2 ONNX path once that artifact is regenerated, but the default installed Linux artifact is currently AutoShot.
