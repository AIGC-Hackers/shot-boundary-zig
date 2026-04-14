# shot-boundary-zig

这个仓库现在作为 shot-boundary detection 本地实现和验收工作区。

## Layout

- `src/`: Zig CLI、macOS MLX-C runtime、Linux ONNX Runtime backend。
- `assets/`: 本地验收资产。
- `scripts/`: 模型转换/导出、Python TF 参考输出、runtime candidate gate。
- `docs/`: 当前 runtime 计划和后续 AutoShot 记录。
- `references/`: 构建系统和依赖管理参考。

## Current Decision

macOS 主线是 Zig + MLX-C；Linux 主线是 Zig + ONNX Runtime。runtime 由编译目标决定，不通过 CLI 选择 backend。

不要再按 `../lens/PLAN.md` 里的旧结论推进。当前方向见 `docs/PLAN.md`；构建细节见 `references/build.md`。

## Zig Commands

从仓库根目录运行：

```bash
uv run --python 3.12 --with torch --with tensorflow==2.16.2 --with onnx==1.16.2 \
  scripts/export_onnx.py \
  --upstream externals/TransNetV2 \
  --output target/models/transnetv2.onnx

zig build -Doptimize=ReleaseFast run -- segment assets/333.mp4 \
  --weights target/models/transnetv2.onnx \
  --window-batch-size 2 \
  --runs 3 \
  --max-frames 20 \
  --format json > target/zig-onnx-cpu-segment-runs3-max20.json
```

runtime gate 用 `--baseline` 表示当前已接受的平台实现输出。Linux ONNX 复用同一份 JSON contract。

```bash
uv run python scripts/evaluate_runtime_candidate.py \
  --python target/python-reference-max20.json \
  --baseline target/python-reference-max20.json \
  --candidate target/zig-onnx-cpu-segment-runs3-max20.json \
  --candidate-name zig-onnx-cpu-max20-runs3 \
  --require-python-fps \
  --output-json target/runtime-candidate-zig-onnx-cpu-max20-runs3.json \
  --output-md target/runtime-candidate-zig-onnx-cpu-max20-runs3.md
```
