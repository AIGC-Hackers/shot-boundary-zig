# shot-boundary-zig

这个仓库现在作为 shot-boundary detection 本地实现和验收工作区。

## Layout

- `src/`: Zig CLI 和 macOS MLX-C runtime。
- `assets/`: 本地验收资产。
- `scripts/`: 模型转换/导出、Python TF 参考输出、runtime candidate gate。
- `docs/`: 当前 runtime 计划和后续 AutoShot 记录。
- `references/`: 构建系统和依赖管理参考。

## Current Decision

macOS 主线现在是 Zig + MLX-C。Zig 版已经输出完整 `segment` JSON，并通过同一套 Python correctness/performance gate。

不要再按 `../lens/PLAN.md` 里的旧结论推进。当前方向见 `docs/PLAN.md`；构建细节见 `references/build.md`。

## Zig Commands

从仓库根目录运行：

```bash
zig build -Doptimize=ReleaseFast run -- segment assets/333.mp4 \
  --weights target/models/transnetv2.safetensors \
  --window-batch-size 2 \
  --runs 5 \
  --format json > target/reports/zig-mlx-segment-runs5.json
```

runtime gate 用 `--baseline` 表示当前已接受的平台实现输出。Linux ONNX 阶段继续复用同一份 JSON contract。

```bash
uv run python scripts/evaluate_runtime_candidate.py \
  --python target/reports/python-reference.json \
  --baseline target/reports/python-reference.json \
  --candidate target/reports/zig-mlx-segment-runs5.json \
  --candidate-name zig-mlx-runs5 \
  --require-python-fps \
  --output-json target/reports/runtime-candidate-zig-mlx-runs5.json \
  --output-md target/reports/runtime-candidate-zig-mlx-runs5.md
```
