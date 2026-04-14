# Runtime Plan

## Current Decision

截至 2026-04-14，macOS 路径是 Zig + MLX-C。

Zig CLI 已经输出完整 `segment` JSON，并通过共享 Python correctness/performance gate。当前仓库只保留 Zig runtime 主线。

## Runtime Boundary

- Zig 负责可移植 CLI、视频 decode/windowing 编排和标准化 JSON 输出。
- 平台 runtime 实现是同一 CLI contract 后面的边界：macOS 走 MLX-C，Linux 下一步走 ONNX Runtime。
- Python 只负责模型转换、导出、reference output 生成和 acceptance gate。
- MLX-C 构建细节在 `references/build.md`；本地 MLX-C 产物放在 gitignored `externals/`，不要再放 `.scratch/`。

核心 contract 应保持小：平台 runtime 接收标准化 TransNetV2 输入并返回 probabilities/timings；CLI 在边界处理 I/O、validation 和 report formatting。

## Next Milestone: Linux ONNX Runtime

1. 从现有 source weights 导出稳定 ONNX model，转换脚本放在 `scripts/`。
2. 用 ONNX Runtime C API 增加 Linux runtime。先跑 CPU execution provider；CPU correctness 和 benchmark gate 通过后再接 CUDA。
3. 复用现有 `segment` JSON 形状和 acceptance gate，让 Linux 结果能和 macOS MLX-C、Python reference 对齐。
4. 增加 build flags 和文档来定位 ORT headers/libs，不把 host-specific path 写进 source。
5. 用同一个 acceptance asset 跑 benchmark，只记录当前 gate 结果，不新增长期维护的临时性能日记。

## Deferred

多平台 runtime 稳定后：

- 补 `docs/AUTOSHOT.md` 里的 AutoShot 任务；
