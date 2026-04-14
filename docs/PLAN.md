# Runtime Plan

## Current Decision

截至 2026-04-14，macOS 路径是 Zig + MLX-C。

Zig CLI 已经输出完整 `segment` JSON，通过共享 Python correctness/performance gate，也通过了直接对比 Rust MLX batch2/runs5 的 gate。旧 Rust 阶段记录只保留为迁移背景，不再指导新的 runtime 工作。

`rust/` 只保留到外部脚本引用检查完成。确认没有 Rust-only 子命令或旧报告文件名依赖后，删除它，不要维护两套 runtime。

## Runtime Boundary

- Zig 负责可移植 CLI、backend selection、视频 decode/windowing 编排和标准化 JSON 输出。
- backend 实现是同一 CLI contract 后面的平台相关边界。
- Python 只负责模型转换、导出、reference output 生成和 acceptance gate。
- MLX-C 构建细节在 `references/build.md`；本地 MLX-C 产物放在 gitignored `externals/`，不要再放 `.scratch/`。

核心 contract 应保持小：backend 接收标准化 TransNetV2 输入并返回 probabilities/timings；CLI 在边界处理 I/O、validation 和 report formatting。

## Next Milestone: Linux ONNX Backend

1. 从现有 source weights 导出稳定 ONNX model，转换脚本放在 `scripts/`。
2. 用 ONNX Runtime C API 增加 Linux backend。先跑 CPU execution provider；CPU correctness 和 benchmark gate 通过后再接 CUDA。
3. 复用现有 `segment` JSON 形状和 acceptance gate，让 Linux 结果能和 macOS MLX-C、Python reference 对齐。
4. 把 `scripts/evaluate_runtime_candidate.py` 的迁移期 `--baseline-rust` 语义泛化成 runtime baseline，不要让 Rust 文件名继续成为跨平台 gate 的 API。
5. 增加 build flags 和文档来定位 ORT headers/libs，不把 host-specific path 写进 source。
6. 用同一个 acceptance asset 跑 benchmark，只记录当前 gate 结果，不再新增一组长期维护的 Rust 阶段性能日记。

## Deferred

多平台 backend 稳定后：

- 补 `docs/AUTOSHOT.md` 里的 AutoShot 任务；
- rename 项目；
- 删除临时 Rust 实现和只为迁移期存在的旧报告命名。
