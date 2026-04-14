# AutoShot: TransNetV2 的 NAS 后继者

> 论文: *AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection* (CVPR 2023)
> 作者: Wentao Zhu et al. (Kuaishou + UT Austin)
> 代码: https://github.com/wentaozhu/AutoShot

## 核心结论

AutoShot 不是全新架构，而是在 TransNetV2 的构件上做 NAS 搜索得到的变体。
最终搜出的最优架构仍然是**纯 (2+1)D 分解卷积 CNN**——Transformer 被搜索淘汰 (`n_layer=0`)。

- FLOPs: 37 GMACs (TransNetV2: 41 GMACs)，少 ~10%，推理更快
- F1: 在 ClipShots 上 +1.1%，在短视频数据集 SHOT 上 +4.2%
- PapersWithCode SBD 排行榜 Top-1 (截至 2025)

## 架构对比

| 维度 | TransNetV2 | AutoShot@F1 |
|---|---|---|
| 核心卷积 | SeparableConv3d (conv2d+conv1d) | 完全相同 |
| DDCNN blocks | 3 层 × 2 blocks, n_dilation=4 | 6 blocks, n_dilation=4~5 |
| 空间卷积 | 每个膨胀分支独立 2D conv | 层 1-3 共享一个 2D conv (DDCNNV2A) |
| Transformer | 无 | SuperNet 设计了, NAS 搜掉了 |
| FrameSimilarity | 有 | 完全保留 |
| ColorHistograms | 有 | 完全保留 |
| 分类头 (fc1 → cls) | 有 | 完全保留 |

## NAS 搜索空间与最终架构

SuperNet 包含 4 种 block 变体:

- **DDCNNV2**: 原始 TransNetV2 风格, 每个膨胀分支有独立的 2D 空间卷积
- **DDCNNV2A**: 所有膨胀分支共享一个 2D 空间卷积, 时序 1D conv 各自独立
- **DDCNNV2B**: (论文中另一变体)
- **DDCNNV2C**: (论文中另一变体)

### AutoShot@F1 (最优 F1)

```
Block 0: DDCNNV2   (n_c=4F,  n_d=4)   ← 原始 TransNetV2 风格
Block 1: DDCNNV2A  (n_c=4F,  n_d=5)   ← 共享 2D conv
Block 2: DDCNNV2A  (n_c=4F,  n_d=5)
Block 3: DDCNNV2A  (n_c=4F,  n_d=5)
Block 4: DDCNNV2   (n_c=12F, n_d=5)
Block 5: DDCNNV2   (n_c=8F,  n_d=5)
Block 6: Attention1D, n_layer=0        ← NAS 选择不用 Transformer
```

### AutoShot@Precision (精度优先)

```
Block 0: DDCNNV2  (n_c=12F, n_d=4)
Block 1: DDCNNV2  (n_c=8F,  n_d=4)
Block 2: DDCNNV2B (n_d=4)
Block 3: DDCNNV2C (n_d=4)
Block 4: DDCNNV2B (n_d=5)
Block 5: DDCNNV2B (n_d=4)
Block 6: n_layer=0
```

## 从现有 TransNetV2 实现迁移所需改动

### 可完全复用 (零改动)

- SeparableConv3d (conv2d + conv1d 分解)
- FrameSimilarity 模块
- ColorHistograms 模块
- 分类头 (fc1 → cls_layer1/cls_layer2)
- 窗口滑动 / 场景后处理 / ffmpeg 解码
- correctness gate 框架

### 需要新增

1. **DDCNNV2A block 变体**: 多个膨胀分支共享一个 2D 空间卷积, 时序 1D conv 各自独立。约 30 行代码。
2. **n_dilation=5 支持**: 现有硬编码 `[1,2,4,8]`, 加一个 `dilation=16` 分支。
3. **可配置层拓扑**: 从固定 `3层×2blocks` 改为 6 个独立配置的 block, 每个指定类型和参数。
4. **权重导出脚本**: `export_autoshot_safetensors.py`, 从 PyTorch checkpoint 导出, 命名规则对齐现有 safetensors 格式。

### 风险点

- 预训练权重: `supernet_best_f1.pickle` 已在 GitHub repo 中, 另有 Google Drive 备份。格式为 Python pickle, 需写脚本转 safetensors。
- DDCNNV2B/C 的具体实现细节需要从 `linear.py` / `utils.py` 确认。
- 架构配置编码在文件名中 (如 `supernet_flattransf_3_8_8_8_13_12_0_1...`), 需对照推理代码解码各数字含义。
