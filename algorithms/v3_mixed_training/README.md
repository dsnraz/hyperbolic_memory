# v3: 混合批次训练 — 消除灾难性遗忘

## 问题

`train.py:465`：
```python
for level_idx in [1, 2, 3]:
    self.train_level_pair(level_idx)
```

三个层级对**严格顺序**训练，且步数失衡（默认 500 / 4000 / 8000）。

**单投射器共享**：`Hyperbolic_projector` 只有一个，三阶段共用同一套 `phi` MLP。phase 1 学到的 DOMAIN 表示，在 phase 2 和 phase 3 里完全没有监督信号——phase 2 批里没有 DOMAIN 节点；phase 3 批里也没有 DOMAIN 节点。**12000 步后，phase 1 的成果已被覆盖**。

**和原点切平面保角性的关系**（回应你的反驳）：原点切平面是保角的，DOMAIN 节点从 O 看出去的 O 内角**理论上**不会因为壳浅而畸变。所以 DOMAIN 层 top-10 动态范围只有 0.008，**真正原因不是几何，而是 137 个 DOMAIN 节点在训练后没被充分散开在方向上**——遗忘是最可能的嫌疑犯。

## 你的担忧

> 我的步数设置是考虑了每层节点数目设置的，感觉如果混合训练可能不太好排布？具体怎么混？

你的顾虑合理。原始步数 `{1: 500, 2: 4000, 3: 8000}` 按层级"复杂度"递增分配。混合训练要**保持这个比例**，只是把"先看完 500 步 phase 1、再看完 4000 步 phase 2"改成"**按比例随机交错**"。

**采样方案**：

设目标步数为 12500（= 500+4000+8000）。每步按概率挑一个层级对：
- `p(DOMAIN→CATEGORY) = 500/12500 = 0.04`
- `p(CATEGORY→KEYWORD) = 4000/12500 = 0.32`
- `p(KEYWORD→DIALOGUE) = 8000/12500 = 0.64`

期望下，每层级对被采样的次数和原始分配一样。差别只在**时间分布**：原来是 500 步 phase 1 → 4000 步 phase 2 → 8000 步 phase 3；现在是三者穿插，每一步都有非 0 概率看到任何一阶。

## 修改

两处改动：

1. `hierarchical_dataset.py`：
   - 给 `SubtreeDataset` 加 `level_pair_weights` 参数（dict: 层级对 → 权重）。
   - `_sample_level_pair_with_coverage` 在传入权重时改为加权随机采样（`random.choices`），否则保持原覆盖式轮转。

2. `train.py`：
   - `TrainConfig` 加 `mixed_training: bool = False` 开关。
   - 加 `train_mixed()` 方法：
     - 调 `extract_nodes_from_store(vector_store, level_pair_index=None)` 载入全部四层节点。
     - 按 `iterations_map` 算权重，创建 `level_pair=None` 的 `SubtreeDataset(level_pair_weights=...)`。
     - 一次性跑完总步数。
   - `train()` 根据 `mixed_training` 分派到 `train_mixed` 或 `train_sequential`。

## 涉及文件

- `model/hyperbolic_utils/hierarchical_dataset.py`
- `model/hyperbolic_utils/train.py`

## 如何应用

```bash
python algorithms/apply.py v3_mixed_training
```

然后跑训练时加开关：

```bash
python -m model.hyperbolic_utils.train --mixed_training
# 或直接改 TrainConfig 默认值
```

## 如何验证

1. **训练过程中的 level_pair 统计**：修复后会打印每 500 步内三个层级对的采样次数，应接近预期比例（4%/32%/64%）。
2. **各 phase 快照对比**：分别保存早期和晚期 checkpoint，跑同一个 DOMAIN 层检索。修复后，**随训练进度 DOMAIN 检索质量应持续改善，而不是在早期达到峰值后退化**。
3. **最终 DOMAIN top-10 动态范围**：应该从 0.008 量级恢复到 0.05+ 量级（和 geodesic 相当）。

## 需不需要重训？

**需要全部重训**。这是训练策略的根本改变。

## 不影响其他版本

本修改：
- 给数据集加了可选 `level_pair_weights` 参数，不影响旧接口调用。
- 给 `TrainConfig` 加了可选 `mixed_training` 开关，默认 False 保留旧行为。
- 与 v1/v2/v8/v9 都兼容；apply 顺序：先 v3，再用 `python algorithms/apply.py v1_softplus_fix` 叠加 v1 的 projector 修改。

## 采样细节

加权采样的实现（`hierarchical_dataset.py` 新增）：

```python
def _sample_level_pair_weighted(self) -> Tuple[str, str]:
    """按权重采样层级对。"""
    pairs = list(self.samplers.keys())
    weights = [self._level_pair_weights[p] for p in pairs]
    return random.choices(pairs, weights=weights, k=1)[0]
```

当 `level_pair_weights` 为 None 时保持原有覆盖式轮转；设置了权重就切换到加权采样。
