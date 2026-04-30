# v8: 径向自由度软化（接受用户反驳后的修订方案）

## 你的反驳

> 直接对每个节点施加层级约束，不会导致同一层级的所有节点径向深度雷同吗？这比质心约束还要硬约束吧。

**完全正确**。我在 v2 诊断里建议的"每节点 d_v 点目标"确实会把同层所有节点的径向深度钉死，反而不如现有的质心约束。这一版撤回那个方案。

## 真正的问题

问题不在"缺少层级约束"，而在 `hyperbolic_projector.py:54-56` 的**硬投影**：

```python
target_norm = (self.alpha + (self.beta - self.alpha) * d_v) * R
z_hat_E = (target_norm / z_star_norm) * z_star_v_32    # 硬改写模长
```

这一步**强制把每个节点的模长改写**为 `target_norm`。加上 `d_v` 在同层会收敛到同一值，实际等于把同层节点全部钉到同一个球壳。

`loss_centroid` 本来只约束**层级平均**，允许层内径向散布——但硬投影把这种散布权利剥夺了。

## 修改方案

**去掉硬投影**，改为**软 MSE**（保持层级模长约束但允许偏离）：

**核心改动**：

```python
# 原：z_hat_E = (target_norm / z_star_norm) * z_star_v_32     # 硬改写
# 新：z_hat_E = z_star_v_32                                    # 保留 phi 原模长
```

**默认行为**（最小改动）：
- 硬投影去掉
- `loss_centroid` **保留不变**（layer 平均深度拉到 target）
- `lambda_soft_norm = 0` 和 `lambda_rank = 0`（可选，默认关闭）

这样默认就是你反驳里想要的："用质心约束管层级均值，层内径向允许自由散布"——不施加任何点目标。

**可选的加强项**（如实验发现层崩塌或层间重叠再打开）：

| 参数 | 作用 | 默认 | 建议何时启用 |
|---|---|---|---|
| `lambda_soft_norm` | 软 MSE：每节点 ‖z‖ 拉向 d_v 预测的 target_norm | 0 | 若观察到层内 ‖z‖ 方差过大（跨层重叠）时启用（0.01 起）|
| `lambda_rank` | 层间 margin：parent_centroid 必须比 child_centroid 浅 `margin` 以上 | 0 | 若观察到层间均值顺序错乱时启用（0.5 起）|
| `rank_margin` | margin 大小 | 0.1 | 与 `lambda_rank` 配套 |

> **注意**：`lambda_soft_norm > 0` 会重新引入 per-node 的点目标（你反驳过的那种），但小 λ（0.01）下只是弱偏置，不会把层内钉死。仍应视作"实验性启用"。

## 进一步：层间顺序 margin（可选，`loss_rank`）

若观察到层间深度顺序错乱（例如训练后 DOMAIN 重心反而比 CATEGORY 深），加一个硬顺序约束：

```python
# 只在同一批次包含相邻两层时生效
margin = 0.1
loss_rank = F.relu(centroid_parent - centroid_child + margin)
```

本版本实现了这个可选项，默认权重 0.0（不启用）；置 0.5 即开启。

## 涉及文件

- `model/hyperbolic_utils/hyperbolic_projector.py`（去硬投影）
- `model/hyperbolic_utils/hierarchical_loss.py`（加 soft MSE + 可选 rank margin）

## 如何应用

```bash
python algorithms/apply.py v8_depth_range
```

## 如何验证

训练后对每层节点统计 `||z_H||` 的均值和标准差：

```python
from collections import defaultdict
import numpy as np
stats = defaultdict(list)
for node in all_nodes:
    z_H = projector(node.embedding)[1]
    stats[node.level.name].append(torch.norm(z_H).item())

for lvl, norms in stats.items():
    print(f"{lvl}: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}")
```

**期望**：
- 修改前：同层 `std < 0.001`（被硬投影钉死）
- 修改后：`std ≈ 0.02 ~ 0.10`，允许层内径向分布
- 层间 mean 保持递增（`mean(DOMAIN) < mean(CATEGORY) < mean(KEYWORD) < mean(DIALOGUE)`）

## 需不需要重训？

**需要**。projector 和损失形式都变了。

## 和其他版本的关系

- 和 v1（softplus）都改 `hyperbolic_projector.py`。apply v8 后，手动把 v1 的 `softplus` 反解合并进 v8 的 projector（只是 `__init__` 里一行）。
- 和 v9（全链条）都改 `hierarchical_loss.py`。apply 顺序：先 v8 再 v9，v9 会在 v8 的 loss 文件基础上增量添加。

## 为什么不用"区间目标"（interval target）？

我在早期设计时考虑过"让每个节点的 target_norm 落在层级区间 [0.05R, 0.15R]"。但实际上它退化为"soft MSE 的阈值化版本"：区间内零损失、区间外 MSE。写起来更复杂、超参更多，效果本质相同。所以本版本用 soft MSE 简化。
