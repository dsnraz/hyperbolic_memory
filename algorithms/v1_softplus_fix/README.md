# v1: softplus 反解 — 让初始曲率真正是 0.1

## 问题

`hyperbolic_projector.py:13,31`：

```python
self.c = nn.Parameter(torch.tensor([float(curvature)], dtype=torch.float32))  # 存 0.1
...
curr_c = torch.nn.functional.softplus(self.c)  # 实际用 softplus(0.1) ≈ 0.7444
```

`initial_curvature=0.1` 被当作"参数原始值"，经过 `softplus` 后得到的有效曲率是 0.7444，是你预期的 7.4 倍。曲率半径 `R = 1/√c` 因此从预期的 3.16 塌缩到 1.16，**所有层级的深度目标（0.1R / 0.3R / …）被等比例压缩**，全部训到远比预期浅的位置。

## 修改

反解 softplus：若想 `softplus(c_raw) = 0.1`，则 `c_raw = log(e^0.1 - 1)`。

```python
import math
# 反解 softplus
c_raw_init = math.log(math.expm1(float(curvature)))
self.c = nn.Parameter(torch.tensor([c_raw_init], dtype=torch.float32))
```

对 `curvature=0.1`，`c_raw_init ≈ -2.252`。前向里 `softplus(-2.252) ≈ 0.1` 就是目标。

## 涉及文件

- `model/hyperbolic_utils/hyperbolic_projector.py`（只改第 13 行附近）

## 如何应用

```bash
python algorithms/apply.py v1_softplus_fix
```

## 如何验证

训练第一步打印 `stats['curvature']`（`train.py:364` 已经记录）。

- 修复前：第一步应该打印 `curvature: 0.7444`
- 修复后：第一步应该打印 `curvature: 0.1000`

快速验证脚本：

```python
from model.hyperbolic_utils.hyperbolic_projector import Hyperbolic_projector
import torch.nn.functional as F

p = Hyperbolic_projector(input_dim=384, hidden_dim=256, curvature=0.1)
effective_c = F.softplus(p.c).item()
print(f"effective curvature = {effective_c:.6f} (expected 0.1)")
```

## 需不需要重训？

**需要**。曲率的改变会影响所有训练好的节点在双曲空间里的位置，旧的 checkpoint 不再有效。

## 期望影响

- `R` 从 ≈1.16 变为 ≈3.16，变大 2.73 倍。
- DOMAIN 深度 `0.1R` 从 0.116 变为 0.316，节点壳离原点更远。
- 半孔径公式 `K(x) = asin(2·min_radius / (||x||·√c))` 的 `||x||·√c` 值从 0.1（被 clamp 到 π/2 饱和）变为真正的 0.1（仍饱和，因为没开 entailment loss 所以不相关）。
- 最关键：各层的**深度分布变宽**，角度分辨率（切空间意义上）应该更稳定。

## 不影响其他版本

本修改只改 `hyperbolic_projector.py` 构造函数的一行。不与 v2-v9 冲突，可叠加应用。
