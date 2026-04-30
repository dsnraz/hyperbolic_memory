# v5: 推理 sim 公式对齐训练目标

## 问题（v1 错误 1）

**训练侧**（`hierarchical_loss.py:312`）：
```python
logits_alpha_parent = -alpha_matrix    # 奖励父-子外角 α 本身小
```

**推理侧**（`hyperbolic_retriver.py:697-702`）：
```python
aq = 外角(query,     parents)
an = 外角(candidate, parents)
delta = aq - an
sim_i = (1.0 + cos(delta)) * 0.5       # 奖励两个外角的差小
```

训练希望"α 小"，推理测"α 差小"。这两个目标**不等价**。

典型陷阱：query 和某个无关候选恰好对某个无关父的外角都很大（都在锥外）→ delta 近 0 → cos(delta) ≈ 1 → 高分。

## 修改

改成**直接使用候选对父的外角 α_D 来打分**，和训练目标一致：

```python
tau = 0.5    # 温度
# 对每个父 i，候选的相似度
sim_i = torch.exp(-an / tau)        # an 即 α_D，外角越小分数越高
```

`aq`（query 对父的外角）在**本版本**不参与打分（v6 会把它用作父权重门控）。

仍然走原有的父权重加权聚合，只是把单父 sim 公式换掉。

**如何和 v6 组合**：v5 负责"怎么给候选打分"，v6 负责"哪些父该算进来"。两者正交；推荐一起用。

## 涉及文件

- `model/retrievers/hyperbolic_retriver.py`

## 修改位置

`MultiParentAngularHyperbolicRetriever._weighted_cosine_angular_opposite_score`（约 678-703 行）。

## 如何应用

```bash
python algorithms/apply.py v5_inference_sim
```

## 如何验证

对同一查询，对比 CATEGORY 层 top-10 的得分分布：

- 修复前（问题 1 日志）：0.9907 ~ 0.9989，动态范围 0.008
- 修复后：预期动态范围显著扩大（> 0.1），**正确候选**（如 `category_2 Social Issues`）排名前列

诊断打印保留在代码里（可自行启用）：
```python
# 在 _weighted_cosine_angular_opposite_score 里
print(f"  an={an.tolist()}, sim_i={sim_i.tolist()}")
```

## 温度参数 τ 怎么选

- τ 大（例如 1.0）：分数接近 `1 - α_D`，线性风味，动态范围小。
- τ 小（例如 0.2）：`exp(-α_D/τ)` 形状陡峭，区分度大但低分饱和到 0 快。
- **推荐 τ = 0.3 ~ 0.5**，作为默认值放 0.5。

可通过构造参数传入：
```python
MultiParentAngularHyperbolicRetriever(..., sim_temperature=0.5)
```

## 需不需要重训？

**不需要**。纯推理改动。

## 不影响其他版本

只改 `hyperbolic_retriver.py` 里 `_weighted_cosine_angular_opposite_score` 这一个函数。

- 和 v6（父权重门控）正交，可组合。
- 和 v7（DOMAIN 改测地）正交，可组合。
- 不触碰训练侧。
