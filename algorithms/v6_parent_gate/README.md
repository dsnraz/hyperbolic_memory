# v6: 父权重加 "query 在父锥内" 门控

## 问题（v1 错误 2）

现有权重方案（`hyperbolic_retriver.py:648-676`）：
- `weight_by_parent_origin_geodesic`：父到原点的测地距离越远，权重越大。
- `weight_by_parent_anchor_geodesic`：父到锚点（query/node）的测地距离越近，权重越大。

**问题**：所有同层父节点被训练到**同一个深度壳**上（CATEGORY 所有父到原点的距离都 ≈ 0.3R），第一个权重对 137 个 CATEGORY 父几乎一视同仁，**无效加权**。第二个权重用"测地距离近"代表"相关"，但几何上测地近不等于"落在父锥内"——两者不同概念。

`keyword_13`（support）有 30+ 父。当前权重约等于均匀平均，正确父的信号被另 29 个无关父稀释，**信噪比 1/30**。

## 修改

用**外角门控**：query 对父 i 的外角 `α_Q^i` 若落在父的半孔径 `K(parent_i)` 内，这个父"管得到 query"，给高权重；落在锥外则快速衰减。

```python
def _parent_aggregation_weights(self, parents_h, query_h, node_h, curv):
    par = parents_h.detach().float().cpu()
    query = _hyperbolic_spatial_row(query_h).unsqueeze(0).repeat(par.shape[0], 1)

    # query 对每个父的外角
    alpha_Q = L.cone_vertex_exterior_angle_vectors(par, query, curv)  # (P,)
    # 每个父的半孔径
    K = L.half_aperture_vectors(par, curv)                             # (P,)

    # ratio < 1 → query 在锥内；ratio > 1 → 在锥外
    ratio = alpha_Q / (K + self.weight_eps)

    # 锥内权重 ≈ 1；锥外按 (ratio-1)/gate_temp 指数衰减
    gate_temp = self.cone_gate_temperature    # 新参数，默认 0.2
    w_gate = torch.exp(-torch.clamp(ratio - 1.0, min=0.0) / gate_temp)

    # 和原有"父到原点/锚点"权重逐元素相乘
    w_origin = ...  (原逻辑)
    w_anchor = ...  (原逻辑)
    w = w_gate * w_origin * w_anchor
    return self._normalize_parent_weights(w)
```

## 几何解释

- **锥内的父 i**：`α_Q^i < K_i`，意思是 query 落在这个父的蕴涵锥里（父"覆盖" query）→ 这个父的判断可靠，给高权重。
- **锥外的父 i**：query 跑到了锥外，这个父"管不到" query → 它对 query 的打分没参考价值，权重衰减到近 0。
- 30 个父里通常只有 1-2 个真正"覆盖" query，剩下的被门控剔除。信噪比 1/30 → 1/2。

## DOMAIN 层的特殊处理

DOMAIN 节点的半孔径 `K ≈ π/2` 饱和（见 v2 诊断讨论）。对 DOMAIN 无父，`_parent_aggregation_weights` 不被调用，本版本无影响。推荐配合 v7（DOMAIN 改测地线），完全绕开 DOMAIN 层的角度路径。

## 涉及文件

- `model/retrievers/hyperbolic_retriver.py`

## 修改位置

`MultiParentAngularHyperbolicRetriever._parent_aggregation_weights`（约 648-676 行）。

## 如何应用

```bash
python algorithms/apply.py v6_parent_gate
```

## 如何验证

对有多父的候选（如 `keyword_13` support，有 30+ 父）打印权重分布：

```python
# 在 _parent_aggregation_weights 里
print(f"alpha_Q: {alpha_Q[:10].tolist()}")
print(f"K       : {K[:10].tolist()}")
print(f"w_gate  : {w_gate[:10].tolist()}")
print(f"final w : {w[:10].tolist()}")
```

**期望观察**：大部分 `alpha_Q > K`（query 不在这些父的锥内）→ `w_gate` 快速衰减到 0；少数 `alpha_Q < K` 的父拿高权重，且这些父应该是语义上最相关的。

## 如何和 v5 组合

v5 改"给候选打什么分"，v6 改"用哪些父来打分"。两者正交。

推荐一起用（分别 apply 两次，因为改的是同一文件的不同函数）：

```bash
python algorithms/apply.py v5_inference_sim
python algorithms/apply.py v6_parent_gate     # 警告：会覆盖 v5 的备份
```

**注意**：两个版本都改 `hyperbolic_retriver.py`，直接串行 apply 会让第二次 apply 丢失第一次的改动。**请用合并版目录** `v5_v6_combined/`（如果下面提供的话），或手动合并两段改动——两段改动所在的函数不冲突，手动合并很直接。

## 需不需要重训？

**不需要**。纯推理改动。

## 不影响其他训练侧版本

推理侧单点修改，不触碰训练流程。和 v1/v2/v3/v8/v9 都可叠加。
