# v7: DOMAIN 层改走测地线（修复检索树断链）

## 问题

实验证据：同一查询、同一训练好的 projector：

- **问题 1（LGBTQ）** 用 hybrid (DOMAIN 走 O 内角)：top-10 得分 0.3757 ~ 0.3832，动态范围 **0.0076**，真父 `domain_2 (Social Issues)` 不在 top-10。
- **问题 2（Education）** 用纯测地线：top-10 得分 0.5970 ~ 0.6275，动态范围 **0.0305**，真父全部在前列。

**注意：这不是双曲几何本身的缺陷。**（你指出了：原点切平面保角，O 内角不会因为壳浅而畸变。）

真正原因：
- **训练不充分**（灾难性遗忘，见 v3）：137 个 DOMAIN 节点没被训练到方向上充分分散。
- **推理公式饱和**（见 v1 错误 4 + v5）：`(1+cos δ)/2` 在小 δ 区域二阶压缩。

在还没修 v3/v5 的情况下，**最便捷的止血方案**是让 DOMAIN 层绕开"多父外角/O 内角"路径，改走测地线——因为测地线即使在弱训练的 projector 上也能给出合理的动态范围。

## 修改

`HybridHyperbolicRetriever` 的默认分界层从 `KEYWORD` 改到 `DOMAIN`：

```python
# hyperbolic_retriver.py:769
# 原：
self._hybrid_scoring_geodesic_from_level = HierarchyLevel.KEYWORD
# 改：
self._hybrid_scoring_geodesic_from_level = HierarchyLevel.DOMAIN
```

逻辑：`HybridHyperbolicRetriever._similarity` 判断 `index(node.level) < index(boundary)` 时走角度，否则走测地。boundary 改为 DOMAIN（索引 0），**所有层级的 node.level 索引都不小于 0**，全部走测地线。

**简化：本版本等价于把 HybridHyperbolicRetriever 退化为 GeodesicHyperbolicRetriever**。保留 hybrid 结构只是为了方便对比不同 boundary 的影响。

## 更精细的替代方案

如果只想让 DOMAIN 层走测地、**CATEGORY/KEYWORD 仍走角度**，把 boundary 设为 `CATEGORY`：

```python
self._hybrid_scoring_geodesic_from_level = HierarchyLevel.CATEGORY
```

- `index(DOMAIN)=0 < index(CATEGORY)=1` → DOMAIN 层走角度…… 等等，**条件反了**。

检查实际代码（`hyperbolic_retriver.py:840`）：
```python
if n_idx < b_idx:
    print("使用多父外角检索")
    return self._angular._similarity(query_h, node)
return GeodesicHyperbolicRetriever._similarity(self, query_h, node)
```

所以 `n_idx < b_idx` 时走角度。
- boundary=CATEGORY (idx=1)：DOMAIN (idx=0) < 1 → 走角度；CATEGORY (idx=1) 不 < 1 → 走测地。
- boundary=DOMAIN (idx=0)：无层级 idx < 0 → 全部走测地。
- boundary=KEYWORD (idx=2)：DOMAIN、CATEGORY 走角度；KEYWORD、DIALOGUE 走测地。（当前默认）

**要让 DOMAIN 走测地、CATEGORY/KEYWORD 走角度**，现有的 `< boundary` 结构做不到（DOMAIN 永远是最小索引）。需要改逻辑：

```python
# v7: 反向逻辑：只让 boundary 及之上走测地，boundary 以下走角度
if n_idx > b_idx:
    return self._angular._similarity(query_h, node)
return GeodesicHyperbolicRetriever._similarity(self, query_h, node)
```

这样：
- boundary=DOMAIN：DOMAIN 走测地；CATEGORY、KEYWORD、DIALOGUE 走角度。
- boundary=CATEGORY：DOMAIN、CATEGORY 走测地；KEYWORD、DIALOGUE 走角度。
- boundary=KEYWORD：只有 DIALOGUE 走角度。

**v7 本版本采用这套反向逻辑，默认 boundary=DOMAIN**（DOMAIN 走测地；其它层走角度）。

## 涉及文件

- `model/retrievers/hyperbolic_retriver.py`

## 修改位置

`HybridHyperbolicRetriever` 的 `_similarity`（约 827-843 行）和默认 boundary（约 769 行）。

## 如何应用

```bash
python algorithms/apply.py v7_domain_geodesic
```

然后用 `retriever_type=hyperbolic_angular_geodesic_hybrid` 运行：

```bash
python -m model.llm_inference.run --retriever_type hyperbolic_angular_geodesic_hybrid
```

## 如何验证

对问题 1（LGBTQ），观察 DOMAIN 层 top-10：

- 修复前：全是艺术/户外类 domain，动态范围 0.0076，`domain_2` 缺席。
- 修复后：预期出现 `domain_2 (Social Issues)`，动态范围 > 0.03。

## 需不需要重训？

**不需要**。纯推理改动。

## 不影响其他版本

本修改只触碰 `hyperbolic_retriver.py` 的 `HybridHyperbolicRetriever._similarity` 和默认 boundary。

- 和 v5（sim 公式）、v6（父权重门控）都改 MultiParentAngularHyperbolicRetriever，v7 改 Hybrid——不冲突。
- 但三者都改同一文件，**建议最后 apply 时使用合并版**（见 README）。
