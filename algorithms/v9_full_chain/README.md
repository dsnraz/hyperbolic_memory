# v9: 全链条（祖-孙）外角约束

## 你的建议

> 可以分析一下，是否可以进行全链条约束，而非只是单层级对父子节点之间进行

采纳。当前训练只对相邻层级对（DOMAIN→CATEGORY、CATEGORY→KEYWORD、KEYWORD→DIALOGUE）做父子外角损失，**没有直接约束祖-孙关系**。即便相邻层级都训练好了，祖-孙的几何一致性不是自动成立的——双曲锥**没有传递律**（child ∈ parent's cone 且 parent ∈ grandparent's cone 不蕴含 child ∈ grandparent's cone）。

## 方案

加入**跨层级对**训练：
- 新跨层对 1：`(DOMAIN, KEYWORD)` — 祖孙关系（隔一层）
- 新跨层对 2：`(CATEGORY, DIALOGUE)` — 祖孙关系
- 新跨层对 3（可选）：`(DOMAIN, DIALOGUE)` — 曾祖-重孙（隔两层），通常用不上

对每个跨层对，构造"祖父 → 孙子"的父子关系表（两跳合并），然后用**同一个** `HierarchicalAngularContrastiveLoss` 跑，只是 batch 的父是祖父、子是孙子。

几何意义：把原来分三段训练的 "DOMAIN 锥轴 → CATEGORY → KEYWORD" 直链，用一个端到端的 α(DOMAIN → KEYWORD) 小损失**直接约束**。和质心正则一起，能让 O → DOMAIN → CATEGORY → KEYWORD → DIALOGUE 的路径在几何上保持一致。

## 涉及文件

- `model/hyperbolic_utils/hierarchical_dataset.py`（加祖孙采样支持）
- `model/hyperbolic_utils/train.py`（加 `--chain_training` 开关）

## 修改细节

### 1. dataset 新增

在 `hierarchical_dataset.py` 里：

```python
SKIP_LEVEL_PAIRS = [
    ("DOMAIN", "KEYWORD"),      # 跨 1 层
    ("CATEGORY", "DIALOGUE"),   # 跨 1 层
]

def build_skip_level_parent_to_children(
    ancestor_nodes, descendant_nodes, middle_nodes,
):
    """构造祖父→孙子映射。"""
    middle_id_to_node = {m.id: m for m in middle_nodes}
    descendant_id_set = {d.id for d in descendant_nodes}
    ancestor_to_descendants = {}
    for a_idx, a in enumerate(ancestor_nodes):
        grand_ids = set()
        for mid_id in a.child_ids:
            m = middle_id_to_node.get(mid_id)
            if m:
                for gid in m.child_ids:
                    if gid in descendant_id_set:
                        grand_ids.add(gid)
        ancestor_to_descendants[a_idx] = grand_ids
    return ancestor_to_descendants

class SkipLevelSubtreeSampler(SubtreeSampler):
    """和 SubtreeSampler 一样，但 parent_to_children 用祖孙映射。"""
    def __init__(self, ancestor_nodes, descendant_nodes, middle_nodes, **kwargs):
        super().__init__(
            parent_nodes=ancestor_nodes,
            child_nodes=descendant_nodes,
            **kwargs,
        )
        # 覆盖 parent_to_children 映射
        # ...
```

### 2. train.py 新增

`TrainConfig`:
```python
chain_training: bool = False
chain_weights: Optional[Dict[Tuple[str, str], float]] = None   # 跨层对权重
```

新方法 `train_chain_mixed()`：
- 调用 `extract_nodes_from_store(None)` 取全部节点
- 为每个 SKIP_LEVEL_PAIR 构建 `SkipLevelSubtreeSampler`
- 合并到 `SubtreeDataset` 的 samplers 里
- 按给定权重（默认均匀）混合采样

默认权重：
- 相邻父子对：0.5（保留原有训练主力）
- 祖孙对：0.5（等权重加强全链）

## 和 v3 的关系

v9 需要 v3 的混合采样机制才有意义（三个相邻对 + 两个跨层对 = 5 个 level_pair 交错训练）。本版本**包含** v3 的混合机制（自包含），不需要先 apply v3。

但如果你已 apply v3，v9 的 `hierarchical_dataset.py` 和 `train.py` 会覆盖 v3 的版本。需要合并两者（v9 提供的文件已经内含 v3 的功能，可直接覆盖）。

## 如何应用

```bash
python algorithms/apply.py v9_full_chain
# 训练时：
python -m model.hyperbolic_utils.train --chain_training
```

## 如何验证

**指标 1**：训练后，对数据集里采一批"祖父-孙子"对，计算 α(祖父 → 孙子) 的分布。

- 无 v9：分布可能有长尾（部分孙子跑到祖父锥外）
- 有 v9：分布更集中在 0 附近

**指标 2**：推理时，计算完整路径 O → DOMAIN → CATEGORY → KEYWORD → DIALOGUE 的"角度和"，看是否接近路径最短值。

**指标 3**：检索 recall@k 在 DIALOGUE 层的绝对值。全链条一致性好应该让整条检索树断链概率降低，最终 DIALOGUE 命中率提高。

## 需不需要重训？

**需要全部重训**。

## 和其他版本的关系

- 包含 v3 混合采样机制，可独立使用。
- 若已 apply v8（软 projector）或 v2（mask 修复），手动把相应改动合并进 v9 的 `hierarchical_dataset.py` / `train.py`（两处非冲突改动，合并直接）。
- 与 v1（softplus）不冲突，可独立叠加（只需合并 `hyperbolic_projector.py` 的那一行）。
