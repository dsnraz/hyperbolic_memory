# v2: parent_child_mask 假负例修复

## 问题

`hierarchical_dataset.py:297-303`：

```python
parent_child_mask = torch.zeros(n_parent, n_child, dtype=torch.float32)
for local_parent_idx, global_child_idx in parent_child_relations:
    if global_child_idx in child_global_to_local:
        local_child_idx = child_global_to_local[global_child_idx]
        parent_child_mask[local_parent_idx, local_child_idx] = 1.0
```

`parent_child_relations` 只记录"**被具体采样**的 (父, 子) 对"。但多父子图里同一个子节点可以有多个真父。

**典型情景**：keyword_13 (support) 有 30+ 个 category 父节点。若批内同时采到父 A、B，A 采到了 keyword_13，B 没采它（采了别的），则：
- `mask[A, keyword_13] = 1` ✓
- `mask[B, keyword_13] = 0` ❌ （其实 B 是 keyword_13 的真父）

Soft-CE 会把 `mask[B, keyword_13] = 0` 当成负样本训练，**推大** `α(B → keyword_13)` 的外角——把 B 从 keyword_13 的锥里推开，违背了训练意图。

## 修改

构造 mask 时**检查所有 (批内父, 批内子) 的真实父子关系**，不只标记被采样的那些：

```python
parent_child_mask = torch.zeros(n_parent, n_child, dtype=torch.float32)

for local_parent_idx, global_parent_idx in enumerate(selected_parent_indices):
    real_children_set = set(self.parent_to_children[global_parent_idx])
    for local_child_idx, global_child_idx in enumerate(selected_child_indices):
        if global_child_idx in real_children_set:
            parent_child_mask[local_parent_idx, local_child_idx] = 1.0
```

## 涉及文件

- `model/hyperbolic_utils/hierarchical_dataset.py`

## 如何应用

```bash
python algorithms/apply.py v2_mask_fix
```

## 如何验证

应用前后跑 `model/hyperbolic_utils/try.py`，看 `parent_child_mask.sum()`：

```python
# 在 try.py 的循环里加：
print(f"mask.sum() = {parent_child_mask.sum().item()}")
print(f"mask.density = {parent_child_mask.sum().item() / (parent_child_mask.shape[0] * parent_child_mask.shape[1]):.4f}")
```

- 修复前：mask.sum() = 批内采样的 (父,子) 对数
- 修复后：mask.sum() ≥ 修复前，等号当且仅当批内不存在"多父共子"的情况

**关键指标**：在 CATEGORY→KEYWORD 层级对（keyword 多父最密集的一阶），差异应该最大。

## 需不需要重训？

**需要**，重训 phase 2（CATEGORY→KEYWORD）即可看到主要差异；其他 phase 差异很小。

## 期望影响

- 训练时正确的 (B, keyword_13) 对不再被错误压制
- 多父节点（如 keyword_13, keyword_76 love）的外角对每个真父都同步收紧
- 推理时多父外角聚合（尤其 v6_parent_gate 启用后）得到更一致的信号

## 不影响其他版本

本修改只改 `hierarchical_dataset.py` 的 mask 构建逻辑。与 v1/v3/v8/v9 训练侧修改不冲突，可叠加。
