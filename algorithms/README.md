# 双曲分层检索算法修复版本集

本目录把 v2 诊断里给出的建议拆成 **9 个完全解耦的版本**，每个版本解决一个独立问题，都可以单独应用/单独跑实验。

> **使用方式**：每个版本是一个子目录，里面包含修改后的源文件和说明。用根目录下的 `apply.py` 把某版本的修改应用到主 codebase（会自动 backup 原文件），用 `revert.py` 还原。

---

## 版本索引

| 版本 | 问题 | 修改面 | 需要重训？ | 解决的 v2 诊断项 |
|---|---|---|---|---|
| v1_softplus_fix | 初始曲率 `softplus(0.1)≈0.744` 而非 0.1 | projector 初始化 | 是 | 训练问题 1 |
| v2_mask_fix | parent_child_mask 假负例 | 数据采样 | 是 | 训练问题 2 |
| v3_mixed_training | 顺序训练导致灾难性遗忘 | 训练脚本 | 是 | 训练问题 3 |
| v4_query_prefix | query 与节点输入分布不一致 | 推理 query 构造 | 否 | 训练问题 4 |
| v5_inference_sim | 推理 sim 公式与训练目标不一致 | retriever 打分 | 否 | v1 错误 1 |
| v6_parent_gate | 多父加权忽略 query 对齐 | retriever 权重 | 否 | v1 错误 2 |
| v7_domain_geodesic | DOMAIN 层检索断链 | retriever boundary | 否 | v1 错误 3 的推理修复 |
| v8_depth_range | 硬投影抹平径向自由度 | projector + 损失 | 是 | v1 错误 6 的软化版本（接受用户反驳后修订） |
| v9_full_chain | 训练只做父子单阶，无全链约束 | 损失 + 训练 | 是 | 用户提出的全链条方向 |

---

## 修改依据（接受的用户反驳）

编写这批修改时，接受了你提出的两条关键反驳并据此调整方案：

**反驳 1：原点处切平面是保角的，O 内角不会因为壳太浅而畸变。**
- **确实如此**。`exp_map0` 在 O 点处是保角的，两个切空间向量的欧氏夹角等于它们双曲射线的 O 内角。
- 所以 DOMAIN 层 top-10 压成 0.008 的**真正原因不是几何退化**，而是 **projector 没有把 137 个 DOMAIN 节点训得方向足够分散**。
- 进一步指向灾难性遗忘（v3）和 β 项冗余（v1 错误 7′）作为训练不充分的来源。
- v7（DOMAIN 改测地线）的定位由"修复几何退化"修订为"**绕开训练不充分的层，提供可用的基线**"。

**反驳 2：对每节点做点目标层级约束会抹掉径向自由度，比质心更硬。**
- **确实如此**。如果强制每个 DOMAIN 节点 `d_v=0`，同层所有节点径向深度都一样，层内判别力反而更差。
- v8 改为"**层间顺序 margin**"：只要 DOMAIN 的深度 < CATEGORY < KEYWORD < DIALOGUE 就行，具体到层内允许自由散布。配合软化 projector（去掉 target_norm 硬投影），让模型自己决定层内径向分布。

---

## 应用流程

```bash
# 1. 应用某版本（示例：v1）
python algorithms/apply.py v1_softplus_fix

# 2. 重新训练（只对需要重训的版本，如 v1, v2, v3, v8, v9）
bash scripts/train.sh

# 3. 跑推理测试
python -m model.llm_inference.run --retriever_type hyperbolic_angular

# 4. 还原
python algorithms/revert.py
```

> 应用任意一个版本前，请确保当前 codebase 已无未提交改动；`apply.py` 会自动 backup 被替换的文件到 `.v_original` 后缀。

---

## 建议的运行顺序

按"易到难、先推理再训练"：

```
v7 (改一行 retriever boundary，立刻跑推理)
 → v5 + v6 (同样只改推理)
 → v4 (可选，只改推理)
 → v1 (小重训，30 分钟级别)
 → v2 (重训 phase 2，验证 mask 假负例影响)
 → v3 (重训全部，混合模式)
 → v8 (重训全部，软化 projector)
 → v9 (重训 + 新增祖-孙采样)
```

每应用一个版本跑完实验、revert，再应用下一个。若想组合（如 v5+v6），可连续 apply 两个（前一个的 backup 会被后一个保留）；如需要干净组合，请参考各版本 README。

---

## 各版本互相的兼容性

| 组合 | 兼容 | 说明 |
|---|---|---|
| v1 + v2 + v3 | ✓ | 训练侧改动不冲突 |
| v1 + v2 + v3 + v8 | ✓ | v8 改 projector 文件，v1 也改同文件；apply v8 后手动合并 v1 的 softplus 修复 |
| v5 + v6 | ✓（推荐）| 两者都改 retriever 文件的不同函数；提供合并版在 `v5_v6_combined/`（可选） |
| v5 + v7 | ✓ | 不同修改点 |
| v6 + v7 | ✓ | 不同修改点 |
| v5 + v6 + v7 | ✓ | 三者修改不同函数 |
| v9 + (v1/v2/v3/v8) | ✓ | v9 加新 loss，不改旧 loss；训练脚本需合并 |

> 当两个版本改同一文件，apply.py 会提示并让你选择：直接覆盖（可能丢失前一个版本的改动）或跳过。
