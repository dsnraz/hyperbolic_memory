# 实验指南

本指南提供一组可直接运行的对照实验，用于逐一验证每个版本的效果。

## 基准查询

从 `结果(3).md` 里摘两个查询（正确答案已知）：

| 编号 | 查询 | 真父（DOMAIN） | 真证据（DIALOGUE） |
|---|---|---|---|
| Q1 | `"When did Caroline go to the LGBTQ support group?"` | domain_2 (Social Issues) | dialogue_4 |
| Q2 | `"What fields would Caroline be likely to pursue in her education?"` | domain_8 (Education and Career) | dialogue_8, dialogue_10 |

## 实验 0：基线（不应用任何版本）

```bash
# 先 revert 所有改动，跑原始代码
python algorithms/revert.py

# 然后诊断 Q1
python algorithms/diagnose.py "When did Caroline go to the LGBTQ support group?" \
    --retriever_type hyperbolic_angular \
    --checkpoint /path/to/your/hyperbolic_projector_final.pt \
    --persist_dir /path/to/your/hierarchical_memory_locomo

# 同样方式跑 Q2
```

记录下每层的 `score range` 和 `dynamic range` 作为基线。

**预期基线（根据 `结果(3).md`）**：
- Q1 DOMAIN: range 0.008，**无真父**
- Q1 CATEGORY: range ≈0.008（saturation）
- Q2 任何层级（纯测地时）正常

---

## 实验 1：v7 — DOMAIN 走测地线

```bash
python algorithms/apply.py v7_domain_geodesic

python algorithms/diagnose.py "When did Caroline go to the LGBTQ support group?" \
    --retriever_type hyperbolic_angular_geodesic_hybrid \
    --checkpoint /path/to/your/hyperbolic_projector_final.pt \
    --persist_dir /path/to/your/hierarchical_memory_locomo
```

**期望**：Q1 DOMAIN 层 top-10 出现 `domain_2 (Social Issues)`；动态范围 > 0.03。

下一步实验前 revert：
```bash
python algorithms/revert.py
```

---

## 实验 2：v5 — 推理 sim 公式

```bash
python algorithms/apply.py v5_inference_sim

python algorithms/diagnose.py "When did Caroline go to the LGBTQ support group?" \
    --retriever_type hyperbolic_angular \
    --checkpoint ... --persist_dir ...
```

**期望**：CATEGORY 层 top-10 分数分布不再挤在 0.98-1.00，动态范围扩大，正确候选上前。

```bash
python algorithms/revert.py
```

---

## 实验 3：v6 — 父权重门控

```bash
python algorithms/apply.py v6_parent_gate
python algorithms/diagnose.py "When did Caroline go to the LGBTQ support group?" ...
python algorithms/revert.py
```

**期望**：对有多父的候选（如 keyword_13 support）的打分，更依赖真正相关的 1-2 个父，不再被 30 个父平均稀释。

---

## 实验 4：v5 + v6 + v7 组合（推理三件套）

因三者都改 `hyperbolic_retriver.py`，直接串行 apply 会丢失前者的改动。需要**手动合并**或使用合并版（如果创建了）。

手动合并步骤（建议）：

1. `python algorithms/apply.py v5_inference_sim`
2. 读取 `algorithms/v6_parent_gate/hyperbolic_retriver.py`，把它对 `_parent_aggregation_weights` 的改动**手动合并**到已经被 v5 改过的 `model/retrievers/hyperbolic_retriver.py`。
3. 同理合并 v7 的 `_similarity` + boundary 改动。

或：

1. 以 v5 文件为起点，手动粘贴 v6 的 `_parent_aggregation_weights` 和 v7 的 `HybridHyperbolicRetriever._similarity` 到一份新文件，存到 `algorithms/v5_v6_v7_combined/`（用户自行创建）。

---

## 实验 5：v1 — softplus 修复（需重训）

```bash
python algorithms/apply.py v1_softplus_fix

# 重训
bash scripts/train.sh
# 或直接：
python -m model.hyperbolic_utils.train \
    --vector_store_path ... --output_dir ./checkpoints_v1 \
    --num_iterations 12500

# 用新 checkpoint 跑诊断
python algorithms/diagnose.py "..." --checkpoint ./checkpoints_v1/hyperbolic_projector_final.pt ...

python algorithms/revert.py
```

**期望**：训练第一步打印 `curvature: 0.1000`（修复前是 0.7444）。训练后 DOMAIN 壳深度从 ≈0.116 变为 ≈0.316。

---

## 实验 6：v3 — 混合训练（需重训）

```bash
python algorithms/apply.py v3_mixed_training

python -m model.hyperbolic_utils.train \
    --mixed_training \
    --vector_store_path ... --output_dir ./checkpoints_v3 \
    --mixed_total_iterations 12500

python algorithms/diagnose.py "..." --checkpoint ./checkpoints_v3/hyperbolic_projector_final.pt ...

python algorithms/revert.py
```

**期望**：训练日志里每 100 步打印一次三个层级对的采样计数，比例接近 {0.04, 0.32, 0.64}。DOMAIN 层动态范围恢复。

---

## 实验 7：v2 — mask 修复（需重训 phase 2）

```bash
python algorithms/apply.py v2_mask_fix

# 重训（尤其 CATEGORY→KEYWORD 阶段最受益）
python -m model.hyperbolic_utils.train \
    --vector_store_path ... --output_dir ./checkpoints_v2 \
    --level_pair_index 2

# 验证：打印一个 batch 的 mask 密度
python - <<EOF
from model.hyperbolic_utils.hierarchical_dataset import create_subtree_dataloader, extract_nodes_from_store
from model.stores.hierarchical_vector_store import HierarchicalVectorStore
vs = HierarchicalVectorStore(persist_directory="...", embedding_function=None)
nodes = extract_nodes_from_store(vs, level_pair_index=2)
dl = create_subtree_dataloader(
    nodes_by_level=nodes, embedding_dim=384,
    level_pair=("CATEGORY","KEYWORD"), num_iterations=10,
)
for b in dl:
    m = b.parent_child_mask
    print(f"shape={m.shape} sum={m.sum().item()} density={m.sum().item()/(m.shape[0]*m.shape[1]):.4f}")
EOF

python algorithms/revert.py
```

**期望**：修复后 density 比修复前更高（至少在多父子密集的 KEYWORD 层）。

---

## 实验 8：v9 — 全链条训练（需重训）

```bash
python algorithms/apply.py v9_full_chain

python -m model.hyperbolic_utils.train \
    --chain_training --chain_skip_weight 0.4 \
    --vector_store_path ... --output_dir ./checkpoints_v9 \
    --mixed_total_iterations 15000

python algorithms/diagnose.py "..." --checkpoint ./checkpoints_v9/hyperbolic_projector_final.pt ...

python algorithms/revert.py
```

**期望**：训练日志里出现 5 个层级对的采样计数（3 相邻 + 2 跨层）。测试时祖-孙外角分布更集中。

---

## 实验 9：v8 — projector 软化（需重训）

```bash
python algorithms/apply.py v8_depth_range

python -m model.hyperbolic_utils.train \
    --vector_store_path ... --output_dir ./checkpoints_v8

# 验证层内径向分布
python - <<EOF
import torch
from model.hyperbolic_utils.hyperbolic_projector import Hyperbolic_projector
from model.stores.hierarchical_vector_store import HierarchicalVectorStore
from model.hierarchical.hierarchy_types import HierarchyLevel
import numpy as np
from collections import defaultdict

ckpt = torch.load("./checkpoints_v8/hyperbolic_projector_final.pt", map_location="cpu")
p = Hyperbolic_projector(input_dim=384, hidden_dim=256)
p.load_state_dict(ckpt['model_state_dict']); p.eval()

vs = HierarchicalVectorStore(persist_directory="...", embedding_function=None)
stats = defaultdict(list)
for lvl in [HierarchyLevel.DOMAIN, HierarchyLevel.CATEGORY, HierarchyLevel.KEYWORD, HierarchyLevel.DIALOGUE]:
    for n in vs.get_nodes_by_level(lvl)[:100]:
        with torch.no_grad():
            x = torch.tensor(n.level_embedding).unsqueeze(0).float()
            _, zH = p(x)
        stats[lvl.name].append(torch.norm(zH).item())
for lvl, norms in stats.items():
    print(f"{lvl}: mean={np.mean(norms):.4f} std={np.std(norms):.4f}")
EOF

python algorithms/revert.py
```

**期望**：同层 std ≥ 0.02（而非修改前的 0.0001）；层间 mean 仍按 DOMAIN < CATEGORY < KEYWORD < DIALOGUE 递增。

---

## 实验 10：v4 — query 前缀（不需重训，快速探索）

```bash
python algorithms/apply.py v4_query_prefix

# 扫不同前缀
for prefix in "" "QUERY: " "DOMAIN: " "CATEGORY: "; do
    echo "===== prefix=$prefix ====="
    python algorithms/diagnose.py "When did Caroline go to the LGBTQ support group?" \
        --retriever_type hyperbolic_angular \
        --checkpoint ... --persist_dir ... \
        --query_prefix "$prefix"
done

python algorithms/revert.py
```

**期望**：某个前缀下 DOMAIN 动态范围最大、真父排名最高——这就是推荐前缀。若所有前缀都差不多，说明前缀不是关键变量（和你的直觉一致）。

---

## 总结实验矩阵

| 版本 | 需要重训 | 关键指标 | 基线比较对象 |
|---|---|---|---|
| v1 | 是（完整） | 初始 curvature = 0.1 | softplus(c) 应用前后输出曲率 |
| v2 | 是（phase 2） | mask 密度 | 同一 batch 的 mask.sum |
| v3 | 是（完整） | 层级对采样比例 | 比原顺序训练平均分布 |
| v4 | 否 | 前缀 recall | 不同 prefix 的 DOMAIN top-10 |
| v5 | 否 | CATEGORY 动态范围 | 0.008 → >0.1 |
| v6 | 否 | 多父候选得分 | 真父占比提升 |
| v7 | 否 | DOMAIN 动态范围 | 0.008 → >0.03 |
| v8 | 是 | 同层 ||z|| 标准差 | 0 → 0.02+ |
| v9 | 是 | 祖孙外角分布 | 尾部外角显著压窄 |

## 对比基线：cosine 检索

每次实验都可以同时跑一次 `--retriever_type cosine`，作为"无 projector、纯欧氏 cosine"的绝对基线。若任何一版都打不过 cosine，说明这条路线需要重新设计。
