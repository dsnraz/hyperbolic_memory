"""
诊断投影器是否保留了语义结构。

对指定 query，对比三种排序方式的斯皮尔曼相关系数：
  A. 原始 embedding 余弦相似度（欧式 baseline）
  B. 投影后 O 点双曲内角（angle_mode="origin"）
  C. 投影后测地线距离

如果 B/A 相关性低（<0.5），说明 projector 破坏了原始语义结构。

用法:
    python algorithms/diagnose_projector.py \
        --persist_dir /path/to/vector_store \
        --checkpoint /path/to/hyperbolic_projector_final.pt \
        --query "What did Caroline research?" \
        --top_k 50
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from model.hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel
from model.hyperbolic_utils import lorentz as L
from model.hyperbolic_utils.hyperbolic_projector import Hyperbolic_projector
from model.retrievers.hyperbolic_retriver import _pair_hyperbolic_angle_at_origin_scores
from model.stores.hierarchical_vector_store import HierarchicalVectorStore


def load_projector(checkpoint_path: str, device: str = "cpu") -> Hyperbolic_projector:
    path = Path(checkpoint_path)
    if path.is_dir():
        pt_files = sorted(path.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pt_files:
            raise FileNotFoundError(f"no .pt found in {checkpoint_path}")
        path = pt_files[0]

    ckpt = torch.load(str(path), map_location="cpu")
    cfg = ckpt.get("config", {})
    projector = Hyperbolic_projector(
        input_dim=cfg.get("embedding_dim", 768),
        hidden_dim=cfg.get("hidden_dim", 2048),
        curvature=cfg.get("initial_curvature", ckpt.get("curvature", 0.1)),
        alpha=cfg.get("alpha", 0.1),
        beta=cfg.get("beta", 0.8),
    )
    projector.load_state_dict(ckpt["model_state_dict"])
    projector.to(device)
    projector.eval()
    return projector


def cosine_score(q_emb: Sequence[float], n_emb: Sequence[float]) -> float:
    a = np.asarray(q_emb, dtype=np.float32)
    b = np.asarray(n_emb, dtype=np.float32)
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def compute_rankings(
    store: HierarchicalVectorStore,
    projector: Hyperbolic_projector,
    query_embedding: List[float],
    target_level: HierarchyLevel,
    top_k: int,
    device: str,
) -> dict:
    """对指定层级的所有节点，用三种方式排序，返回排名列表。"""
    nodes = store.get_nodes_by_level(target_level)
    valid = [(n, n.level_embedding) for n in nodes if n.level_embedding is not None]
    if len(valid) < 2:
        raise RuntimeError(f"not enough valid nodes at {target_level.name}")

    # 投影所有节点
    with torch.no_grad():
        q_tensor = torch.tensor([query_embedding], dtype=torch.float32, device=device)
        _, query_h = projector(q_tensor)
        query_h_cpu = query_h.cpu()

        node_embs = torch.tensor([e for _, e in valid], dtype=torch.float32, device=device)
        node_euclidean, node_h_all = projector(node_embs)
        node_h_all_cpu = node_h_all.cpu()

    curv = float(torch.nn.functional.softplus(projector.c).item())

    items = []
    for idx, (node, _) in enumerate(valid):
        # A. 原始余弦
        cos = cosine_score(query_embedding, node.embedding if node.embedding is not None else node.level_embedding)

        # B. O 点内角
        node_h_i = node_h_all_cpu[idx]
        _, angle_opp = _pair_hyperbolic_angle_at_origin_scores(query_h_cpu, node_h_i, curv)
        angle_score = 1.0 / (1.0 + angle_opp) if math.isfinite(angle_opp) else 0.0

        # C. 测地线
        dist = L.pairwise_dist_vectors(query_h_cpu, node_h_i.unsqueeze(0), curv=curv).squeeze()
        dist_v = float(dist.item())
        geo_score = 1.0 / (1.0 + dist_v) if math.isfinite(dist_v) else 0.0

        items.append({
            "node": node,
            "cosine": cos,
            "angle_score": angle_score,
            "angle_opposite": angle_opp if math.isfinite(angle_opp) else float("inf"),
            "geodesic_score": geo_score,
            "geodesic_dist": dist_v,
        })

    # 排名：分数降序（opposite_score 升序）
    rank_cos = sorted(items, key=lambda x: -x["cosine"])
    rank_ang = sorted(items, key=lambda x: x["angle_opposite"])  # 内角越小越好
    rank_geo = sorted(items, key=lambda x: x["geodesic_dist"])   # 距离越小越好

    node_to_rank_cos = {it["node"].id: i for i, it in enumerate(rank_cos)}
    node_to_rank_ang = {it["node"].id: i for i, it in enumerate(rank_ang)}
    node_to_rank_geo = {it["node"].id: i for i, it in enumerate(rank_geo)}

    common_ids = set(node_to_rank_cos) & set(node_to_rank_ang) & set(node_to_rank_geo)
    ranks_cos = [node_to_rank_cos[nid] for nid in common_ids]
    ranks_ang = [node_to_rank_ang[nid] for nid in common_ids]
    ranks_geo = [node_to_rank_geo[nid] for nid in common_ids]

    return {
        "top_cosine": [(it["node"].content[:60], it["cosine"]) for it in rank_cos[:top_k]],
        "top_angle": [(it["node"].content[:60], it["angle_score"]) for it in rank_ang[:top_k]],
        "top_geodesic": [(it["node"].content[:60], it["geodesic_score"]) for it in rank_geo[:top_k]],
        "spearman_cos_ang": spearmanr(ranks_cos, ranks_ang),
        "spearman_cos_geo": spearmanr(ranks_cos, ranks_geo),
        "spearman_ang_geo": spearmanr(ranks_ang, ranks_geo),
        "n_nodes": len(common_ids),
        "curvature": curv,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", 
                        default = "/share/home/leiyh5/Memory/data/memory_running_category/round_1_conv-26",
                        )
    parser.add_argument("--checkpoint", 
                        default = "/share/home/leiyh5/Memory/checkpoints_locomo_category_c0p1/hyperbolic_projector_final.pt",
                        )
    parser.add_argument("--query", default="What did Caroline research?")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--level", default="ALL", help="层级名或 ALL")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(args.embedding_model)
    query_embedding = emb_model.encode(args.query).tolist()

    store = HierarchicalVectorStore(
        persist_directory=args.persist_dir,
        embedding_function=None,
        delayed_write=False,
    )
    projector = load_projector(args.checkpoint, args.device)

    levels = list(HierarchyLevel) if args.level == "ALL" else [HierarchyLevel[args.level]]

    print(f"\nQuery: {args.query}")
    print(f"Curvature (softplus(c)): {float(torch.nn.functional.softplus(projector.c).item()):.6f}")
    print(f"Alpha: {projector.alpha}, Beta: {projector.beta}")
    print(f"Projector input dim: {projector.phi[0].in_features}")
    print(f"Embedding dim: {len(query_embedding)}")
    print("=" * 70)

    for level in levels:
        try:
            results = compute_rankings(store, projector, query_embedding, level, args.top_k, args.device)
        except RuntimeError as e:
            print(f"\n[SKIP] {level.name}: {e}")
            continue

        sp_cos_ang = results["spearman_cos_ang"]
        sp_cos_geo = results["spearman_cos_geo"]
        sp_ang_geo = results["spearman_ang_geo"]

        print(f"\n{'=' * 70}")
        print(f"Level: {level.name}  ({results['n_nodes']} nodes, curv={results['curvature']:.4f})")
        print(f"  Spearman: cos↔angle={sp_cos_ang.statistic:.4f} (p={sp_cos_ang.pvalue:.2e})")
        print(f"  Spearman: cos↔geo  ={sp_cos_geo.statistic:.4f} (p={sp_cos_geo.pvalue:.2e})")
        print(f"  Spearman: angle↔geo={sp_ang_geo.statistic:.4f} (p={sp_ang_geo.pvalue:.2e})")

        verdict = (
            "✓ 投影保留了语义结构" if sp_cos_ang.statistic > 0.7
            else "⚠ 投影部分改变了排序" if sp_cos_ang.statistic > 0.4
            else "✗ 投影严重破坏了原始语义排序"
        )
        print(f"  → {verdict}")

        # 打印 top-k
        print(f"\n  --- TOP {args.top_k} COSINE ---")
        for i, (content, score) in enumerate(results["top_cosine"]):
            print(f"  {i+1:2d}. [{score:.4f}] {content}")

        print(f"\n  --- TOP {args.top_k} ANGLE (O-point) ---")
        for i, (content, score) in enumerate(results["top_angle"]):
            print(f"  {i+1:2d}. [{score:.4f}] {content}")

        print(f"\n  --- TOP {args.top_k} GEODESIC ---")
        for i, (content, score) in enumerate(results["top_geodesic"]):
            print(f"  {i+1:2d}. [{score:.4f}] {content}")


if __name__ == "__main__":
    main()
