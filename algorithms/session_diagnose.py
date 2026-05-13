"""
诊断脚本：在任何检索器上跑一个查询，统计各层 top-10 的关键指标。

用法：
    python algorithms/session_diagnose.py --query ... [--retriever1 ...] [--retriever2 ...]

    --retriever1：主检索（分层 retrieve 的打分方式）
    --retriever2：正确答案溯源回溯时的对比列（可与 retriever1 不同）；省略则与 retriever1 相同

    别名：--retriever_type 等同于 --retriever1；--retriever_type2 等同于 --retriever2

输出的关键指标：
    - 每层候选数
    - 每层 top-10 得分范围（动态范围）
    - top-10 里包含的父节点 id（便于判断树是否在正确分支）

目的：
    对同一个查询，用不同版本的代码（应用不同的修复）跑一次，对比输出。
    比较的关键：DOMAIN 层的动态范围从 0.008（未修复）是否扩大到 >0.03。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import model.retrievers.try_retriver2 as tr2
from algorithms.locomo_qa_evidence import reference_dialogues_for_query

from model.hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel
from model.stores.hierarchical_vector_store import HierarchicalVectorStore

RETRIEVER_CHOICES = [
    "cosine",
    "hyperbolic_geodesic",
    "hyperbolic_angular",
    "hyperbolic_angular_geodesic_hybrid",
]


def score_node_with_retriever_kind(
    kind: str,
    retriever,
    query_embedding: Sequence[float],
    projected_query: Optional[object],
    node: HierarchicalNode,
) -> float:
    """与对应检索器类内打分一致：cosine 用欧式余弦，其余用 project_query 后的 _similarity。"""
    if kind == "cosine":
        return tr2.retriever_euclidean._cosine_similarity(query_embedding, node.embedding)
    assert projected_query is not None
    s, _ = retriever._similarity(projected_query, node)
    return float(s)


def keyword_session_ids(
    keyword_node: HierarchicalNode,
    store: HierarchicalVectorStore,
) -> list[str]:
    """单个 keyword 子 dialogue 上去重、排序后的 session_id 列表。"""
    if keyword_node.level != HierarchyLevel.KEYWORD:
        raise ValueError(f"expected KEYWORD node, got {keyword_node.level!r}")

    seen: set[str] = set()
    meta = keyword_node.metadata or {}
    sid = meta.get("session_id")
    if sid is not None and str(sid).strip():
        seen.add(str(sid).strip())
    for child_id in keyword_node.child_ids:
        dialogue = store.get_node(child_id, HierarchyLevel.DIALOGUE)
        if dialogue is None:
            raise ValueError(f"dialogue child not found: {child_id!r} under keyword {keyword_node.id!r}")
        meta = dialogue.metadata or {}
        sid = meta.get("session_id")
        if sid is None or str(sid).strip() == "":
            raise ValueError(f"missing or empty session_id on dialogue {child_id!r}")
        seen.add(str(sid).strip())
    return sorted(seen)


def memory_unit_extra_text(node: HierarchicalNode) -> str:
    meta = node.metadata or {}
    unit_type = meta.get("unit_type") or meta.get("memory_unit_mode") or "keyword"
    if unit_type == "fact":
        return (
            f"unit_type=fact session_id={meta.get('session_id', '')} "
            f"dialogue_indices={meta.get('dialogue_indices', [])} "
            f"subject={meta.get('subject', '')!r} time={meta.get('time', '')!r}"
        )
    return f"unit_type={unit_type}"


def count_session_ids_across_keyword_hits(
    keyword_hits: list,
    store: HierarchicalVectorStore,
) -> list[tuple[str, int]]:
    """
    本轮 KEYWORD 层每个 hit：取其 session_id 集合；
    某 session_id 在多少个不同 keyword hit 中出现，计数即加几。
    """
    counts: Counter[str] = Counter()
    for h in keyword_hits:
        for sid in keyword_session_ids(h.node, store):
            counts[sid] += 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))


def print_ancestor_chain(
    store: HierarchicalVectorStore,
    node: HierarchicalNode,
    score_fn: Callable[[HierarchicalNode], tuple[float, float]],
    primary_label: str,
    compare_label: str,
    label: str,
) -> None:
    """与 diagnose.py 一致：gold dialogue → keyword 得分 → 原句得分 → category/domain 回溯。"""
    print(node.content)

    keywords = node.parent_ids
    print(keywords)

    for keyword in keywords:
        node_embedding_keyword, node_level_embedding_keyword, node_keyword = tr2.load_node_embedding(
            node_id=keyword, level=HierarchyLevel.KEYWORD
        )
        print(node_keyword.content)
        score1, score2 = score_fn(node_keyword)
        print("关于", keyword, node_keyword.content, "的得分")
        print(f"{primary_label}：", score1)
        print(f"{compare_label}：", score2)

    score1, score2 = score_fn(node)
    print(f"关于原句的得分{node.id,node.content}")
    print(f"{primary_label}：", score1)
    print(f"{compare_label}：", score2)

    category_ids: List[str] = []
    seen_category_ids: set[str] = set()
    for keyword in keywords:
        _, _, node_keyword = tr2.load_node_embedding(
            node_id=keyword, level=HierarchyLevel.KEYWORD
        )
        for category_id in node_keyword.parent_ids:
            if category_id in seen_category_ids:
                continue
            seen_category_ids.add(category_id)
            category_ids.append(category_id)

    print("--------------------------------")
    print("由关键词回溯得到的 category 节点:", category_ids)

    seen_domain_ids: set[str] = set()
    for category_id in category_ids:
        _, _, node_category = tr2.load_node_embedding(
            node_id=category_id, level=HierarchyLevel.CATEGORY
        )
        score_c_e, score_c_h = score_fn(node_category)
        print(f"关于{category_id}({node_category.content})的得分")
        print(f"{primary_label}：", score_c_e)
        print(f"{compare_label}：", score_c_h)
        print(f"{category_id} 的 domain 父节点: {node_category.parent_ids}")

        for domain_id in node_category.parent_ids:
            if domain_id in seen_domain_ids:
                continue
            seen_domain_ids.add(domain_id)
            _, _, node_domain = tr2.load_node_embedding(
                node_id=domain_id, level=HierarchyLevel.DOMAIN
            )
            score_d_e, score_d_h = score_fn(node_domain)
            print(f"关于{category_id}的父节点{node_domain.content}({domain_id})的得分")
            print(f"{primary_label}：", score_d_e)
            print(f"{compare_label}：", score_d_h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="What is Caroline's identity?",
        help="查询文本（与 locomo_qa_test.json 中某条 question 一致时可自动拼 gold evidence）",
    )
    parser.add_argument(
        "--qa_json",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data/locomo/locomo1_10.json"),
        help="LoCoMo QA+conversation，用于按 question 匹配 evidence 并生成与 store 一致的参考文本",
    )
    parser.add_argument(
        "--retriever1",
        "--retriever_type",
        dest="retriever1",
        type=str,
        default="cosine",
        choices=RETRIEVER_CHOICES,
        help="主检索器类型（分层 retrieve）；别名 --retriever_type",
    )
    parser.add_argument(
        "--retriever2",
        "--retriever_type2",
        dest="retriever2",
        type=str,
        default="cosine",
        choices=RETRIEVER_CHOICES,
        help=(
            "溯源回溯对比列②的检索器类型（四种均可）；省略则与 --retriever1 相同；别名 --retriever_type2"
        ),
    )
    parser.add_argument("--checkpoint", type=str, required=False, default="/share/home/leiyh5/Memory/checkpoints_locomo_category_c0p1/hyperbolic_projector_final.pt")
    parser.add_argument("--persist_dir", type=str, default="/share/home/leiyh5/Memory/data/memory_running_category_384_2stage/round_1_conv-26",
                        help="vector store 持久化目录")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="用于生成 query embedding 的模型名（需与 projector 输入维度匹配）。",
    )
    parser.add_argument("--top_k", type=int, nargs=4, default=[20, 20, 15, 8])
    parser.add_argument("--memory_unit_mode", choices=["keyword", "fact"], default="fact")
    parser.add_argument("--query_prefix", type=str, default=None,
                        help="v4_query_prefix: 可选的 query 前缀")
    args = parser.parse_args()

    # 延迟导入以便在 v* 被 apply 后生效
    from model.llm_inference.session_llm_inference import SessionMemoryAugmentedLLMInference
    from model.hierarchical.hierarchy_types import HierarchyLevel

    retriever2_kind = args.retriever2 or args.retriever1

    inf = SessionMemoryAugmentedLLMInference(
        persist_directory=args.persist_dir,
        retriever_type=args.retriever1,
        projector_checkpoint_path=args.checkpoint,
        retriever_top_k=args.top_k,
        embedding_model=args.embedding_model,
        memory_unit_mode=args.memory_unit_mode,
    )

    # v4 支持：构造后设置前缀
    if args.query_prefix is not None and hasattr(inf.retriever, "query_prefix"):
        inf.retriever.query_prefix = args.query_prefix

    def build_retriever_by_type(kind: str):
        if kind == "cosine":
            return tr2.CosineRetriever(vector_store=inf.manager.vector_store)
        if kind == "hyperbolic_geodesic":
            return tr2.GeodesicHyperbolicRetriever(
                vector_store=inf.manager.vector_store,
                checkpoint_path=args.checkpoint,
            )
        if kind == "hyperbolic_angular":
            return tr2.MultiParentAngularHyperbolicRetriever(
                vector_store=inf.manager.vector_store,
                checkpoint_path=args.checkpoint,
            )
        if kind == "hyperbolic_angular_geodesic_hybrid":
            return tr2.HybridHyperbolicRetriever(
                vector_store=inf.manager.vector_store,
                checkpoint_path=args.checkpoint,
            )
        raise ValueError(f"unsupported retriever type: {kind}")

    result = inf.retriever.retrieve(
        query_text=args.query,
        top_k=args.top_k,
        start_level=HierarchyLevel.DOMAIN,
        target_level=HierarchyLevel.DIALOGUE,
        # hybrid_scoring_boundary = HierarchyLevel.KEYWORD
    )

    store = inf.manager.vector_store

    print()
    print("=" * 80)
    print(f"Query: {args.query}")
    print(f"主检索 retriever1: {args.retriever1}")
    print(f"溯源对比 retriever2: {retriever2_kind}")
    if args.query_prefix is not None:
        print(f"Query prefix: {args.query_prefix!r}")
    print("=" * 80)

    for lvl_res in result.level_results:
        print(f"\n--- Level: {lvl_res.level.name} | candidates: {lvl_res.candidate_count} ---")
        scores = [h.score for h in lvl_res.hits]
        if scores:
            mn, mx = min(scores), max(scores)
            print(f"  top-{len(scores)} score range: [{mn:.4f}, {mx:.4f}]  dynamic range: {mx - mn:.4f}")
        for h in lvl_res.hits:
            content = h.node.content.replace("\n", " ")
            content = content[:80] + "..." if len(content) > 80 else content
            parents = ",".join(h.node.parent_ids) if h.node.parent_ids else "(root)"
            print(f"  {h.node.id:<16}  score={h.score:.4f}  parents=[{parents[:40]}]  {content}")
            if lvl_res.level == HierarchyLevel.KEYWORD:
                print(f"      {memory_unit_extra_text(h.node)}")
                for sid in keyword_session_ids(h.node, store):
                    print(f"      session_id={sid}")

        if lvl_res.level == HierarchyLevel.KEYWORD and lvl_res.hits:
            agg = count_session_ids_across_keyword_hits(lvl_res.hits, store)
            print(f"  [session_id, memory_unit_hit_count]: {agg}")

    print("=" * 80)
    # try_retriver2 的工具函数依赖模块级全局变量；被 import 使用时需要手动绑定。
    tr2.store = store
    tr2.retriever_euclidean = tr2.CosineRetriever(vector_store=store)

    query_embedding = tr2.retriever_euclidean._prepare_query_embedding(args.query, None)
    retriever_r1 = inf.retriever
    retriever_r2 = build_retriever_by_type(retriever2_kind)
    if args.query_prefix is not None and hasattr(retriever_r2, "query_prefix"):
        retriever_r2.query_prefix = args.query_prefix

    projected_r1: Optional[object] = None
    if args.retriever1 != "cosine":
        projected_r1 = retriever_r1.project_query(query_embedding)
    projected_r2: Optional[object] = None
    if retriever2_kind != "cosine":
        projected_r2 = retriever_r2.project_query(query_embedding)

    primary_label = f"retriever1[{args.retriever1}]"
    compare_label = f"retriever2[{retriever2_kind}]"

    def score_fn(node: HierarchicalNode) -> tuple[float, float]:
        score_primary = score_node_with_retriever_kind(
            args.retriever1, retriever_r1, query_embedding, projected_r1, node
        )
        score_secondary = score_node_with_retriever_kind(
            retriever2_kind, retriever_r2, query_embedding, projected_r2, node
        )
        return score_primary, score_secondary

    print(
        f"[回溯分数] 列① = retriever1({args.retriever1})；列② = retriever2({retriever2_kind})"
    )

    # ---- gold evidence: 逐条独立格式化，支持多 evidence 和含图像 turn ----
    try:
        evidence_texts, gold_evidence = reference_dialogues_for_query(
            args.query, Path(args.qa_json)
        )
    except (FileNotFoundError, LookupError, KeyError, ValueError) as e:
        print(f"[locomo_qa_evidence] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[gold evidence] {gold_evidence}")

    for ev_idx, ev_text in enumerate(evidence_texts):
        ev_id = gold_evidence[ev_idx]
        print("--------------------------------")
        print(f"[evidence {ev_idx}] dia_id={ev_id}")

        try:
            _e, _le, node = tr2.load_node_embedding(text=ev_text)
            node_embedding, node_level_embedding, node = tr2.load_node_embedding(node_id=node.id)
        except ValueError as exc:
            print(f"  !! 未在 store 中找到该 evidence 节点: {exc}")
            # 尝试用 content 模糊搜索
            found = False
            for d_node in store.get_nodes_by_level(HierarchyLevel.DIALOGUE):
                if ev_text.strip()[:80] in (d_node.content or ""):
                    print(f"  -> 模糊匹配到: {d_node.id}  {d_node.content[:100]}...")
                    node = d_node
                    found = True
                    break
            if not found:
                continue

        print_ancestor_chain(
            store,
            node,
            score_fn,
            primary_label,
            compare_label,
            label=f"evidence[{ev_idx}] {ev_id}",
        )


if __name__ == "__main__":
    main()
