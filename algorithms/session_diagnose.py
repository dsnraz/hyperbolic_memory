"""
诊断脚本：在任何检索器上跑一个查询，统计各层 top-10 的关键指标。

用法：
    python algorithms/session_diagnose.py <query_text> [--retriever_type ...] [--checkpoint ...]

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
from typing import List

import model.retrievers.try_retriver2 as tr2
from algorithms.locomo_qa_evidence import reference_dialogues_for_query

from model.hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel
from model.stores.hierarchical_vector_store import HierarchicalVectorStore


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
    query_embedding: list,
    query_embedding_hyperbolic,
    label: str,
) -> None:
    """沿 node 的 parent_ids 回溯，打印从 keyword → category → domain 的得分链。"""
    print(f"\n{'='*60}")
    print(f"[{label}] dialogue: {node.id}  {node.content[:100]}...")
    score_d_e, score_d_h = tr2.score_query_against_node(
        node, query_embedding, query_embedding_hyperbolic,
    )
    print(f"  dialogue 得分: euclidean={score_d_e:.4f}  hyperbolic={score_d_h:.4f}")

    # keyword 父节点
    keyword_ids = node.parent_ids
    print(f"  keyword 父节点: {keyword_ids}")

    category_ids: List[str] = []
    seen_cat: set = set()
    for kw_id in keyword_ids:
        _, _, kw_node = tr2.load_node_embedding(node_id=kw_id, level=HierarchyLevel.KEYWORD)
        score_k_e, score_k_h = tr2.score_query_against_node(
            kw_node, query_embedding, query_embedding_hyperbolic,
        )
        print(f"    keyword {kw_id} ({kw_node.content[:60]}...): euclidean={score_k_e:.4f}  hyperbolic={score_k_h:.4f}")
        print(f"      {memory_unit_extra_text(kw_node)}")
        for cat_id in kw_node.parent_ids:
            if cat_id not in seen_cat:
                seen_cat.add(cat_id)
                category_ids.append(cat_id)

    print(f"  category 父节点: {category_ids}")
    seen_domain: set = set()
    for cat_id in category_ids:
        _, _, cat_node = tr2.load_node_embedding(node_id=cat_id, level=HierarchyLevel.CATEGORY)
        score_c_e, score_c_h = tr2.score_query_against_node(
            cat_node, query_embedding, query_embedding_hyperbolic,
        )
        print(f"    category {cat_id} ({cat_node.content[:60]}...): euclidean={score_c_e:.4f}  hyperbolic={score_c_h:.4f}")
        for dom_id in cat_node.parent_ids:
            if dom_id not in seen_domain:
                seen_domain.add(dom_id)
                _, _, dom_node = tr2.load_node_embedding(node_id=dom_id, level=HierarchyLevel.DOMAIN)
                score_d_e, score_d_h = tr2.score_query_against_node(
                    dom_node, query_embedding, query_embedding_hyperbolic,
                )
                print(f"      domain {dom_id} ({dom_node.content[:60]}...): euclidean={score_d_e:.4f}  hyperbolic={score_d_h:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="When did Melanie run a charity race?",
        help="查询文本（与 locomo_qa_test.json 中某条 question 一致时可自动拼 gold evidence）",
    )
    parser.add_argument(
        "--qa_json",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data/locomo/locomo_qa_test.json"),
        help="LoCoMo QA+conversation，用于按 question 匹配 evidence 并生成与 store 一致的参考文本",
    )
    parser.add_argument("--retriever_type", type=str, default="hyperbolic_geodesic",
                        choices=["cosine", "hyperbolic_geodesic", "hyperbolic_angular",
                                 "hyperbolic_angular_geodesic_hybrid"])
    parser.add_argument("--checkpoint", type=str, required=False, default="/share/home/leiyh5/Memory/checkpoints_locomo_fact/hyperbolic_projector_final.pt")
    parser.add_argument("--persist_dir", type=str, default="/share/home/leiyh5/Memory/data/memory_running_fact/round_1_conv-26",
                        help="vector store 持久化目录")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="用于生成 query embedding 的模型名（需与 projector 输入维度匹配）。",
    )
    parser.add_argument("--top_k", type=int, nargs=4, default=[20, 30, 10, 8])
    parser.add_argument("--memory_unit_mode", choices=["keyword", "fact"], default="fact")
    parser.add_argument("--query_prefix", type=str, default=None,
                        help="v4_query_prefix: 可选的 query 前缀")
    args = parser.parse_args()

    # 延迟导入以便在 v* 被 apply 后生效
    from model.llm_inference.session_llm_inference import SessionMemoryAugmentedLLMInference
    from model.hierarchical.hierarchy_types import HierarchyLevel

    inf = SessionMemoryAugmentedLLMInference(
        persist_directory=args.persist_dir,
        retriever_type=args.retriever_type,
        projector_checkpoint_path=args.checkpoint,
        retriever_top_k=args.top_k,
        embedding_model=args.embedding_model,
        memory_unit_mode=args.memory_unit_mode,
    )

    # v4 支持：构造后设置前缀
    if args.query_prefix is not None and hasattr(inf.retriever, "query_prefix"):
        inf.retriever.query_prefix = args.query_prefix

    result = inf.retriever.retrieve(
        query_text=args.query,
        top_k=args.top_k,
        start_level=HierarchyLevel.DOMAIN,
        target_level=HierarchyLevel.DIALOGUE,
    )

    store = inf.manager.vector_store

    print()
    print("=" * 80)
    print(f"Query: {args.query}")
    print(f"Retriever: {args.retriever_type}")
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
    if getattr(inf, "retriever", None) is not None:
        tr2.retriever_hyperbolic = inf.retriever
    tr2.retriever_euclidean = tr2.CosineRetriever(vector_store=store)

    query_embedding = tr2.retriever_hyperbolic._prepare_query_embedding(args.query, None)
    query_embedding_hyperbolic = tr2.retriever_hyperbolic.project_query(query_embedding)

    # ---- gold evidence: 逐条独立格式化，支持多 evidence 和含图像 turn ----
    try:
        evidence_texts, gold_evidence = reference_dialogues_for_query(
            args.query, Path(args.qa_json)
        )
    except (FileNotFoundError, LookupError, KeyError, ValueError) as e:
        print(f"[locomo_qa_evidence] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[gold evidence] {gold_evidence}  ({len(evidence_texts)} turns)")

    for ev_idx, ev_text in enumerate(evidence_texts):
        ev_id = gold_evidence[ev_idx]
        print(f"\n{'─'*60}")
        print(f"[evidence {ev_idx}] dia_id={ev_id}")
        print(f"  raw: {ev_text[:120]}...")

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
            store, node, query_embedding, query_embedding_hyperbolic,
            label=f"evidence[{ev_idx}] {ev_id}",
        )


if __name__ == "__main__":
    main()
