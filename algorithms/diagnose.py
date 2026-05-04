"""
诊断脚本：在任何检索器上跑一个查询，统计各层 top-10 的关键指标。

用法：
    python algorithms/diagnose.py <query_text> [--retriever_type ...] [--checkpoint ...]

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
from pathlib import Path

import model.retrievers.try_retriver2 as tr2
from algorithms.locomo_qa_evidence import reference_dialogue_for_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default = "What did Caroline research?",help="查询文本（与 locomo_qa_test.json 中某条 question 一致时可自动拼 gold evidence）")
    parser.add_argument(
        "--qa_json",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data/locomo/locomo_qa_test.json"),
        help="LoCoMo QA+conversation，用于按 question 匹配 evidence 并生成与 store 一致的参考文本",
    )
    parser.add_argument("--retriever_type", type=str, default="hyperbolic_angular",
                        choices=["cosine", "hyperbolic_geodesic", "hyperbolic_angular",
                                 "hyperbolic_angular_geodesic_hybrid"])
    parser.add_argument("--checkpoint", type=str, required=False, default="/share/home/leiyh5/Memory/checkpoints_locomo_total/hyperbolic_projector_final.pt")
    parser.add_argument("--persist_dir", type=str,default="/share/home/leiyh5/Memory/data/try_retriever2_test_built",
                        help="vector store 持久化目录")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="用于生成 query embedding 的模型名（需与 projector 输入维度匹配）。",
    )
    parser.add_argument("--top_k", type=int, nargs=4, default=[20, 15, 10, 8])
    parser.add_argument("--query_prefix", type=str, default=None,
                        help="v4_query_prefix: 可选的 query 前缀")
    args = parser.parse_args()

    # 延迟导入以便在 v* 被 apply 后生效
    from model.llm_inference.llm_inference import MemoryAugmentedLLMInference
    from model.hierarchical.hierarchy_types import HierarchyLevel

    inf = MemoryAugmentedLLMInference(
        persist_directory=args.persist_dir,
        retriever_type=args.retriever_type,
        projector_checkpoint_path=args.checkpoint,
        retriever_top_k=args.top_k,
        embedding_model=args.embedding_model,
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

    print("=" * 80)
    store = inf.manager.vector_store
    # try_retriver2 的工具函数依赖模块级全局变量；被 import 使用时需要手动绑定。
    tr2.store = store
    if getattr(inf, "retriever", None) is not None:
        tr2.retriever_hyperbolic = inf.retriever
    tr2.retriever_euclidean = tr2.CosineRetriever(vector_store=store)

    query_embedding = tr2.retriever_hyperbolic._prepare_query_embedding(args.query, None)
    query_embedding_hyperbolic = tr2.retriever_hyperbolic.project_query(query_embedding)

    try:
        reference_dialogue_content, gold_evidence = reference_dialogue_for_query(
            args.query, Path(args.qa_json)
        )
    except (FileNotFoundError, LookupError, KeyError, ValueError) as e:
        print(f"[locomo_qa_evidence] {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[gold evidence] {gold_evidence}")
    _e, _le, node = tr2.load_node_embedding(text=reference_dialogue_content)
    node_embedding, node_level_embedding, node = tr2.load_node_embedding(node_id=node.id)
    print(node.content)


    keywords = node.parent_ids
    print(keywords)

    for keyword in keywords:
        node_embedding_keyword, node_level_embedding_keyword, node_keyword = tr2.load_node_embedding(
            node_id=keyword, level=HierarchyLevel.KEYWORD
        )
        print(node_keyword.content)
        score1, score2 = tr2.score_query_against_node(
            node_keyword,
            query_embedding,
            query_embedding_hyperbolic,
        )
        print("关于", keyword, node_keyword.content, "的得分")
        print("欧式检索得分：", score1)
        print("双曲检索得分：", score2)

    score1, score2 = tr2.score_query_against_node(
        node,
        query_embedding,
        query_embedding_hyperbolic,
    )
    print(f"关于原句的得分{node.id,node.content}")
    print("欧式检索得分：", score1)
    print("双曲检索得分：", score2)

    # 从当前节点的 keyword 父节点出发，遍历所有 category 父节点，再遍历对应 domain 父节点
    category_ids: list[str] = []
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
        score_c_e, score_c_h = tr2.score_query_against_node(
            node_category,
            query_embedding,
            query_embedding_hyperbolic,
        )
        print(f"关于{category_id}({node_category.content})的得分")
        print("欧式检索得分：", score_c_e)
        print("双曲检索得分：", score_c_h)
        print(f"{category_id} 的 domain 父节点: {node_category.parent_ids}")

        for domain_id in node_category.parent_ids:
            if domain_id in seen_domain_ids:
                continue
            seen_domain_ids.add(domain_id)
            _, _, node_domain = tr2.load_node_embedding(
                node_id=domain_id, level=HierarchyLevel.DOMAIN
            )
            score_d_e, score_d_h = tr2.score_query_against_node(
                node_domain,
                query_embedding,
                query_embedding_hyperbolic,
            )
            print(f"关于{category_id}的父节点{node_domain.content}({domain_id})的得分")
            print("欧式检索得分：", score_d_e)
            print("双曲检索得分：", score_d_h)

    


if __name__ == "__main__":
    main()
