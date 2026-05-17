"""
可执行入口：读 LoCoMo 测试集 → 按样本清库建库 → QA 检索并（可选）生成，打印问题与输出。

路径类参数在 `parse_args` 里一律给 **绝对路径形式的 default**（可按本机改 default 字符串）；
命令行仍可覆盖。运行：python -m model.llm_inference.session_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.llm_inference.session_llm_inference import SessionMemoryAugmentedLLMInference
from model.llm_inference.session_memory_builder import SessionConversationMemoryBuilder


def print_store_stats(inference: SessionMemoryAugmentedLLMInference, sid: str) -> None:
    stats = inference.manager.get_stats().to_dict()
    print(f"[库状态][{sid}] 总节点数: {stats['total_nodes']}")
    print(
        f"[库状态][{sid}] DOMAIN={stats['domain_count']}, "
        f"CATEGORY={stats['category_count']}, "
        f"KEYWORD={stats['keyword_count']}, "
        f"DIALOGUE={stats['dialogue_count']}"
    )


def print_readable_context(context: Any) -> None:
    print("上下文:")
    if context is None:
        print("None")
        return
    context_text = str(context)
    print(context_text)


def print_fact_dialogue_tree(retrieval_result: Any, vector_store: Any) -> None:
    """打印检索到的 fact 节点及其子 dialogue 节点，展示层级检索路径。"""
    from model.hierarchical.hierarchy_types import HierarchyLevel

    print("\n" + "=" * 70)
    print("检索事实-对话树 (Fact → Dialogue)")
    print("=" * 70)

    # 找到 FACT (KEYWORD) 层的结果
    fact_results = None
    dialogue_results = None
    for lr in (retrieval_result.level_results or []):
        if lr.level == HierarchyLevel.KEYWORD:
            fact_results = lr
        elif lr.level == HierarchyLevel.DIALOGUE:
            dialogue_results = lr

    if fact_results is None or not fact_results.hits:
        print("(未检索到 fact 节点)")
        return

    # 构建 dialogue hit 的索引（按 node.id）
    dialogue_hit_by_id = {}
    if dialogue_results is not None:
        for dh in dialogue_results.hits:
            dialogue_hit_by_id[dh.node.id] = dh

    for fi, fact_hit in enumerate(fact_results.hits):
        fact_node = fact_hit.node
        print(f"\n{'─' * 60}")
        print(f"[Fact #{fi + 1}]  score={fact_hit.score:.4f}")
        print(f"  {fact_node.content}")

        # 打印元数据
        meta = fact_node.metadata or {}
        if meta.get("subject"):
            print(f"  subject={meta['subject']}  predicate={meta.get('predicate', '')}"
                  f"  time={meta.get('time', '')}")

        # 获取 fact 的子 dialogue 节点
        child_ids = fact_node.child_ids or []
        if not child_ids:
            print(f"  (无子节点)")
            continue

        print(f"  子节点 ({len(child_ids)} 个):")
        shown_children = 0
        for child_id in child_ids:
            child_node = vector_store.get_node(child_id, HierarchyLevel.DIALOGUE)
            if child_node is None:
                child_node = vector_store.get_node(child_id)
            if child_node is None:
                continue

            shown_children += 1
            # 检查这个 dialogue 是否在检索结果中
            dh = dialogue_hit_by_id.get(child_id)
            score_str = f"score={dh.score:.4f}" if dh else "not in top-k"

            content_preview = (child_node.content or "").replace("\n", " ")
            if len(content_preview) > 120:
                content_preview = content_preview[:120] + "..."

            print(f"    [{shown_children}] {score_str}  {content_preview}")

    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoCoMo QA + 分层记忆检索/生成（打印）")
    p.add_argument(
        "--data-file",
        type=str,
        default="/share/home/leiyh5/Memory/data/locomo/locomo_qa_test.json",
        help="LoCoMo QA 测试集 JSON",
    )
    p.add_argument(
        "--persist-directory",
        type=str,
        default="/share/home/leiyh5/Memory/data/memory_running_fact",
        help="Chroma 持久化目录",
    )
    p.add_argument(
        "--llm-model-path",
        type=str,
        default=None,
        help="建库 LLM 本地权重路径（transformers 类型）",
    )
    p.add_argument(
        "--llm-model-name",
        type=str,
        default=None,
        help="建库 LLM 模型名（ollama/openai 类型）",
    )
    p.add_argument(
        "--llm-handler-type",
        type=str,
        default="transformers",
        choices=("transformers", "ollama", "openai"),
        help="建库 LLM 后端类型",
    )
    p.add_argument(
        "--llm-api-base",
        type=str,
        default="http://localhost:11434",
        help="建库 LLM API 地址（openai 类型时用）",
    )
    p.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="建库 LLM API key（openai 类型时用，默认读 OPENAI_API_KEY 环境变量）",
    )
    p.add_argument(
        "--projector-checkpoint-path",
        type=str,
        default="/share/home/leiyh5/Memory/checkpoints_locomo_fact/hyperbolic_projector_final.pt",
        help="双曲 projector .pt ",
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="句向量：本地模型用绝对目录；若目录不存在可改为 HF Hub 名并自行覆盖",
    )
    p.add_argument(
        "--retriever-type",
        type=str,
        default="hyperbolic_geodesic",
        choices=(
            "cosine",
            "hyperbolic_geodesic",
            "hyperbolic_angular",
            "hyperbolic_angular_geodesic_hybrid",
        ),
        help=(
            "cosine=余弦；hyperbolic_geodesic=测地线双曲；"
            "hyperbolic_angular=多父外角；"
            "hyperbolic_angular_geodesic_hybrid=按深度分界上外角下测地"
        ),
    )
    p.add_argument("--max-samples", type=int, default=100000000)
    p.add_argument("--max-questions", type=int, default=100000000)
    p.add_argument("--memory-llm-batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--retriever-top-k", type=int, nargs=4, default=[20, 20, 15, 8],
                   help="四层 top-k: [DOMAIN CATEGORY KEYWORD DIALOGUE]")
    p.add_argument("--generation-handler-type", type=str, default="transformers")
    p.add_argument("--generation-model-name", type=str, default=None)
    p.add_argument(
        "--build-mode",
        type=str,
        default="cached",
        choices=("natural", "cached"),
        help=(
            "natural: 保持原逻辑，每轮清库并重建；"
            "cached: 按轮次持久化建库结果，优先复用已有库。"
        ),
    )
    p.add_argument(
        "--out-file",
        type=str,
        default="/share/home/leiyh5/Memory/data/locomo/locomo_qa_test_pred_fact.json",
        help="保存预测结果的 JSON（LoCoMo 评测可读）",
    )
    p.add_argument(
        "--prediction-key",
        type=str,
        default="memory_prediction",
        help="写入每个 qa 条目的预测字段名，如 gpt-4-turbo_prediction",
    )
    p.add_argument(
        "--generation-model-path",
        type=str,
        default=None,
        help="本地生成权重绝对路径，仅在与 --generation-handler-type 同用时生效",
    )
    p.add_argument("--generation-api-base", type=str, default="http://localhost:11434")
    p.add_argument(
        "--generation-api-key",
        type=str,
        default=None,
        help="生成 LLM API key（openai 类型时用，默认读 OPENAI_API_KEY 环境变量）",
    )
    p.add_argument(
        "--memory-unit-mode",
        choices=("keyword", "fact"),
        default="fact",
        help="session third layer mode: keyword keeps old behavior; fact stores factual statements in the third layer.",
    )
    p.add_argument(
        "--extraction-mode",
        choices=("single", "two_stage"),
        default="single",
        help="single: one-shot LLM extraction (fact + SPO); two_stage: stage1=fact only, stage2=per-fact SPO.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_file)
    if not data_path.is_file():
        raise FileNotFoundError(f"测试集不存在: {args.data_file}")

    with open(data_path, encoding="utf-8") as f:
        samples: List[Dict[str, Any]] = json.load(f)

    infer_kw: Dict[str, Any] = dict(
        llm_model_path=args.llm_model_path,
        llm_model_name=args.llm_model_name,
        persist_directory=args.persist_directory,
        embedding_model=args.embedding_model,
        device=args.device,
        retriever_type=args.retriever_type,
        retriever_top_k=args.retriever_top_k,
        generation_handler_type=args.generation_handler_type,
        generation_model_name=args.generation_model_name,
        generation_model_path=args.generation_model_path,
        generation_api_base=args.generation_api_base,
        generation_api_key=args.generation_api_key,
        memory_unit_mode=args.memory_unit_mode,
        extraction_mode=args.extraction_mode,
        llm_handler_type=args.llm_handler_type,
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
    )
    if args.retriever_type in (
        "hyperbolic_geodesic",
        "hyperbolic_angular",
        "hyperbolic_angular_geodesic_hybrid",
    ):
        infer_kw["projector_checkpoint_path"] = args.projector_checkpoint_path

    n_samples = min(len(samples), max(1, args.max_samples))
    output_samples: List[Dict[str, Any]] = []
    if args.build_mode == "natural":
        inference = SessionMemoryAugmentedLLMInference(**infer_kw)
        builder = SessionConversationMemoryBuilder(
            inference.manager,
            llm_batch_size=args.memory_llm_batch_size,
        )
        if inference.retriever:
            print(inference.retriever)
        else:
            print("没找到检索器")

        # 原逻辑：首轮前清库，且每轮结束后清库
        if n_samples > 0:
            builder.clear()
            inference.clear_retriever_cache()

        for si in range(n_samples):
            sample = samples[si]
            sid = sample.get("sample_id", f"index_{si}")
            print(f"\n========== 样本 {sid} ({si + 1}/{n_samples}) ==========")

            builder.build_from_sample(
                sample,
                dataset_name="locomo",
                clear_before_build=False,
                generate_embedding=True,
                session_id=str(sid),
            )
            print("[建库成功！！！]")
            print_store_stats(inference, str(sid))
            inference.clear_retriever_cache()

            out_sample: Dict[str, Any] = {"sample_id": sid, "qa": [dict(q) for q in (sample.get("qa") or [])]}
            qa_list = out_sample["qa"]
            n_q = min(len(qa_list), max(1, args.max_questions))
            for qi in range(n_q):
                item = qa_list[qi]
                question = str(item.get("question", "")).strip()
                print(question)
                if not question:
                    continue
                out = inference.answer(question)
                context = out.get("context")
                gen = out.get("answer")
                item[args.prediction_key] = "" if gen is None else str(gen).strip()
                print_fact_dialogue_tree(
                    out.get("retrieval_result"),
                    inference.manager.vector_store,
                )
                print_readable_context(context)
                print(f"问题: {question}")
                print(f"生成: {gen!r}")

            output_samples.append(out_sample)
            builder.clear()
            inference.clear_retriever_cache()
    else:
        # cached 模式：每轮一个独立持久化目录，存在则直接复用，避免重复建库
        persist_root = Path(args.persist_directory)
        persist_root.mkdir(parents=True, exist_ok=True)
        shared_generation_handler = None

        for si in range(n_samples):
            sample = samples[si]
            sid = sample.get("sample_id", f"index_{si}")
            round_dir = persist_root / f"round_{si + 1}_{sid}"
            round_dir.mkdir(parents=True, exist_ok=True)
            build_marker = round_dir / "build_complete.json"
            marker_matches_mode = False
            if build_marker.exists():
                try:
                    with open(build_marker, encoding="utf-8") as f:
                        marker_data = json.load(f)
                        marker_matches_mode = (
                            marker_data.get("memory_unit_mode", "keyword") == args.memory_unit_mode
                            and marker_data.get("extraction_mode", "single") == args.extraction_mode
                        )
                except (OSError, json.JSONDecodeError):
                    marker_matches_mode = False

            print(f"\n========== 样本 {sid} ({si + 1}/{n_samples}) ==========")
            print(f"[cached 模式] 轮次目录: {round_dir}")

            round_infer_kw = dict(infer_kw)
            round_infer_kw["persist_directory"] = str(round_dir)

            # 已有缓存时无需加载建库 LLM（节省启动与显存）
            if marker_matches_mode:
                round_infer_kw["llm_model_path"] = None
                round_infer_kw["llm_model_name"] = None

            # 复用生成器，避免每轮重复加载生成模型
            if shared_generation_handler is not None:
                round_infer_kw["generation_handler_type"] = None
                round_infer_kw["generation_model_name"] = None
                round_infer_kw["generation_model_path"] = None

            inference = SessionMemoryAugmentedLLMInference(**round_infer_kw)
            if shared_generation_handler is None:
                shared_generation_handler = inference.generation_handler
            else:
                inference.generation_handler = shared_generation_handler

            builder = SessionConversationMemoryBuilder(
                inference.manager,
                llm_batch_size=args.memory_llm_batch_size,
            )

            if marker_matches_mode:
                print("[复用缓存建库] 检测到已持久化结果，跳过建库。")
                print_store_stats(inference, str(sid))
            else:
                print("[首次建库] 未检测到缓存，开始建库并持久化。")
                builder.clear()
                inference.clear_retriever_cache()
                builder.build_from_sample(
                    sample,
                    dataset_name="locomo",
                    clear_before_build=False,
                    generate_embedding=True,
                    session_id=str(sid),
                )
                marker_payload = {
                    "sample_id": sid,
                    "round_index": si + 1,
                    "persist_directory": str(round_dir),
                    "memory_unit_mode": args.memory_unit_mode,
                    "extraction_mode": args.extraction_mode,
                }
                with open(build_marker, "w", encoding="utf-8") as f:
                    json.dump(marker_payload, f, ensure_ascii=False, indent=2)
                print("[建库成功并已持久化]")
                print_store_stats(inference, str(sid))
                inference.clear_retriever_cache()

            out_sample: Dict[str, Any] = {"sample_id": sid, "qa": [dict(q) for q in (sample.get("qa") or [])]}
            qa_list = out_sample["qa"]
            n_q = min(len(qa_list), max(1, args.max_questions))
            for qi in range(n_q):
                item = qa_list[qi]
                question = str(item.get("question", "")).strip()
                if not question:
                    continue
                out = inference.answer(question)
                context = out.get("context")
                gen = out.get("answer")
                item[args.prediction_key] = "" if gen is None else str(gen).strip()
                print_fact_dialogue_tree(
                    out.get("retrieval_result"),
                    inference.manager.vector_store,
                )
                # print_readable_context(context)
                print(f"问题: {question}")
                print(f"生成: {gen!r}")

            output_samples.append(out_sample)
            inference.clear_retriever_cache()

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: {out_path}")
    print(f"预测字段: {args.prediction_key}")

if __name__ == "__main__":
    main()
