"""
可执行入口：读 LoCoMo 测试集 → 按样本清库建库 → QA 检索并（可选）生成，打印问题与输出。

路径类参数在 `parse_args` 里一律给 **绝对路径形式的 default**（可按本机改 default 字符串）；
命令行仍可覆盖。运行：python -m model.llm_inference.run
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

from model.llm_inference.llm_inference import MemoryAugmentedLLMInference
from model.llm_inference.memory_builder import ConversationMemoryBuilder


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
        default="/share/home/leiyh5/Memory/data/memory_running",
        help="Chroma 持久化目录",
    )
    p.add_argument(
        "--llm-model-path",
        type=str,
        default="/share/home/leiyh5/models/Qwen2.5-7B-Instruct",
        help="建库用 DialogueAnalyzer",
    )
    p.add_argument(
        "--projector-checkpoint-path",
        type=str,
        default="/share/home/leiyh5/Memory/checkpoints_locomo/hyperbolic_projector_final.pt",
        help="双曲 projector .pt ",
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default="/share/home/leiyh5/Memory/models/sentence-transformers_all-MiniLM-L6-v2",
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
    p.add_argument("--max-samples", type=int, default=1)
    p.add_argument("--max-questions", type=int, default=1)
    p.add_argument("--memory-llm-batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--retriever-top-k", type=int, default=7)
    p.add_argument("--generation-handler-type", type=str, default="transformers")
    p.add_argument("--generation-model-name", type=str, default=None)
    p.add_argument(
        "--generation-model-path",
        type=str,
        default="/share/home/leiyh5/models/Qwen2.5-7B-Instruct",
        help="本地生成权重绝对路径，仅在与 --generation-handler-type 同用时生效",
    )
    p.add_argument("--generation-api-base", type=str, default="http://localhost:11434")
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
        persist_directory=args.persist_directory,
        embedding_model=args.embedding_model,
        device=args.device,
        retriever_type=args.retriever_type,
        retriever_top_k=args.retriever_top_k,
        generation_handler_type=args.generation_handler_type,
        generation_model_name=args.generation_model_name,
        generation_model_path=args.generation_model_path,
        generation_api_base=args.generation_api_base,
    )
    if args.retriever_type in (
        "hyperbolic_geodesic",
        "hyperbolic_angular",
        "hyperbolic_angular_geodesic_hybrid",
    ):
        infer_kw["projector_checkpoint_path"] = args.projector_checkpoint_path

    inference = MemoryAugmentedLLMInference(**infer_kw)
    builder = ConversationMemoryBuilder(
        inference.manager,
        llm_batch_size=args.memory_llm_batch_size,
    )
    if inference.retriever:
        print(inference.retriever)
    else:
        print("没找到检索器")

    n_samples = min(len(samples), max(1, args.max_samples))
    # 清库：循环首行之前清一次（首样本建库前需空库），每一样本 QA 后清空（等同“下一样本开始再清”）。
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
            show_progress=True,
        )
        print(
            f"[建库成功！！！]"
        )
        print(builder.manager.vector_store.get_stats())
        inference.clear_retriever_cache()

        qa_list = sample.get("qa") or []
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
            print(f"上下文: {context!r}")
            print(f"问题: {question}")
            print(f"生成: {gen!r}")

        builder.clear()
        inference.clear_retriever_cache()

if __name__ == "__main__":
    main()
