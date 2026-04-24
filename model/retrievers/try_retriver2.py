"""
用 locomo_qa_test 第 0 条样本的 conversation 建库（非训练集持久化库），
其余行为与 try_retrive.py 对齐：双曲/欧式检索、同一句与 OOD 句打分、父节点多档打印。

  python -m model.retrievers.try_retriver2
"""

from __future__ import annotations

import json
import warnings
from typing import Sequence

import torch

from .cosine_retriver import CosineRetriever
from .hyperbolic_retriver import GeodesicHyperbolicRetriever, MultiParentAngularHyperbolicRetriever, HybridHyperbolicRetriever
from ..hierarchical.hierarchical_manager import create_hierarchical_manager
from ..hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel
from ..llm_inference.data_adapter import extract_interactions
from ..stores.hierarchical_vector_store import HierarchicalVectorStore

LOCOMO_QA_TEST = "/share/home/leiyh5/Memory/data/locomo/locomo_qa_test.json"
PERSIST_DIR = "/share/home/leiyh5/Memory/data/try_retriever2_test_built"
PROJECTOR_PATH = "/share/home/leiyh5/Memory/checkpoints_locomo/hyperbolic_projector_final.pt"
LLM_MODEL_PATH = "/share/home/leiyh5/models/Qwen2.5-7B-Instruct"
DEVICE = "auto"
LLM_BATCH_SIZE = 8

# try_retrive 中这些在 import 后即有值；本脚本在 main 里赋值后再跑检索/打分
store: HierarchicalVectorStore
retriever_euclidean: CosineRetriever
retriever_hyperbolic: HybridHyperbolicRetriever

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\..*",
    category=FutureWarning,
)

LEVEL_ORDER = [
    HierarchyLevel.DOMAIN,
    HierarchyLevel.CATEGORY,
    HierarchyLevel.KEYWORD,
    HierarchyLevel.DIALOGUE,
]


def build_level_hit_sets(level_results: list) -> dict[HierarchyLevel, set[str]]:
    """把每层入围节点整理成按层级索引的集合。"""
    level_hit_sets = {level: set() for level in LEVEL_ORDER}
    for level_result in level_results:
        level_hit_sets[level_result.level] = {
            hit.node.id for hit in level_result.hits
        }
    return level_hit_sets


def load_node_embedding(
    node_id: str | None = None,
    text: str | None = None,
    *,
    level: HierarchyLevel = HierarchyLevel.DIALOGUE,
) -> tuple[Sequence[float], Sequence[float], HierarchicalNode]:
    """
    按 `node_id` 或 `content` 在指定 `level` 的 collection 中查找节点，返回
    (embedding, level_embedding, node)。默认 `level` 为 DIALOGUE，兼容原先「只取对话句」的用法。
    """
    if node_id is not None:
        node = store.get_node(node_id=node_id, level=level)
    elif text is not None:
        node = store.get_node_by_content(content=text, level=level)
    else:
        raise ValueError("需要 node_id 或 text")

    if node is None:
        preview = (text[:120] + "…") if text and len(text) > 120 else text
        raise ValueError(
            f"未找到{level.name} 节点: node_id={node_id!r}, level={level.name!r}, "
            f"text_preview={preview!r}"
        )
    if node.embedding is None:
        raise ValueError(f"节点 {node.id} ({level.name}) 没有 embedding")
    if node.level_embedding is None:
        raise ValueError(f"节点 {node.id} ({level.name}) 没有 level_embedding")
    return node.embedding, node.level_embedding, node


# 旧名保留，避免外部脚本断掉
load_dialogue_embedding = load_node_embedding


def score_query_against_node(
    node: HierarchicalNode,
    query_embedding: Sequence[float],
    query_hyperbolic: torch.Tensor,
) -> tuple[float, float]:
    """
    欧式余弦与双曲成对分；双曲分经 `HybridHyperbolicRetriever._similarity(query, node)`：
    与 `retrieve` 中设置的 hybrid 分界一致时（如先 `retrieve` 或 `set_hybrid_scoring_boundary`），
    按节点层与分界层在「多父外角 / O 点内角」与「测地线」间切换；否则退化为测地线。
    """
    if node.embedding is None or node.level_embedding is None:
        raise ValueError(f"节点 {node.id!r} 缺少 embedding 或 level_embedding")
    cosine_score = retriever_euclidean._cosine_similarity(
        query_embedding, node.embedding
    )
    hyperbolic_score, _ = retriever_hyperbolic._similarity(
        query_hyperbolic,
        node,
    )
    return cosine_score, hyperbolic_score


def get_ancestor_nodes(node: HierarchicalNode) -> list[HierarchicalNode]:
    """获取并按层级顺序排序祖先节点。"""
    ancestors = store.get_ancestors(node.id, node.level)
    return sorted(
        ancestors,
        key=lambda ancestor: LEVEL_ORDER.index(ancestor.level),
    )


def print_ancestor_scores(
    node: HierarchicalNode,
    query_embedding: Sequence[float],
    query_hyperbolic: torch.Tensor,
    title: str,
    level_hit_sets: dict[HierarchyLevel, set[str]],
) -> None:
    """按层级遍历 ancestors，只输出出现在对应层集合中的节点。"""
    ancestors = get_ancestor_nodes(node)
    print(title)
    scored_by_level: dict[HierarchyLevel, list[dict[str, object]]] = {
        level: [] for level in LEVEL_ORDER
    }

    for level in LEVEL_ORDER:
        current_level_hits = level_hit_sets.get(level, set())
        for ancestor in ancestors:
            if ancestor.level != level:
                continue
            if ancestor.id not in current_level_hits:
                continue
            if ancestor.embedding is None or ancestor.level_embedding is None:
                print(f"跳过节点 {ancestor.id}，缺少 embedding 或 level_embedding")
                continue

            cosine_score, hyperbolic_score = score_query_against_node(
                ancestor,
                query_embedding,
                query_hyperbolic,
            )
            scored_by_level[level].append(
                {
                    "id": ancestor.id,
                    "content": ancestor.content,
                    "cosine_score": cosine_score,
                    "hyperbolic_score": hyperbolic_score,
                }
            )

    has_output = False
    for level in LEVEL_ORDER:
        level_items = scored_by_level[level]
        if not level_items:
            continue

        has_output = True
        print("--------------------------------")
        print("层级:", level.name)

        print("按欧式检索得分降序：")
        for item in sorted(level_items, key=lambda item: float(item["cosine_score"]), reverse=True):
            print(
                {
                    "id": item["id"],
                    "content": item["content"],
                    "euclidean_score": item["cosine_score"],
                    "hyperbolic_score": item["hyperbolic_score"],
                }
            )

        print("按双曲检索得分降序：")
        for item in sorted(level_items, key=lambda item: float(item["hyperbolic_score"]), reverse=True):
            print(
                {
                    "id": item["id"],
                    "content": item["content"],
                    "euclidean_score": item["cosine_score"],
                    "hyperbolic_score": item["hyperbolic_score"],
                }
            )

    if has_output:
        print("--------------------------------")


def print_all_ancestor_scores(
    node: HierarchicalNode,
    query_embedding: Sequence[float],
    query_hyperbolic: torch.Tensor,
    title: str,
) -> None:
    """输出所有 ancestors 的得分，再按每层两种分数分别降序打印。"""
    ancestors = get_ancestor_nodes(node)
    print(title)
    scored_by_level: dict[HierarchyLevel, list[dict[str, object]]] = {
        level: [] for level in LEVEL_ORDER
    }

    for ancestor in ancestors:
        if ancestor.embedding is None or ancestor.level_embedding is None:
            print(f"跳过节点 {ancestor.id}，缺少 embedding 或 level_embedding")
            continue

        cosine_score, hyperbolic_score = score_query_against_node(
            ancestor,
            query_embedding,
            query_hyperbolic,
        )
        scored_by_level[ancestor.level].append(
            {
                "id": ancestor.id,
                "content": ancestor.content,
                "cosine_score": cosine_score,
                "hyperbolic_score": hyperbolic_score,
            }
        )

    has_output = False
    for level in LEVEL_ORDER:
        level_items = scored_by_level[level]
        if not level_items:
            continue

        has_output = True
        print("--------------------------------")
        print("层级:", level.name)

        print("按欧式检索得分降序：")
        for item in sorted(level_items, key=lambda item: float(item["cosine_score"]), reverse=True):
            print(
                {
                    "id": item["id"],
                    "content": item["content"],
                    "euclidean_score": item["cosine_score"],
                    "hyperbolic_score": item["hyperbolic_score"],
                }
            )

        print("按双曲检索得分降序：")
        for item in sorted(level_items, key=lambda item: float(item["hyperbolic_score"]), reverse=True):
            print(
                {
                    "id": item["id"],
                    "content": item["content"],
                    "euclidean_score": item["cosine_score"],
                    "hyperbolic_score": item["hyperbolic_score"],
                }
            )

    if has_output:
        print("--------------------------------")


if __name__ == "__main__":
    with open(LOCOMO_QA_TEST, encoding="utf-8") as f:
        first = json.load(f)[0]
    interactions = extract_interactions(first, dataset_name="locomo")
    query_text = "What fields would Caroline be likely to pursue in her educaton?"

    manager = create_hierarchical_manager(
        llm_model_path=LLM_MODEL_PATH,
        persist_directory=PERSIST_DIR,
        device=DEVICE,
        delayed_write=False,
    )
    # manager.batch_process_dialogues(
    #     interactions,
    #     llm_batch_size=LLM_BATCH_SIZE,
    #     generate_embedding=True,
    #     show_progress=True,
    # )
    # manager.flush()

    store = manager.vector_store
    retriever_euclidean = CosineRetriever(vector_store=store)
    retriever_hyperbolic = GeodesicHyperbolicRetriever(
        vector_store=store,
        checkpoint_path=PROJECTOR_PATH,
    )
    retriever_hyperbolic2 = MultiParentAngularHyperbolicRetriever(
        vector_store=store,
        checkpoint_path=PROJECTOR_PATH,
    )

    # ----- 以下与 try_retrive 流程一致（仅 query / 选节点来源不同） -----
    reu = retriever_euclidean.retrieve(
        query_text=query_text,
        top_k=10,
        start_level=HierarchyLevel.DOMAIN,
        target_level=HierarchyLevel.DIALOGUE,
    )
    rhy = retriever_hyperbolic.retrieve(
        query_text=query_text,
        top_k=10,
        start_level=HierarchyLevel.DOMAIN,
        target_level=HierarchyLevel.DIALOGUE,
        adaptive_start_level=False,
    )
    # rhy2 = retriever_hyperbolic2.retrieve(
    #     query_text=query_text,
    #     top_k=7,
    #     start_level=HierarchyLevel.DOMAIN,
    #     target_level=HierarchyLevel.DIALOGUE,
    #     adaptive_start_level=True,
    # )
    result_euclidean = reu.level_results
    result_hyperbolic_levels = rhy.level_results
    # result_hyperbolic_levels2 = rhy2.level_results

    print("欧式检索结果：")
    for i in result_euclidean:
        print("--------------------------------")
        print("该层level", i.level)
        print("该层候选数量", i.candidate_count)
        for j in i.hits:
            _id = j.node.id
            print(f"{_id} 节点信息")
            print(f"节点内容: {j.node.content}")
            print(f"节点父节点: {j.node.parent_ids}")
            print(f"节点子节点: {j.node.child_ids}")
            print(f"节点得分: {j.score}")
        print("--------------------------------")

    print("双曲检索结果：")
    for i in result_hyperbolic_levels:
        print("--------------------------------")
        print("该层level", i.level)
        print("该层候选数量", i.candidate_count)
        for j in i.hits:
            _id = j.node.id
            print(f"{_id} 节点信息")
            print(f"节点内容: {j.node.content}")
            print(f"节点父节点: {j.node.parent_ids}")
            print(f"节点子节点: {j.node.child_ids}")
            print(f"节点得分: {j.score}")
        print("--------------------------------")
        
    # print("多父外角加权双曲检索结果：")
    # for i in result_hyperbolic_levels2:
    #     print("--------------------------------")
    #     print("该层level", i.level)
    #     print("该层候选数量", i.candidate_count)
    #     for j in i.hits:
    #         _id = j.node.id
    #         print(f"{_id} 节点信息")
    #         print(f"节点内容: {j.node.content}")
    #         print(f"节点父节点: {j.node.parent_ids}")
    #         print(f"节点子节点: {j.node.child_ids}")
    #         print(f"节点得分: {j.score}")
    #     print("--------------------------------")

    euclidean_level_hit_sets = build_level_hit_sets(result_euclidean)
    hyperbolic_level_hit_sets = build_level_hit_sets(result_hyperbolic_levels)

    query_embedding = retriever_hyperbolic._prepare_query_embedding(query_text, None)
    query_embedding_hyperbolic = retriever_hyperbolic.project_query(query_embedding)

    query_embedding2 = retriever_hyperbolic._prepare_query_embedding(
        "What was the area of the political and economical union who's special legislative procedure is the consent procedure?",
        None,
    )
    query_embedding_hyperbolic2 = retriever_hyperbolic.project_query(query_embedding2)


    # 与 llm_inference/data_adapter.turn_to_text 一致：时间 + 换行 + speaker: utterance
    # 对应 locomo session_1 中 dia_id D1:3（Caroline）入库时的 dialogue.content
    d1_3_dialogue_content = (
        "1:56 pm on 8 May, 2023\n"
        "Caroline: I went to a LGBTQ support group yesterday and it was so powerful."
    )
    _e, _le, node = load_node_embedding(text=d1_3_dialogue_content)
    node_embedding, node_level_embedding, node = load_node_embedding(node_id=node.id)
    print(node.content)


    keywords = node.parent_ids
    print(keywords)

    for keyword in keywords:
        node_embedding_keyword, node_level_embedding_keyword, node_keyword = load_node_embedding(
            node_id=keyword, level=HierarchyLevel.KEYWORD
        )
        print(node_keyword.content)
        score1, score2 = score_query_against_node(
            node_keyword,
            query_embedding,
            query_embedding_hyperbolic,
        )
        print("关于", keyword, "的得分")
        print("欧式检索得分：", score1)
        print("双曲检索得分：", score2)


    node1_embedding, node1_level_embedding, node1 = load_node_embedding(
        node_id="category_2", level=HierarchyLevel.CATEGORY
    )
    node2_embedding, node2_level_embedding, node2 = load_node_embedding(
        node_id="category_94", level=HierarchyLevel.CATEGORY
    )
    node3_embedding, node3_level_embedding, node3 = load_node_embedding(
        node_id="category_157", level=HierarchyLevel.CATEGORY
    )


    score1, score2 = score_query_against_node(
        node,
        query_embedding,
        query_embedding_hyperbolic,
    )
    print("关于原句的得分")
    print("欧式检索得分：", score1)
    print("双曲检索得分：", score2)


    print("--------------------------------")
    print(node1.parent_ids)
    node4_embedding, node4_level_embedding, node4 = load_node_embedding(
        node_id=node1.parent_ids[0], level=HierarchyLevel.DOMAIN
    )
    score9, score10 = score_query_against_node(
        node4,
        query_embedding,
        query_embedding_hyperbolic,
    )
    print(f"关于category_2的父节点{node4.content}的得分")
    print("欧式检索得分：", score9)
    print("双曲检索得分：", score10)
    score3, score4 = score_query_against_node(
        node1,
        query_embedding,
        query_embedding_hyperbolic,
    )
    print("关于category_2的得分")
    print("欧式检索得分：", score3)
    print("双曲检索得分：", score4)


    print(node2.parent_ids)
    node5_embedding, node5_level_embedding, node5 = load_node_embedding(
        node_id=node2.parent_ids[0], level=HierarchyLevel.DOMAIN
    )
    score11, score12 = score_query_against_node(
        node5,
        query_embedding,
        query_embedding_hyperbolic,
    )
    print(f"关于category_94的父节点{node5.content}的得分")
    print("欧式检索得分：", score11)
    print("双曲检索得分：", score12)
    score5, score6 = score_query_against_node(
        node2,
        query_embedding,
        query_embedding_hyperbolic,
    )
    print("关于category_94的得分")
    print("欧式检索得分：", score5)
    print("双曲检索得分：", score6)


    print(node3.parent_ids)
    node6_embedding, node6_level_embedding, node6 = load_node_embedding(
        node_id=node3.parent_ids[0], level=HierarchyLevel.DOMAIN
    )
    score13, score14 = score_query_against_node(
        node6,
        query_embedding,
        query_embedding_hyperbolic,
    )
    print(f"关于category_157的父节点{node6.content}的得分")
    print("欧式检索得分：", score13)
    print("双曲检索得分：", score14)
    score7, score8 = score_query_against_node(
        node3,
        query_embedding,
        query_embedding_hyperbolic,
    )
    print("关于category_157的得分")
    print("欧式检索得分：", score7)
    print("双曲检索得分：", score8)

    # score3, score4 = score_query_against_node(
    #     node_embedding,
    #     node_level_embedding,
    #     query_embedding2,
    #     query_embedding_hyperbolic2,
    # )
    # print("关于奇怪句子的得分")
    # print("欧式检索得分：", score3)
    # print("双曲检索得分：", score4)

    # print_all_ancestor_scores(
    #     node,
    #     query_embedding,
    #     query_embedding_hyperbolic,
    #     "关于原句的所有父节点得分：",
    # )
    # print_ancestor_scores(
    #     node,
    #     query_embedding,
    #     query_embedding_hyperbolic,
    #     "关于原句的父节点得分（仅保留欧式检索链路中入围的节点）：",
    #     euclidean_level_hit_sets,
    # )
    # print_ancestor_scores(
    #     node,
    #     query_embedding,
    #     query_embedding_hyperbolic,
    #     "关于原句的父节点得分（仅保留双曲检索链路中入围的节点）：",
    #     hyperbolic_level_hit_sets,
    # )
