import warnings
from typing import Sequence

import torch

from .cosine_retriver import CosineRetriever
from .hyperbolic_retriver import GeodesicHyperbolicRetriever
from ..stores.hierarchical_vector_store import HierarchicalVectorStore
from ..encoders.embedding_encoder import EmbeddingEncoder
from ..hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel

PERSIST_DIR="/share/home/leiyh5/Memory/data/hierarchical_memory_locomo"
projector_path = "/share/home/leiyh5/Memory/checkpoints_locomo/hyperbolic_projector_final.pt"
query_text="Who did Maria have dinner with on May 3, 2023?"
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\..*",
    category=FutureWarning,
)

embedding_encoder = EmbeddingEncoder()
store = HierarchicalVectorStore(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_encoder.generate_embedding,
    delayed_write=False,
)

retriever_euclidean = CosineRetriever(
    vector_store=store,
)
retriever_hyperbolic = GeodesicHyperbolicRetriever(
    vector_store=store,
    checkpoint_path=projector_path,
)

LEVEL_ORDER = [
    HierarchyLevel.DOMAIN,
    HierarchyLevel.CATEGORY,
    HierarchyLevel.KEYWORD,
    HierarchyLevel.DIALOGUE,
]


result_euclidean = retriever_euclidean.retrieve(
    query_text=query_text,
    top_k=5,
    start_level=HierarchyLevel.DOMAIN,
    target_level=HierarchyLevel.DIALOGUE,
)


result_hyperbolic = retriever_hyperbolic.retrieve(
    query_text=query_text,
    top_k=5,
    start_level=HierarchyLevel.DOMAIN,
    target_level=HierarchyLevel.DIALOGUE,
    adaptive_start_level = True
)

result_hyperbolic = result_hyperbolic.level_results
result_euclidean = result_euclidean.level_results


print("欧式检索结果：")
for i in result_euclidean:
    print("--------------------------------")
    print("该层level",i.level)
    print("该层候选数量",i.candidate_count)
    for j in i.hits:
        id = j.node.id
        print(f"{id} 节点信息")
        print(f"节点内容: {j.node.content}")
        print(f"节点父节点: {j.node.parent_ids}")
        print(f"节点子节点: {j.node.child_ids}")
        print(f"节点得分: {j.score}")
    print("--------------------------------")


print("双曲检索结果：")
for i in result_hyperbolic:
    print("--------------------------------")
    print("该层level",i.level)
    print("该层候选数量",i.candidate_count)
    for j in i.hits:
        id = j.node.id
        print(f"{id} 节点信息")
        print(f"节点内容: {j.node.content}")
        print(f"节点父节点: {j.node.parent_ids}")
        print(f"节点子节点: {j.node.child_ids}")
        print(f"节点得分: {j.score}")
    print("--------------------------------")


def build_level_hit_sets(level_results: list) -> dict[HierarchyLevel, set[str]]:
    """把每层入围节点整理成按层级索引的集合。"""
    level_hit_sets = {level: set() for level in LEVEL_ORDER}
    for level_result in level_results:
        level_hit_sets[level_result.level] = {
            hit.node.id for hit in level_result.hits
        }
    return level_hit_sets


def load_dialogue_embedding(node_id=None,text=None) -> tuple[Sequence[float], Sequence[float], HierarchicalNode]:
    """统一处理从存储读出的 list / numpy.ndarray embedding。"""
    if node_id is not None:
        node = store.get_node(node_id=node_id, level=HierarchyLevel.DIALOGUE)
    elif text is not None:
        node = store.get_node_by_content(content=text, level=HierarchyLevel.DIALOGUE)

    if node is None:
        raise ValueError(f"未找到节点: {node_id}")
    if node.embedding is None:
        raise ValueError(f"节点 {node_id} 没有 embedding")
    if node.level_embedding is None:
        raise ValueError(f"节点 {node_id} 没有 level_embedding")
    return node.embedding, node.level_embedding, node


def score_query_against_node(
    node: HierarchicalNode,
    query_embedding: Sequence[float],
    query_hyperbolic: torch.Tensor,
) -> tuple[float, float]:
    """分别计算欧式余弦与双曲 ``_similarity(query, node)`` 分数。"""
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


euclidean_level_hit_sets = build_level_hit_sets(result_euclidean)
hyperbolic_level_hit_sets = build_level_hit_sets(result_hyperbolic)

query_embedding = retriever_hyperbolic._prepare_query_embedding(query_text, None)
query_embedding_hyperbolic = retriever_hyperbolic.project_query(query_embedding)

query_embedding2 = retriever_hyperbolic._prepare_query_embedding(
    "What was the area of the political and economical union who's special legislative procedure is the consent procedure?",
    None,
)
query_embedding_hyperbolic2 = retriever_hyperbolic.project_query(query_embedding2)

node_embedding,node_level_embedding,node = load_dialogue_embedding(text = "Wow, John! It's great when you have that kind of support. My mom and I made some dinner together last night!")
print(node.content)

score1, score2 = score_query_against_node(
    node,
    query_embedding,
    query_embedding_hyperbolic,
)
print("关于原句的得分")
print("欧式检索得分：", score1)
print("双曲检索得分：", score2)

score3, score4 = score_query_against_node(
    node,
    query_embedding2,
    query_embedding_hyperbolic2,
)
print("关于奇怪句子的得分")
print("欧式检索得分：", score3)
print("双曲检索得分：", score4)

print_all_ancestor_scores(
    node,
    query_embedding,
    query_embedding_hyperbolic,
    "关于原句的所有父节点得分：",
)
print_ancestor_scores(
    node,
    query_embedding,
    query_embedding_hyperbolic,
    "关于原句的父节点得分（仅保留欧式检索链路中入围的节点）：",
    euclidean_level_hit_sets,
)
print_ancestor_scores(
    node,
    query_embedding,
    query_embedding_hyperbolic,
    "关于原句的父节点得分（仅保留双曲检索链路中入围的节点）：",
    hyperbolic_level_hit_sets,
)
