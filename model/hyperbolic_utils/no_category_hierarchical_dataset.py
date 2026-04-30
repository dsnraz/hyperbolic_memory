from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from model.hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel
from torch.utils.data import DataLoader

from model.hyperbolic_utils.hierarchical_dataset import (
    SubtreeBatch,
    SubtreeDataset,
    SubtreeSampler,
    subtree_collate_fn,
)


NO_CATEGORY_LEVEL_PAIRS = [
    ("DOMAIN", "KEYWORD"),
    ("KEYWORD", "DIALOGUE"),
]


class NoCategorySubtreeDataset(SubtreeDataset):
    def __init__(
        self,
        nodes_by_level: Dict[str, List[HierarchicalNode]],
        embedding_dim: int,
        device=None,
        num_iterations: int = 1000,
        num_parents_per_batch: int = 16,
        num_children_per_parent: int = 4,
        max_children_per_parent: int = 10,
        level_pair: Tuple[str, str] | None = None,
        load_feats_by_level: bool = False,
        use_level_embedding: bool = False,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.device = device
        self.num_iterations = num_iterations
        self.level_pair = level_pair
        self.load_feats_by_level = load_feats_by_level
        self.use_level_embedding = use_level_embedding
        self.samplers = {}
        for parent_level, child_level in NO_CATEGORY_LEVEL_PAIRS:
            parent_nodes = nodes_by_level.get(parent_level, [])
            child_nodes = nodes_by_level.get(child_level, [])
            if parent_nodes and child_nodes:
                self.samplers[(parent_level, child_level)] = SubtreeSampler(
                    parent_nodes=parent_nodes,
                    child_nodes=child_nodes,
                    parent_level=parent_level,
                    child_level=child_level,
                    embedding_dim=embedding_dim,
                    device=device,
                    num_parents_per_batch=num_parents_per_batch,
                    num_children_per_parent=num_children_per_parent,
                    max_children_per_parent=max_children_per_parent,
                    use_level_embedding=use_level_embedding,
                )
        self.available_level_pairs = list(self.samplers.keys())
        self._reset_level_pair_sampling_pool()
        self.feats_by_level = {}
        if load_feats_by_level:
            self._load_feats_by_level(nodes_by_level)


def create_no_category_subtree_dataloader(
    nodes_by_level: Dict[str, List[HierarchicalNode]],
    embedding_dim: int,
    batch_size: int = 1,
    device=None,
    num_iterations: int = 1000,
    num_parents_per_batch: int = 16,
    num_children_per_parent: int = 4,
    max_children_per_parent: int = 10,
    level_pair: Tuple[str, str] | None = None,
    load_feats_by_level: bool = False,
    use_level_embedding: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
):
    dataset = NoCategorySubtreeDataset(
        nodes_by_level=nodes_by_level,
        embedding_dim=embedding_dim,
        device=device,
        num_iterations=num_iterations,
        num_parents_per_batch=num_parents_per_batch,
        num_children_per_parent=num_children_per_parent,
        max_children_per_parent=max_children_per_parent,
        level_pair=level_pair,
        load_feats_by_level=load_feats_by_level,
        use_level_embedding=use_level_embedding,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=subtree_collate_fn,
    )


def extract_no_category_nodes_from_store(
    vector_store,
    level_pair_index: Optional[int] = None,
) -> Dict[str, List[HierarchicalNode]]:
    mapping = {
        None: ["DOMAIN", "KEYWORD", "DIALOGUE"],
        1: ["DOMAIN", "KEYWORD"],
        2: ["KEYWORD", "DIALOGUE"],
    }
    if level_pair_index not in mapping:
        raise ValueError(f"unsupported level_pair_index: {level_pair_index}")
    result: Dict[str, List[HierarchicalNode]] = {}
    for level_name in mapping[level_pair_index]:
        result[level_name] = vector_store.get_nodes_by_level(HierarchyLevel[level_name])
    return result


__all__ = [
    "NO_CATEGORY_LEVEL_PAIRS",
    "SubtreeBatch",
    "NoCategorySubtreeDataset",
    "SubtreeSampler",
    "create_no_category_subtree_dataloader",
    "extract_no_category_nodes_from_store",
    "subtree_collate_fn",
]
