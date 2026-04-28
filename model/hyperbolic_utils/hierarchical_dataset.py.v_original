"""
多层级双曲训练数据集构建。

采用子树采样 + 批次内负采样策略：
1. 随机采样 k 个父节点
2. 每个父节点随机采样若干子节点
3. batch内其他父节点的子节点作为负样本

提供 SubtreeDataset 子树采样数据集：
- 可通过设置 load_feats_by_level=True 加载层级对节点特征到 GPU
- 可通过设置 level_pair 固定层级对，或每次随机选择
- 可通过设置大采样参数一次性采样所有节点（模拟全局模式）
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from model.hierarchical.hierarchy_types import (
    HierarchicalNode, 
    HierarchyLevel,
)


# ============================================================================
# 层级对常量定义
# ============================================================================

LEVEL_PAIRS = [
    ("DOMAIN", "CATEGORY"),      # 索引 1
    ("CATEGORY", "KEYWORD"),     # 索引 2
    ("KEYWORD", "DIALOGUE"),     # 索引 3
]


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class SubtreeBatch:
    """
    子树采样批次数据结构。
    
    属性:
        parent_level: 父层级名称
        child_level: 子层级名称
        parent_feats: 父节点特征张量，形状 (k, D)
        child_feats: 子节点特征张量，形状 (k*m, D)
        parent_child_mask: 多父归属矩阵，形状 (k, k*m)
        parent_child_map: 兼容旧逻辑的单父映射，占位字段，形状 (k*m,)
        parent_ids: 父节点ID列表
        child_ids: 子节点ID列表
        n_parent: 父节点数量
        n_child: 子节点数量
    """
    parent_level: str
    child_level: str
    parent_feats: torch.Tensor
    child_feats: torch.Tensor
    parent_child_mask: torch.Tensor
    parent_child_map: torch.Tensor
    parent_ids: List[str]
    child_ids: List[str]
    n_parent: int
    n_child: int
    
    def to_dict(self) -> Dict:
        """转换为字典格式。"""
        return {
            'parent_level': self.parent_level,
            'child_level': self.child_level,
            'n_parent': self.n_parent,
            'n_child': self.n_child,
        }
    
    def get_batch_attrs(self):
        """
        统一获取批次数据的属性。
        
        支持 SubtreeBatch 和 GlobalLevelBatch。
        """
        return {
            'parent_level': self.parent_level,
            'child_level': self.child_level,
            'parent_feats': self.parent_feats,
            'child_feats': self.child_feats,
            'parent_child_mask': self.parent_child_mask,
            # 多父训练应优先使用 parent_child_mask。
            'parent_child_map': self.parent_child_map,
            'n_parent': self.n_parent,
            'n_child': self.n_child,
        }


# ============================================================================
# 子树采样器
# ============================================================================

class SubtreeSampler:
    """
    子树采样器：从层级结构中采样子树形成训练批次。
    
    属性:
        parent_level: 父层级名称
        child_level: 子层级名称
        embedding_dim: 嵌入向量维度
        device: 计算设备
        parent_nodes: 原始父节点列表
        child_nodes: 原始子节点列表
        parent_to_children: 父节点到子节点的映射
        valid_parent_indices: 有效父节点索引列表
    """
    
    def __init__(
        self,
        parent_nodes: List[HierarchicalNode],
        child_nodes: List[HierarchicalNode],
        parent_level: str,
        child_level: str,
        embedding_dim: int,
        device: torch.device = None,
        num_parents_per_batch: int = 16,
        num_children_per_parent: int = 4,
        max_children_per_parent: int = 10,
        use_level_embedding: bool = False,
    ):
        self.parent_level = parent_level
        self.child_level = child_level
        self.embedding_dim = embedding_dim
        self.device = device or torch.device('cpu')
        
        self.num_parents_per_batch = num_parents_per_batch
        self.num_children_per_parent = num_children_per_parent
        self.max_children_per_parent = max_children_per_parent
        self.use_level_embedding = use_level_embedding
        
        self.parent_nodes = parent_nodes
        self.child_nodes = child_nodes
        
        self._build_index_mappings()
        self._reset_parent_sampling_pool()
        self._reset_child_sampling_pools()

    def _get_node_embedding(self, node: HierarchicalNode) -> Optional[List[float]]:
        """根据开关选择节点使用的 embedding。"""
        if self.use_level_embedding and node.level_embedding is not None:
            return node.level_embedding
        return node.embedding
    
    def _build_index_mappings(self):
        """构建索引映射和父子关系。"""
        self.parent_id_to_idx = {node.id: i for i, node in enumerate(self.parent_nodes)}
        self.child_id_to_idx = {node.id: i for i, node in enumerate(self.child_nodes)}
        
        self.parent_to_children: Dict[int, List[int]] = {}
        
        for parent_idx, parent_node in enumerate(self.parent_nodes):
            child_indices = []
            for child_id in parent_node.child_ids:
                if child_id in self.child_id_to_idx:
                    child_indices.append(self.child_id_to_idx[child_id])
            self.parent_to_children[parent_idx] = child_indices
        
        self.valid_parent_indices = [
            idx for idx, children in self.parent_to_children.items()
            if len(children) > 0
        ]
        
        self.total_valid_parents = len(self.valid_parent_indices)
        self.total_children = sum(len(c) for c in self.parent_to_children.values())

    def _reset_parent_sampling_pool(self) -> None:
        """重置父节点采样池，尽量在一轮内覆盖所有有效父节点。"""
        self.parent_sampling_pool = self.valid_parent_indices.copy()
        random.shuffle(self.parent_sampling_pool)
        self.parent_sampling_cursor = 0

    def _reset_child_sampling_pools(self) -> None:
        """为每个父节点初始化子节点覆盖式采样池。"""
        self.child_sampling_pools: Dict[int, List[int]] = {}
        self.child_sampling_cursors: Dict[int, int] = {}

        for parent_idx, child_indices in self.parent_to_children.items():
            child_pool = child_indices.copy()
            random.shuffle(child_pool)
            self.child_sampling_pools[parent_idx] = child_pool
            self.child_sampling_cursors[parent_idx] = 0

    def _sample_parent_indices_with_coverage(self) -> List[int]:
        """
        按覆盖优先的方式采样父节点。

        一轮内尽量不重复，轮空后再重新打乱开始下一轮。
        """
        selected_parent_indices: List[int] = []

        while len(selected_parent_indices) < self.num_parents_per_batch:
            remaining = self.num_parents_per_batch - len(selected_parent_indices)
            available = len(self.parent_sampling_pool) - self.parent_sampling_cursor

            if available == 0:
                self._reset_parent_sampling_pool()
                available = len(self.parent_sampling_pool) - self.parent_sampling_cursor

            take_n = min(remaining, available)
            start = self.parent_sampling_cursor
            end = start + take_n
            selected_parent_indices.extend(self.parent_sampling_pool[start:end])
            self.parent_sampling_cursor = end

        return selected_parent_indices

    def _sample_children_with_coverage(
        self,
        parent_idx: int,
        num_to_sample: int,
    ) -> List[int]:
        """
        按覆盖优先的方式采样某个父节点的子节点。

        对同一父节点，尽量先把所有子节点轮一遍，再开始下一轮。
        """
        if num_to_sample <= 0:
            return []

        available_children = self.parent_to_children[parent_idx]
        if not available_children:
            return []

        selected_children: List[int] = []
        pool = self.child_sampling_pools[parent_idx]

        while len(selected_children) < num_to_sample:
            cursor = self.child_sampling_cursors[parent_idx]
            available = len(pool) - cursor

            if available == 0:
                pool = available_children.copy()
                random.shuffle(pool)
                self.child_sampling_pools[parent_idx] = pool
                self.child_sampling_cursors[parent_idx] = 0
                cursor = 0
                available = len(pool)

            take_n = min(num_to_sample - len(selected_children), available)
            selected_children.extend(pool[cursor:cursor + take_n])
            self.child_sampling_cursors[parent_idx] = cursor + take_n

        return selected_children
    
    def num_batches(self) -> int:
        """估算可生成的批次数量。"""
        return self.total_valid_parents // self.num_parents_per_batch
    
    def sample_batch(self, dynamic_sampling: bool = True) -> Optional[SubtreeBatch]:
        """采样一个训练批次。"""
        if len(self.valid_parent_indices) < self.num_parents_per_batch:
            return None
        
        selected_parent_indices = self._sample_parent_indices_with_coverage()
        
        selected_child_indices = []
        parent_child_relations = []
        
        for local_parent_idx, global_parent_idx in enumerate(selected_parent_indices):
            available_children = self.parent_to_children[global_parent_idx]
            
            if dynamic_sampling:
                num_to_sample = min(len(available_children), self.max_children_per_parent)
            else:
                num_to_sample = min(len(available_children), self.num_children_per_parent)
            
            sampled_children = self._sample_children_with_coverage(
                global_parent_idx,
                num_to_sample,
            )
            
            for child_idx in sampled_children:
                selected_child_indices.append(child_idx)
                parent_child_relations.append((local_parent_idx, child_idx))
        
        if not selected_child_indices:
            return None
        
        selected_child_indices = list(set(selected_child_indices))
        n_parent = len(selected_parent_indices)
        n_child = len(selected_child_indices)
        
        child_global_to_local = {
            global_idx: local_idx 
            for local_idx, global_idx in enumerate(selected_child_indices)
        }
        
        parent_child_mask = torch.zeros(n_parent, n_child, dtype=torch.float32)
        
        for local_parent_idx, global_child_idx in parent_child_relations:
            if global_child_idx in child_global_to_local:
                local_child_idx = child_global_to_local[global_child_idx]
                parent_child_mask[local_parent_idx, local_child_idx] = 1.0
        
        # 保留该字段仅用于兼容旧接口；多父训练语义以后续的 parent_child_mask 为准。
        parent_child_map = parent_child_mask.argmax(dim=0)
        
        # 收集特征
        parent_feats_list = []
        for parent_idx in selected_parent_indices:
            node = self.parent_nodes[parent_idx]
            node_embedding = self._get_node_embedding(node)
            feat = torch.tensor(node_embedding, dtype=torch.float32) if node_embedding is not None else torch.zeros(self.embedding_dim)
            parent_feats_list.append(feat)
        
        child_feats_list = []
        for child_idx in selected_child_indices:
            node = self.child_nodes[child_idx]
            node_embedding = self._get_node_embedding(node)
            feat = torch.tensor(node_embedding, dtype=torch.float32) if node_embedding is not None else torch.zeros(self.embedding_dim)
            child_feats_list.append(feat)
        
        parent_feats = torch.stack(parent_feats_list).to(self.device)
        child_feats = torch.stack(child_feats_list).to(self.device)
        parent_child_mask = parent_child_mask.to(self.device)
        parent_child_map = parent_child_map.to(self.device)
        
        return SubtreeBatch(
            parent_level=self.parent_level,
            child_level=self.child_level,
            parent_feats=parent_feats,
            child_feats=child_feats,
            parent_child_mask=parent_child_mask,
            parent_child_map=parent_child_map,
            parent_ids=[self.parent_nodes[i].id for i in selected_parent_indices],
            child_ids=[self.child_nodes[i].id for i in selected_child_indices],
            n_parent=n_parent,
            n_child=n_child,
        )


# ============================================================================
# 子树采样Dataset
# ============================================================================

class SubtreeDataset(Dataset):
    """
    子树采样数据集。
    
    属性:
        embedding_dim: 嵌入向量维度
        device: 计算设备
        num_iterations: 总迭代次数
        level_pair: 固定的层级对
        use_level_embedding: 是否使用带层级前缀的 embedding
        samplers: 各层级对的采样器字典
        feats_by_level: 层级对节点特征（仅在load_feats_by_level=True时加载）
    """
    
    def __init__(
        self,
        nodes_by_level: Dict[str, List[HierarchicalNode]],
        embedding_dim: int,
        device: torch.device = None,
        num_iterations: int = 1000,
        num_parents_per_batch: int = 16,
        num_children_per_parent: int = 4,
        max_children_per_parent: int = 10,
        level_pair: Tuple[str, str] = None,
        load_feats_by_level: bool = False,
        use_level_embedding: bool = False,
    ):
        """
        初始化子树采样数据集。
        
        参数:
            nodes_by_level: 各层级节点字典
            embedding_dim: 嵌入向量维度
            device: 计算设备
            num_iterations: 总迭代次数
            num_parents_per_batch: 每batch采样的父节点数
            num_children_per_parent: 每父节点固定子节点数
            max_children_per_parent: 动态采样时的最大子节点数
            level_pair: 固定层级对，如 ("CATEGORY", "KEYWORD")
                        None表示每次随机选择
            load_feats_by_level: 是否加载层级对节点特征到GPU
                        True: 根据 level_pair 加载对应层级对的节点特征
                        False: 不加载，节省显存
            use_level_embedding: 是否优先使用 node.level_embedding 作为训练输入
        """
        self.embedding_dim = embedding_dim
        self.device = device or torch.device('cpu')
        self.num_iterations = num_iterations
        self.level_pair = level_pair
        self.load_feats_by_level = load_feats_by_level
        self.use_level_embedding = use_level_embedding
        
        self.samplers: Dict[Tuple[str, str], SubtreeSampler] = {}
        
        # 创建采样器
        for parent_level, child_level in LEVEL_PAIRS:
            parent_nodes = nodes_by_level.get(parent_level, [])
            child_nodes = nodes_by_level.get(child_level, [])
            
            if parent_nodes and child_nodes:
                sampler = SubtreeSampler(
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
                self.samplers[(parent_level, child_level)] = sampler
        
        self.available_level_pairs = list(self.samplers.keys())
        self._reset_level_pair_sampling_pool()
        
        # 根据开关决定是否加载 feats_by_level
        self.feats_by_level: Dict[str, torch.Tensor] = {}
        if load_feats_by_level:
            self._load_feats_by_level(nodes_by_level)

    def _reset_level_pair_sampling_pool(self) -> None:
        """重置层级对采样池，避免多层级训练时长期偏向某一层。"""
        self.level_pair_sampling_pool = self.available_level_pairs.copy()
        random.shuffle(self.level_pair_sampling_pool)
        self.level_pair_sampling_cursor = 0

    def _sample_level_pair_with_coverage(self) -> Tuple[str, str]:
        """按覆盖优先的方式轮转采样层级对。"""
        if not self.level_pair_sampling_pool:
            raise ValueError("当前没有可用的层级对采样器")

        if self.level_pair_sampling_cursor >= len(self.level_pair_sampling_pool):
            self._reset_level_pair_sampling_pool()

        chosen_pair = self.level_pair_sampling_pool[self.level_pair_sampling_cursor]
        self.level_pair_sampling_cursor += 1
        return chosen_pair
    
    def _load_feats_by_level(self, nodes_by_level: Dict[str, List[HierarchicalNode]]) -> None:
        """
        根据 level_pair 加载对应层级对的节点特征到 GPU。
        
        如果 level_pair 为 None，加载所有层级节点。
        """
        def get_node_embedding(node: HierarchicalNode) -> Optional[List[float]]:
            if self.use_level_embedding and node.level_embedding is not None:
                return node.level_embedding
            return node.embedding

        if self.level_pair:
            # 只加载当前层级对的节点
            parent_level, child_level = self.level_pair
            for level in [parent_level, child_level]:
                nodes = nodes_by_level.get(level, [])
                feats_list = [
                    torch.tensor(node_embedding, dtype=torch.float32)
                    for n in nodes
                    if (node_embedding := get_node_embedding(n)) is not None
                ]
                if feats_list:
                    self.feats_by_level[level] = torch.stack(feats_list).to(self.device)
        else:
            # level_pair 为 None，加载所有层级节点
            for level_name, nodes in nodes_by_level.items():
                feats_list = [
                    torch.tensor(node_embedding, dtype=torch.float32)
                    for n in nodes
                    if (node_embedding := get_node_embedding(n)) is not None
                ]
                if feats_list:
                    self.feats_by_level[level_name] = torch.stack(feats_list).to(self.device)
    
    def __len__(self) -> int:
        """返回数据集长度。"""
        return self.num_iterations
    
    def __getitem__(self, idx: int) -> SubtreeBatch:
        """获取第idx个采样的batch。"""
        if self.level_pair:
            sampler = self.samplers[self.level_pair]
        else:
            chosen_pair = self._sample_level_pair_with_coverage()
            sampler = self.samplers[chosen_pair]
        
        batch = sampler.sample_batch(dynamic_sampling=True)
        
        if batch is None:
            batch = SubtreeBatch(
                parent_level="EMPTY",
                child_level="EMPTY",
                parent_feats=torch.zeros(1, self.embedding_dim).to(self.device),
                child_feats=torch.zeros(1, self.embedding_dim).to(self.device),
                parent_child_mask=torch.zeros(1, 1).to(self.device),
                parent_child_map=torch.zeros(1, dtype=torch.long).to(self.device),
                parent_ids=["empty"],
                child_ids=["empty"],
                n_parent=1,
                n_child=1,
            )
        
        return batch
    
    def get_feats_by_level(self) -> Dict[str, torch.Tensor]:
        """获取层级对节点特征（仅在load_feats_by_level=True时有效）。"""
        return self.feats_by_level
    
    def get_sampler_stats(self) -> Dict:
        """获取各采样器的统计信息。"""
        stats = {}
        for pair, sampler in self.samplers.items():
            stats[f"{pair[0]}_{pair[1]}"] = {
                'total_valid_parents': sampler.total_valid_parents,
                'total_children': sampler.total_children,
                'num_batches': sampler.num_batches(),
            }
        return stats


# ============================================================================
# 自定义 collate 函数
# ============================================================================

def subtree_collate_fn(batch: List[SubtreeBatch]) -> SubtreeBatch:
    """
    自定义 collate 函数，用于处理 SubtreeBatch 类型。
    
    由于 SubtreeDataset.__getitem__ 已经返回完整的批次数据，
    DataLoader 的 batch_size 通常为 1，此函数直接返回单个 SubtreeBatch。
    
    参数:
        batch: SubtreeBatch 列表（长度通常为 1）
    
    返回:
        单个 SubtreeBatch 对象
    """
    if len(batch) == 1:
        return batch[0]
    
    # 如果 batch_size > 1，合并多个 SubtreeBatch
    first_batch = batch[0]
    
    parent_feats_list = [b.parent_feats for b in batch]
    child_feats_list = [b.child_feats for b in batch]
    parent_ids_combined = []
    child_ids_combined = []
    
    for b in batch:
        parent_ids_combined.extend(b.parent_ids)
        child_ids_combined.extend(b.child_ids)
    
    parent_feats = torch.cat(parent_feats_list, dim=0)
    child_feats = torch.cat(child_feats_list, dim=0)
    
    n_parent = sum(b.n_parent for b in batch)
    n_child = sum(b.n_child for b in batch)
    
    # 重建归属矩阵
    parent_child_mask = torch.zeros(n_parent, n_child, dtype=torch.float32)
    parent_child_map = torch.zeros(n_child, dtype=torch.long)
    
    parent_offset = 0
    child_offset = 0
    for b in batch:
        local_n_parent = b.n_parent
        local_n_child = b.n_child
        parent_child_mask[parent_offset:parent_offset+local_n_parent, 
                          child_offset:child_offset+local_n_child] = b.parent_child_mask
        parent_child_map[child_offset:child_offset+local_n_child] = b.parent_child_map + parent_offset
        parent_offset += local_n_parent
        child_offset += local_n_child
    
    return SubtreeBatch(
        parent_level=first_batch.parent_level,
        child_level=first_batch.child_level,
        parent_feats=parent_feats,
        child_feats=child_feats,
        parent_child_mask=parent_child_mask,
        parent_child_map=parent_child_map,
        parent_ids=parent_ids_combined,
        child_ids=child_ids_combined,
        n_parent=n_parent,
        n_child=n_child,
    )


# ============================================================================
# DataLoader 工厂函数
# ============================================================================

def create_subtree_dataloader(
    nodes_by_level: Dict[str, List[HierarchicalNode]],
    embedding_dim: int,
    batch_size: int = 1,
    device: torch.device = None,
    num_iterations: int = 1000,
    num_parents_per_batch: int = 16,
    num_children_per_parent: int = 4,
    max_children_per_parent: int = 10,
    level_pair: Tuple[str, str] = None,
    load_feats_by_level: bool = False,
    use_level_embedding: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    创建子树采样DataLoader。
    
    参数:
        nodes_by_level: 各层级节点字典
        embedding_dim: 嵌入维度
        device: 计算设备
        num_iterations: 总迭代次数
        num_parents_per_batch: 每采样batch的父节点数
        num_children_per_parent: 每父节点子节点数
        max_children_per_parent: 动态采样时的最大子节点数
        level_pair: 固定层级对
        load_feats_by_level: 是否加载层级对节点特征到GPU
        use_level_embedding: 是否优先使用带层级前缀的 embedding
        shuffle: 是否打乱
        num_workers: DataLoader worker数
    """
    dataset = SubtreeDataset(
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


# ============================================================================
# 从VectorStore提取节点的工具函数
# ============================================================================

def extract_nodes_from_store(
    vector_store,
    level_pair_index: Optional[int] = None
) -> Dict[str, List[HierarchicalNode]]:
    """
    从HierarchicalVectorStore提取层级节点。
    
    参数:
        vector_store: HierarchicalVectorStore 实例
        level_pair_index: 层级对索引，控制提取范围
            None: 提取全层节点（DOMAIN, CATEGORY, KEYWORD, DIALOGUE）
            1: 提取第一层级对节点（DOMAIN, CATEGORY）
            2: 提取第二层级对节点（CATEGORY, KEYWORD）
            3: 提取第三层级对节点（KEYWORD, DIALOGUE）
    
    返回:
        字典格式: {"DOMAIN": [...], "CATEGORY": [...], ...}
    """
    nodes_by_level = {}
    
    level_names = ["DOMAIN", "CATEGORY", "KEYWORD", "DIALOGUE"]
    
    if level_pair_index is None:
        # 提取全层节点
        target_levels = level_names
    elif level_pair_index == 1:
        # 第一层级对: DOMAIN → CATEGORY
        target_levels = ["DOMAIN", "CATEGORY"]
    elif level_pair_index == 2:
        # 第二层级对: CATEGORY → KEYWORD
        target_levels = ["CATEGORY", "KEYWORD"]
    elif level_pair_index == 3:
        # 第三层级对: KEYWORD → DIALOGUE
        target_levels = ["KEYWORD", "DIALOGUE"]
    else:
        raise ValueError(f"level_pair_index 必须是 None, 1, 2 或 3，当前值: {level_pair_index}")
    
    for level_name in target_levels:
        level_enum = HierarchyLevel[level_name]
        nodes = vector_store.get_nodes_by_level(level_enum)
        nodes_by_level[level_name] = nodes

    return nodes_by_level