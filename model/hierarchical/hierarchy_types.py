"""
分层记忆类型定义。

定义分层记忆架构的层级类型和节点结构。
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import uuid4


class HierarchyLevel(Enum):
    """
    分层记忆的层级。
    
    从上到下依次为：
    - DOMAIN: 领域层（宏观领域，如电影、科技、体育）
    - CATEGORY: 细化类别（子类，如动作片、科幻片）
    - KEYWORD: 关键词层（具体关键词，如成龙、武打）
    - DIALOGUE: 原始对话层（最底层，存储原始对话文本）
    """
    
    DOMAIN = auto()      # 领域层
    CATEGORY = auto()    # 细化类别层
    KEYWORD = auto()     # 关键词层
    DIALOGUE = auto()    # 原始对话层
    
    def get_parent_level(self) -> Optional["HierarchyLevel"]:
        """获取父层级。"""
        hierarchy_order = [self.DOMAIN, self.CATEGORY, self.KEYWORD, self.DIALOGUE]
        idx = hierarchy_order.index(self)
        if idx == 0:
            return None  # 领域层没有父级
        return hierarchy_order[idx - 1]
    
    def get_child_level(self) -> Optional["HierarchyLevel"]:
        """获取子层级。"""
        hierarchy_order = [self.DOMAIN, self.CATEGORY, self.KEYWORD, self.DIALOGUE]
        idx = hierarchy_order.index(self)
        if idx == len(hierarchy_order) - 1:
            return None  # 对话层没有子级
        return hierarchy_order[idx + 1]


@dataclass
class HierarchicalNode:
    """
    分层记忆节点。
    
    每个节点代表层级结构中的一个元素，包含：
    - 自身内容
    - 向量嵌入
    - 父子节点索引
    - 元数据
    """
    
    # 基本信息
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""                          # 节点内容
    level: HierarchyLevel = HierarchyLevel.DIALOGUE
    
    # 向量嵌入
    embedding: Optional[List[float]] = None                    # 原始内容 embedding
    level_embedding: Optional[List[float]] = None              # 带层级前缀的 embedding
    
    # 层级关系索引
    parent_ids: List[str] = field(default_factory=list)  # 父节点 ID 列表
    child_ids: List[str] = field(default_factory=list)  # 子节点 ID 列表
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_child(self, child_id: str) -> None:
        """添加子节点。"""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
    
    def remove_child(self, child_id: str) -> None:
        """移除子节点。"""
        if child_id in self.child_ids:
            self.child_ids.remove(child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "id": self.id,
            "content": self.content,
            "level": self.level.name,
            "embedding": self.embedding,
            "level_embedding": self.level_embedding,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalNode":
        """从字典反序列化。"""
        return cls(
            id=data["id"],
            content=data["content"],
            level=HierarchyLevel[data["level"]],
            embedding=data.get("embedding"),
            level_embedding=data.get("level_embedding"),
            parent_ids=data.get("parent_ids", []),
            child_ids=data.get("child_ids", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


@dataclass
class DialogueAnalysisResult:
    """
    对话分析结果。
    
    LLM 分析对话后返回的结构化结果。
    """
    
    domain: str                                    # 领域
    category: str                                  # 细化类别
    keywords: List[str]                            # 关键词列表
    raw_dialogue: str                              # 原始对话
    summary: Optional[str] = None                  # 对话摘要（可选）
    entities: Optional[List[str]] = None           # 实体列表（可选）
    confidence: float = 1.0                        # 置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "domain": self.domain,
            "category": self.category,
            "keywords": self.keywords,
            "raw_dialogue": self.raw_dialogue,
            "summary": self.summary,
            "entities": self.entities,
        }


@dataclass  
class HierarchicalMemoryStats:
    """分层记忆统计信息。"""
    
    total_nodes: int = 0
    domain_count: int = 0
    category_count: int = 0
    keyword_count: int = 0
    dialogue_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "domain_count": self.domain_count,
            "category_count": self.category_count,
            "keyword_count": self.keyword_count,
            "dialogue_count": self.dialogue_count,
        }