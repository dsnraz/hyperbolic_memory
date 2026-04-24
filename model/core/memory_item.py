"""
记忆条目数据结构。

本模块定义记忆条目的基本数据结构。
不同类型的记忆条目对应不同的记忆系统。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .memory_types import MemoryType, MemoryPriority, MemoryState, MemoryMetadata


@dataclass
class MemoryItem(ABC):
    """
    所有记忆条目的抽象基类。
    
    记忆条目表示智能体记忆系统中的单个记忆单元。
    遵循认知启发的记忆架构，不同类型的记忆承担不同功能。
    
    属性:
        id: 记忆条目的唯一标识符
        content: 记忆的实际内容/表示
        memory_type: 记忆的类型分类
        metadata: 跟踪访问模式和生命周期的元数据
        embedding: 用于相似度搜索的向量嵌入
        state: 记忆生命周期中的当前状态
        priority: 检索决策的重要性级别
        tags: 用于分类的用户定义标签
        source: 记忆的来源（感知、推理等）
        associations: 相关记忆条目的ID列表
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    content: Any = None
    memory_type: MemoryType = MemoryType.SHORT_TERM
    metadata: MemoryMetadata = field(default_factory=lambda: MemoryMetadata(
        created_at=datetime.now(),
        last_accessed=datetime.now()
    ))
    embedding: Optional[List[float]] = None
    state: MemoryState = MemoryState.ENCODED
    priority: MemoryPriority = MemoryPriority.MEDIUM
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    associations: List[str] = field(default_factory=list)
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """将记忆条目序列化为字典。"""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """从字典反序列化记忆条目。"""
        pass
    
    def update_access(self) -> None:
        """更新访问元数据。"""
        self.metadata.update_access()
        self.state = MemoryState.RETRIEVED
    
    def add_association(self, memory_id: str) -> None:
        """添加到另一个记忆条目的关联。"""
        if memory_id not in self.associations:
            self.associations.append(memory_id)
    
    def remove_association(self, memory_id: str) -> None:
        """移除关联。"""
        if memory_id in self.associations:
            self.associations.remove(memory_id)
    
    def add_tag(self, tag: str) -> None:
        """为记忆添加标签。"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """移除标签。"""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def get_strength(self) -> float:
        """
        计算记忆强度。
        
        基于访问模式和时间衰减计算。
        返回 0.0（完全衰减）到 1.0（最大强度）之间的值。
        可重写此方法以实现自定义衰减函数。
        """
        base_strength = 1.0
        
        # 因素1：访问频率（对数归一化）
        access_factor = min(1.0, 0.1 * (1 + self.metadata.access_count))
        
        # 因素2：新近度（指数衰减）
        recency_factor = self.metadata.get_recency_score()
        
        # 因素3：巩固（每次巩固都会增强记忆）
        consolidation_factor = min(1.0, 0.1 * (1 + self.metadata.consolidation_count))
        
        # 因素4：时间衰减
        age_seconds = self.metadata.get_age_seconds()
        time_decay = max(0.0, 1.0 - (self.metadata.decay_rate * age_seconds / 3600.0))
        
        # 综合强度
        strength = (
            0.2 * base_strength +
            0.3 * access_factor +
            0.2 * recency_factor +
            0.15 * consolidation_factor +
            0.15 * time_decay
        )
        
        return max(0.0, min(1.0, strength))
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryItem):
            return False
        return self.id == other.id


@dataclass
class SensoryMemoryItem(MemoryItem):
    """
    感觉记忆条目，用于短暂的感知印象。
    
    感觉记忆以高容量但极短的持续时间存储信息。
    此类记忆通常用于初始感知处理，
    之后信息被转移到短期记忆。
    
    属性:
        attention_weight: 此感知输入的注意分配权重
        decay_timestamp: 感觉记忆应该衰减的时间
    """
    
    memory_type: MemoryType = MemoryType.SENSORY
    attention_weight: float = 0.5
    decay_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        # 感觉记忆衰减非常快（默认：1秒）
        if self.decay_timestamp is None:
            self.decay_timestamp = datetime.now()
    
    def is_decayed(self) -> bool:
        """检查感觉记忆是否已衰减。"""
        if self.decay_timestamp is None:
            return False
        return datetime.now() > self.decay_timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.name,
            "attention_weight": self.attention_weight,
            "decay_timestamp": self.decay_timestamp.isoformat() if self.decay_timestamp else None,
            "state": self.state.name,
            "priority": self.priority.name,
            "tags": self.tags,
            "source": self.source,
            "associations": self.associations,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "last_accessed": self.metadata.last_accessed.isoformat(),
                "access_count": self.metadata.access_count,
                "importance_score": self.metadata.importance_score,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensoryMemoryItem":
        metadata = MemoryMetadata(
            created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
            last_accessed=datetime.fromisoformat(data["metadata"]["last_accessed"]),
            access_count=data["metadata"].get("access_count", 0),
            importance_score=data["metadata"].get("importance_score", 0.0),
        )
        return cls(
            id=data["id"],
            content=data["content"],
            attention_weight=data.get("attention_weight", 0.5),
            decay_timestamp=datetime.fromisoformat(data["decay_timestamp"]) if data.get("decay_timestamp") else None,
            state=MemoryState[data.get("state", "ENCODED")],
            priority=MemoryPriority[data.get("priority", "MEDIUM")],
            tags=data.get("tags", []),
            source=data.get("source"),
            associations=data.get("associations", []),
            metadata=metadata,
        )


@dataclass
class EpisodicMemoryItem(MemoryItem):
    """
    情景记忆条目，用于个人经历和事件。
    
    情景记忆存储带有时间和空间上下文的自传体事件。
    包括经历的"什么"、"何时"和"何地"。
    
    属性:
        event: 主要事件描述
        context: 情境上下文（位置、环境等）
        timestamp: 事件发生时间
        end_timestamp: 事件结束时间（对于延续性事件）
        participants: 事件涉及的实体
        emotional_valence: 情感关联（积极/消极）
        outcome: 事件的结果或后果
    """
    
    memory_type: MemoryType = MemoryType.EPISODIC
    event: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    end_timestamp: Optional[datetime] = None
    participants: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0  # -1.0（消极）到 1.0（积极）
    outcome: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.name,
            "event": self.event,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat() if self.end_timestamp else None,
            "participants": self.participants,
            "emotional_valence": self.emotional_valence,
            "outcome": self.outcome,
            "embedding": self.embedding,
            "state": self.state.name,
            "priority": self.priority.name,
            "tags": self.tags,
            "source": self.source,
            "associations": self.associations,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "last_accessed": self.metadata.last_accessed.isoformat(),
                "access_count": self.metadata.access_count,
                "importance_score": self.metadata.importance_score,
                "consolidation_count": self.metadata.consolidation_count,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemoryItem":
        metadata = MemoryMetadata(
            created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
            last_accessed=datetime.fromisoformat(data["metadata"]["last_accessed"]),
            access_count=data["metadata"].get("access_count", 0),
            importance_score=data["metadata"].get("importance_score", 0.0),
            consolidation_count=data["metadata"].get("consolidation_count", 0),
        )
        return cls(
            id=data["id"],
            content=data["content"],
            event=data.get("event", ""),
            context=data.get("context", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            end_timestamp=datetime.fromisoformat(data["end_timestamp"]) if data.get("end_timestamp") else None,
            participants=data.get("participants", []),
            emotional_valence=data.get("emotional_valence", 0.0),
            outcome=data.get("outcome"),
            embedding=data.get("embedding"),
            state=MemoryState[data.get("state", "ENCODED")],
            priority=MemoryPriority[data.get("priority", "MEDIUM")],
            tags=data.get("tags", []),
            source=data.get("source"),
            associations=data.get("associations", []),
            metadata=metadata,
        )


@dataclass
class SemanticMemoryItem(MemoryItem):
    """
    语义记忆条目，用于事实和一般知识。
    
    语义记忆存储不与特定个人经历绑定的
    概念性知识、事实和一般信息。
    
    属性:
        concept: 此记忆关于的概念或实体
        facts: 关于该概念的事实或知识列表
        relationships: 与其他概念的关系
        confidence: 对此知识的置信度
        source_credibility: 来源的可信度
        last_verified: 此知识最后验证/更新的时间
    """
    
    memory_type: MemoryType = MemoryType.SEMANTIC
    concept: str = ""
    facts: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)  # 相关概念 -> 关系类型
    confidence: float = 1.0  # 0.0 到 1.0
    source_credibility: float = 1.0
    last_verified: datetime = field(default_factory=datetime.now)
    
    def add_fact(self, fact: str) -> None:
        """向此语义记忆添加事实。"""
        if fact not in self.facts:
            self.facts.append(fact)
    
    def remove_fact(self, fact: str) -> None:
        """从此语义记忆移除事实。"""
        if fact in self.facts:
            self.facts.remove(fact)
    
    def add_relationship(self, concept: str, relationship_type: str) -> None:
        """添加到另一个概念的关系。"""
        self.relationships[concept] = relationship_type
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.name,
            "concept": self.concept,
            "facts": self.facts,
            "relationships": self.relationships,
            "confidence": self.confidence,
            "source_credibility": self.source_credibility,
            "last_verified": self.last_verified.isoformat(),
            "embedding": self.embedding,
            "state": self.state.name,
            "priority": self.priority.name,
            "tags": self.tags,
            "source": self.source,
            "associations": self.associations,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "last_accessed": self.metadata.last_accessed.isoformat(),
                "access_count": self.metadata.access_count,
                "importance_score": self.metadata.importance_score,
                "consolidation_count": self.metadata.consolidation_count,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemoryItem":
        metadata = MemoryMetadata(
            created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
            last_accessed=datetime.fromisoformat(data["metadata"]["last_accessed"]),
            access_count=data["metadata"].get("access_count", 0),
            importance_score=data["metadata"].get("importance_score", 0.0),
            consolidation_count=data["metadata"].get("consolidation_count", 0),
        )
        return cls(
            id=data["id"],
            content=data["content"],
            concept=data.get("concept", ""),
            facts=data.get("facts", []),
            relationships=data.get("relationships", {}),
            confidence=data.get("confidence", 1.0),
            source_credibility=data.get("source_credibility", 1.0),
            last_verified=datetime.fromisoformat(data["last_verified"]) if data.get("last_verified") else datetime.now(),
            embedding=data.get("embedding"),
            state=MemoryState[data.get("state", "ENCODED")],
            priority=MemoryPriority[data.get("priority", "MEDIUM")],
            tags=data.get("tags", []),
            source=data.get("source"),
            associations=data.get("associations", []),
            metadata=metadata,
        )