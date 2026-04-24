"""
记忆系统核心模块。

提供记忆系统的抽象基类和基础类型定义。
"""

from .memory_types import (
    MemoryType,
    MemoryPriority,
    MemoryState,
    MemoryMetadata,
)
from .memory_item import (
    MemoryItem,
    SensoryMemoryItem,
    EpisodicMemoryItem,
    SemanticMemoryItem,
)
from .base_encoder import (
    BaseMemoryEncoder,
    EncodingResult,
    TextEncoder,
)
from .base_store import (
    BaseMemoryStore,
    StorageFullError,
    InvalidMemoryItemError,
)
from .base_retriever import (
    BaseMemoryRetriever,
    RetrievalQuery,
    RetrievalResult,
    SemanticRetriever,
    HybridRetriever,
)

__all__ = [
    # 记忆类型
    "MemoryType",
    "MemoryPriority",
    "MemoryState",
    "MemoryMetadata",
    # 记忆条目
    "MemoryItem",
    "SensoryMemoryItem",
    "EpisodicMemoryItem",
    "SemanticMemoryItem",
    # 编码器
    "BaseMemoryEncoder",
    "EncodingResult",
    "TextEncoder",
    # 存储
    "BaseMemoryStore",
    "StorageFullError",
    "InvalidMemoryItemError",
    # 检索器
    "BaseMemoryRetriever",
    "RetrievalQuery",
    "RetrievalResult",
    "SemanticRetriever",
    "HybridRetriever",
]