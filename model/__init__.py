"""
智能体记忆系统。

提供分层记忆架构，用于组织和检索对话数据。

模块结构：
- core: 核心抽象基类和类型定义
- encoders: 编码器实现（LLM编码器、嵌入编码器）
- stores: 存储实现（内存存储、向量存储）
- retrievers: 检索器实现
- hierarchical: 分层记忆协调器
- hyperbolic_utils: 双曲空间数学工具

快速开始:
    from memory import create_hierarchical_manager
    
    manager = create_hierarchical_manager(
        llm_model_path="/path/to/model",
        persist_directory="./data/memory"
    )
    
    # 处理对话
    manager.process_dialogue("今天讨论了关于机器学习的内容...")
    
    # 搜索
    results = manager.search("机器学习")
"""

# 核心类型
from .core import (
    MemoryType,
    MemoryPriority,
    MemoryState,
    MemoryMetadata,
    MemoryItem,
    SensoryMemoryItem,
    EpisodicMemoryItem,
    SemanticMemoryItem,
    BaseMemoryEncoder,
    EncodingResult,
    BaseMemoryStore,
    BaseMemoryRetriever,
    RetrievalQuery,
    RetrievalResult,
)

# 编码器
from .encoders import (
    LLMEncoder,
    DialogueAnalyzer,
    EmbeddingEncoder,
)

# 存储
from .stores import (
    InMemoryStore,
    SerializableInMemoryStore,
    HierarchicalVectorStore,
    VectorStoreFactory,
)

# 分层记忆
from .hierarchical import (
    HierarchyLevel,
    HierarchicalNode,
    DialogueAnalysisResult,
    HierarchicalMemoryStats,
    HierarchicalMemoryManager,
    create_hierarchical_manager,
)

# 测试时记忆增强推理
from .llm_inference import (
    normalize_interaction,
    extract_interactions,
    ConversationMemoryBuildResult,
    ConversationMemoryBuilder,
    MemoryAugmentedLLMInference,
)

__all__ = [
    # 核心类型
    "MemoryType",
    "MemoryPriority",
    "MemoryState",
    "MemoryMetadata",
    "MemoryItem",
    "SensoryMemoryItem",
    "EpisodicMemoryItem",
    "SemanticMemoryItem",
    # 编码器
    "BaseMemoryEncoder",
    "EncodingResult",
    "LLMEncoder",
    "DialogueAnalyzer",
    "EmbeddingEncoder",
    # 存储
    "BaseMemoryStore",
    "InMemoryStore",
    "SerializableInMemoryStore",
    "HierarchicalVectorStore",
    "VectorStoreFactory",
    # 检索器
    "BaseMemoryRetriever",
    "RetrievalQuery",
    "RetrievalResult",
    # 分层记忆
    "HierarchyLevel",
    "HierarchicalNode",
    "DialogueAnalysisResult",
    "HierarchicalMemoryStats",
    "HierarchicalMemoryManager",
    "create_hierarchical_manager",
    # 记忆增强推理
    "normalize_interaction",
    "extract_interactions",
    "ConversationMemoryBuildResult",
    "ConversationMemoryBuilder",
    "MemoryAugmentedLLMInference",
]