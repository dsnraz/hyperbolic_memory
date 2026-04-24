"""
分层记忆模块。

提供分层记忆架构，用于组织和检索对话数据。

层级结构：领域 -> 类别 -> 关键词 -> 对话

使用方式:
    from model.hierarchical import DataProcessor
    
    processor = DataProcessor(
        llm_model_path="/path/to/model",
        persist_directory="./data/memory"
    )
    
    # 处理文件
    processor.process_file("data.json")
"""

from .hierarchy_types import (
    HierarchyLevel,
    HierarchicalNode,
    DialogueAnalysisResult,
    HierarchicalMemoryStats,
)
from .hierarchical_manager import (
    HierarchicalMemoryManager,
    create_hierarchical_manager,
)
from .data_process import DataProcessor

# 注意：HierarchicalVectorStore 和 VectorStoreFactory 请从 model.stores 导入
# from model.stores import HierarchicalVectorStore, VectorStoreFactory

# 向后兼容：从 encoders 导入（不涉及循环）
from ..encoders import LLMEncoder, DialogueAnalyzer

__all__ = [
    # 类型定义
    "HierarchyLevel",
    "HierarchicalNode",
    "DialogueAnalysisResult",
    "HierarchicalMemoryStats",
    # 协调器
    "HierarchicalMemoryManager",
    "create_hierarchical_manager",
    # 数据处理器
    "DataProcessor",
    # 向后兼容（仅 encoders）
    "LLMEncoder",
    "DialogueAnalyzer",
]