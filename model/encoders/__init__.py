"""
编码器模块。

提供多种编码器实现：
- LLMEncoder: 基于大语言模型的编码器
- EmbeddingEncoder: 嵌入向量编码器
- model_handler: 模型加载和调用处理器
"""

from .llm_encoder import LLMEncoder, DialogueAnalyzer
from .embedding_encoder import EmbeddingEncoder
from .model_handler import (
    BaseModelHandler,
    TransformersModelHandler,
    OllamaModelHandler,
    OpenAICompatibleHandler,
    create_model_handler,
)

__all__ = [
    # LLM 编码器
    "LLMEncoder",
    "DialogueAnalyzer",  # 兼容旧命名
    # 嵌入编码器
    "EmbeddingEncoder",
    # 模型处理器
    "BaseModelHandler",
    "TransformersModelHandler",
    "OllamaModelHandler",
    "OpenAICompatibleHandler",
    "create_model_handler",
]