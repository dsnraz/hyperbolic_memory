"""
嵌入编码器模块。

使用预训练模型生成文本嵌入向量。
"""

from typing import Any, Dict, List, Optional

from ..core.base_encoder import BaseMemoryEncoder, EncodingResult
from ..core.memory_item import MemoryItem, EpisodicMemoryItem
from ..core.memory_types import MemoryType, MemoryPriority


# 默认本地模型路径
DEFAULT_LOCAL_MODEL_PATH = "/share/home/leiyh5/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130"


class EmbeddingEncoder(BaseMemoryEncoder):
    """
    嵌入向量编码器。
    
    使用 sentence-transformers 或其他预训练模型
    生成文本的向量表示。
    
    支持从本地路径加载模型（适用于无法访问 HuggingFace 的环境）。
    """
    
    def __init__(
        self,
        model_name: str = None,
        model_path: str = None,
        device: str = "auto",
        normalize: bool = True,
        local_files_only: bool = True,
        **kwargs
    ):
        """
        初始化嵌入编码器。
        
        参数:
            model_name: 预训练模型名称（如 "sentence-transformers/all-MiniLM-L6-v2"）
            model_path: 本地模型路径（优先于 model_name）
            device: 设备选择（cuda/cpu/auto）
            normalize: 是否对嵌入向量进行 L2 归一化
            local_files_only: 是否只使用本地文件（不尝试从网络下载）
            **kwargs: 额外配置
        """
        super().__init__(embedding_model=None, embedding_dim=384, **kwargs)
        
        # 仅当显式传入时才使用本地路径；否则按模型名称加载。
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.local_files_only = local_files_only
        self._model = None
    
    def _init_model(self) -> bool:
        """初始化嵌入模型。"""
        if self._model is not None:
            return True
        
        try:
            from sentence_transformers import SentenceTransformer
            
            device = None if self.device == "auto" else self.device
            
            model_source = self.model_path or self.model_name or DEFAULT_LOCAL_MODEL_PATH
            source_kind = "model_path" if self.model_path else ("model_name" if self.model_name else "default_local")
            
            print(f"正在加载嵌入模型({source_kind}): {model_source}")
            
            self._model = SentenceTransformer(
                model_source,
                device=device,
                local_files_only=self.local_files_only,
            )
            self.embedding_dim = self._model.get_sentence_embedding_dimension()
            
            print(f"嵌入模型加载完成，维度: {self.embedding_dim}")
            return True
            
        except ImportError:
            print("请安装 sentence-transformers: pip install sentence-transformers")
            return False
        except Exception as e:
            print(f"嵌入模型加载失败: {e}")
            # 如果本地加载失败，尝试不使用 local_files_only
            try:
                print("尝试从网络加载...")
                self._model = SentenceTransformer(
                    self.model_name or "sentence-transformers/all-MiniLM-L6-v2",
                    device=device,
                )
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
                return True
            except Exception as e2:
                print(f"从网络加载也失败: {e2}")
                return False
    
    def encode(self, input_data: Any, **kwargs) -> EncodingResult:
        """
        编码输入数据并生成嵌入。
        
        参数:
            input_data: 输入文本
            **kwargs: 额外参数
            
        返回:
            EncodingResult 编码结果
        """
        import time
        start_time = time.time()
        
        if not self._init_model():
            raise RuntimeError("嵌入模型初始化失败")
        
        # 转换输入为字符串
        if isinstance(input_data, dict):
            text = str(input_data)
        else:
            text = str(input_data)
        
        # 生成嵌入
        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_tensor=False,
        ).tolist()
        
        # 创建记忆条目
        memory_item = EpisodicMemoryItem(
            content=text,
            event=text[:200],
            memory_type=self.get_target_memory_type(input_data),
            priority=self.estimate_importance(input_data),
            embedding=embedding,
        )
        
        encoding_time = (time.time() - start_time) * 1000
        
        return EncodingResult(
            memory_item=memory_item,
            encoding_quality=1.0,
            attention_score=1.0,
            encoding_time_ms=encoding_time,
            raw_embedding=embedding,
        )
    
    def encode_batch(self, inputs: List[Any], **kwargs) -> List[EncodingResult]:
        """
        批量编码输入数据。
        
        参数:
            inputs: 输入列表
            **kwargs: 额外参数
            
        返回:
            EncodingResult 列表
        """
        if not self._init_model():
            raise RuntimeError("嵌入模型初始化失败")
        
        # 转换输入为字符串列表
        texts = [str(inp) if not isinstance(inp, dict) else str(inp) for inp in inputs]
        
        # 批量生成嵌入
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_tensor=False,
        )
        
        # 创建结果列表
        results = []
        for text, embedding in zip(texts, embeddings):
            memory_item = EpisodicMemoryItem(
                content=text,
                event=text[:200],
                memory_type=MemoryType.EPISODIC,
                priority=MemoryPriority.MEDIUM,
                embedding=embedding.tolist(),
            )
            
            results.append(EncodingResult(
                memory_item=memory_item,
                encoding_quality=1.0,
                attention_score=1.0,
                encoding_time_ms=0.0,
                raw_embedding=embedding.tolist(),
            ))
        
        return results
    
    def get_target_memory_type(self, input_data: Any) -> MemoryType:
        """确定输入应存储为哪种记忆类型。"""
        return MemoryType.EPISODIC
    
    def estimate_importance(self, input_data: Any) -> MemoryPriority:
        """估计输入的重要性。"""
        return MemoryPriority.MEDIUM
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        生成单个文本的嵌入向量。
        
        参数:
            text: 输入文本
            
        返回:
            嵌入向量
        """
        if not self._init_model():
            raise RuntimeError("嵌入模型初始化失败")
        
        return self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_tensor=False,
        ).tolist()
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成嵌入向量。
        
        参数:
            texts: 文本列表
            
        返回:
            嵌入向量列表
        """
        if not self._init_model():
            raise RuntimeError("嵌入模型初始化失败")
        
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_tensor=False,
        )
        
        return [emb.tolist() for emb in embeddings]