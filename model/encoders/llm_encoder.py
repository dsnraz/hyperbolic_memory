"""
LLM 编码器模块。

使用大语言模型（LLM）对输入进行编码，
提取结构化信息（领域、类别、关键词等）。

模型加载和调用细节由 model_handler.py 处理。
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import json_repair          
import re

from ..core.base_encoder import BaseMemoryEncoder, EncodingResult
from ..core.memory_item import MemoryItem, EpisodicMemoryItem
from ..core.memory_types import MemoryType, MemoryPriority
from .model_handler import create_model_handler, BaseModelHandler


class LLMEncoder(BaseMemoryEncoder):
    """
    基于大语言模型的编码器。
    
    使用 LLM 分析输入内容，提取：
    - 领域（Domain）
    - 类别（Category）
    - 关键词（Keywords）
    - 摘要（Summary）
    
    支持多种模型部署方式和批量推理。
    """
    
    ANALYSIS_PROMPT = """
[INST]
You are a professional dialogue structured extractor.
You ONLY output STANDARD JSON.
You DO NOT generate any extra words, explanations, notes, translations, repetitions, or comments.
You DO NOT output any content outside of JSON.

Your output MUST strictly follow this format:
{{
    "domain": "macro domain",
    "category": "specific category",
    "keywords": ["key1", "key2", "key3", "key4", "key5"],
    "summary": "one-sentence summary"
}}

Rules:
1. Only extract facts from the dialogue.
2. keywords: must 3–5 key terms, ranked by Relevance with the category.
3. summary: one clear, complete, short sentence, which must be less than 100 words.
4. OUTPUT ONLY JSON. NO OTHER TEXT.
5. Escape any internal double quotes with a backslash (e.g., \"Title\").
[/INST]

Dialogue:
{dialogue}

JSON OUTPUT:
"""

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        model_path: Optional[str] = None,
        api_base: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        model_type: str = "ollama",
        device: str = "auto",
        **kwargs
    ):
        super().__init__(embedding_model=None, embedding_dim=768, **kwargs)

        self.model_name = model_name
        self.model_path = model_path
        self.api_base = api_base
        self.api_key = api_key
        self.model_type = model_type
        self.device = device

        self._handler: BaseModelHandler = None

    def _init_handler(self) -> bool:
        """初始化模型处理器。"""
        if self._handler is not None and self._handler.is_loaded():
            return True

        handler_kwargs = {"api_base": self.api_base}
        if self.api_key:
            handler_kwargs["api_key"] = self.api_key

        if self.model_type == "transformers":
            self._handler = create_model_handler("transformers", **handler_kwargs)
            model_source = self.model_path or self.model_name
            return self._handler.load(
                model_source,
                device=self.device,
                load_in_8bit=self.config.get("load_in_8bit"),
                load_in_4bit=self.config.get("load_in_4bit"),
            )
        elif self.model_type == "ollama":
            self._handler = create_model_handler("ollama", **handler_kwargs)
            return self._handler.load(self.model_name)
        else:
            self._handler = create_model_handler("openai", **handler_kwargs)
            return self._handler.load(self.model_name)
    
    def _format_input(self, input_data: Any) -> str:
        """格式化输入数据。"""
        if isinstance(input_data, dict):
            return json.dumps(input_data, ensure_ascii=False)
        return str(input_data)
    
    def _sanitize_json_response(self, response: str) -> str:
        """清理JSON响应中的无效转义字符。"""
        # 标准 JSON 转义字符: \" \\ \b \f \n \r \t \uXXXX
        # 其他如 \_ \: \# \- 等都是无效的，替换为普通字符
        
        def replace_invalid_escape(match):
            char = match.group(1)
            # 如果是标准转义字符，保留
            if char in '"\\bfnrt':
                return match.group(0)
            # 其他无效转义，将反斜杠去掉（即替换为普通字符）
            else:
                return char
        
        # 匹配 \ 后面跟着非标准转义字符
        sanitized = re.sub(r'\\([^"\\bfnrtu])', replace_invalid_escape, response)
        
        return sanitized
    
    def _parse_response(self, response: str, original_dialogue: str) -> Tuple[Dict[str, Any], bool]:
        """解析 LLM 响应，提取结构化信息。"""
        is_ok = True
        try:
            response = response.strip()
            
            # 清理无效转义字符
            sanitized_response = self._sanitize_json_response(response)
            
            if sanitized_response.startswith("{"):
                data = json_repair.loads(sanitized_response)
                return self._build_result(data, original_dialogue), is_ok
            
            json_match = re.search(r'\{[\s\S]*\}', sanitized_response)
            if json_match:
                data = json_repair.loads(json_match.group())
                return self._build_result(data, original_dialogue), is_ok
            
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            print(f"原始响应: {response}")
        
        return {
            "domain": "未知",
            "category": "未知",
            "keywords": [],
            "summary": "",
            "raw_dialogue": original_dialogue,
        }, False
    
    def _normalize_result_payload(self, data: Any) -> Dict[str, Any]:
        """将解析结果规范化为单个字典。"""
        if isinstance(data, dict):
            return data

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    return item
            return {}

        return {}

    def _build_result(self, data: Any, original_dialogue: str) -> Dict[str, Any]:
        """构建分析结果。"""
        normalized = self._normalize_result_payload(data)
        keywords = normalized.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = [keywords] if keywords else []

        return {
            "domain": normalized.get("domain", "未知"),
            "category": normalized.get("category", "未知"),
            "keywords": keywords,
            "summary": normalized.get("summary"),
            "raw_dialogue": original_dialogue,
        }
    
    def analyze(self, dialogue: str, **kwargs) -> Tuple[Dict[str, Any], bool]:
        """分析单条对话内容。"""
        if not self._init_handler():
            raise RuntimeError("模型初始化失败")
        
        dialogue = self._format_input(dialogue)
        prompt = self.ANALYSIS_PROMPT.format(dialogue=dialogue)
        
        response = self._handler.generate(prompt, **kwargs)
        result, is_ok = self._parse_response(response, dialogue)
        
        return result, is_ok
    
    def batch_analyze(
        self,
        dialogues: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[bool]]:
        """
        批量分析对话（使用批量推理提升 GPU 利用率）。
        
        参数:
            dialogues: 对话列表
            batch_size: 批量大小（根据 GPU 内存调整，建议 8-16）
            show_progress: 是否显示进度条
            **kwargs: 生成参数
            
        返回:
            (results, is_ok_list): 分析结果列表和成功标志列表
        """
        if not self._init_handler():
            raise RuntimeError("模型初始化失败")
        
        results = []
        is_ok_list = []
        
        # 格式化所有对话
        formatted_dialogues = [self._format_input(d) for d in dialogues]
        
        # 分批处理
        total_batches = (len(dialogues) + batch_size - 1) // batch_size
        
        batch_iter = range(total_batches)
        if show_progress:
            try:
                from tqdm import tqdm
                batch_iter = tqdm(batch_iter, desc="批量分析", unit="batch")
            except ImportError:
                pass
        
        for batch_idx in batch_iter:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dialogues))
            batch_dialogues = formatted_dialogues[start_idx:end_idx]
            
            # 构建批量 prompts
            prompts = [self.ANALYSIS_PROMPT.format(dialogue=d) for d in batch_dialogues]
            
            # 批量推理
            responses = self._handler.batch_generate(prompts, **kwargs)
            # 解析批量响应
            for i, (response, original) in enumerate(zip(responses, batch_dialogues)):
                result, is_ok = self._parse_response(response, original)
                results.append(result)
                is_ok_list.append(is_ok)
                
                if not is_ok:
                    print(f"[批次 {batch_idx}] 样本 {start_idx + i} 解析失败")
        
        return results, is_ok_list
    
    def encode(self, input_data: Any, **kwargs) -> EncodingResult:
        """使用 LLM 编码输入数据。"""
        import time
        start_time = time.time()
        
        result, is_ok = self.analyze(input_data, **kwargs)
        
        dialogue = self._format_input(input_data)
        
        memory_item = EpisodicMemoryItem(
            content=dialogue,
            event=result.get("summary", dialogue[:100]),
            memory_type=self.get_target_memory_type(input_data),
            priority=self.estimate_importance(input_data),
        )
        memory_item.metadata.analysis_result = result
        
        encoding_time = (time.time() - start_time) * 1000
        
        return EncodingResult(
            memory_item=memory_item,
            encoding_quality=0.9 if is_ok else 0.5,
            attention_score=0.8,
            encoding_time_ms=encoding_time,
        )
    
    def get_target_memory_type(self, input_data: Any) -> MemoryType:
        """确定输入应存储为哪种记忆类型。"""
        return MemoryType.EPISODIC
    
    def estimate_importance(self, input_data: Any) -> MemoryPriority:
        """估计输入的重要性。"""
        return MemoryPriority.MEDIUM


DialogueAnalyzer = LLMEncoder