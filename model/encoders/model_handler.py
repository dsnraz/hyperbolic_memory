"""
模型处理器模块。

专门处理不同 LLM 模型的加载和调用细节，
支持多种模型类型（Qwen、Llama、Vicuna、ChatGLM 等）。
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class BaseModelHandler:
    """模型处理器基类。"""
    
    def load(self, model_path: str, device: str = "auto", **kwargs) -> bool:
        """加载模型。"""
        raise NotImplementedError
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成响应。"""
        raise NotImplementedError
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成响应。"""
        raise NotImplementedError
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        raise NotImplementedError


class TransformersModelHandler(BaseModelHandler):
    """
    Transformers 本地模型处理器。
    
    支持 HuggingFace transformers 库加载的模型，
    自动检测模型类型并进行相应的配置。
    """
    
    MODEL_TYPE_SIGNATURES = {
        "qwen": ["qwen"],
        "llama": ["llama", "vicuna", "alpaca"],
        "chatglm": ["chatglm"],
        "baichuan": ["baichuan"],
        "internlm": ["internlm"],
        "mistral": ["mistral"],
    }
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_type = "default"
        self._device = "auto"
    
    def detect_model_type(self, model_path: str) -> str:
        """根据路径检测模型类型。"""
        model_path_lower = model_path.lower()
        for model_type, signatures in self.MODEL_TYPE_SIGNATURES.items():
            if any(sig in model_path_lower for sig in signatures):
                return model_type
        return "default"
    
    def load(self, model_path: str, device: str = "auto", **kwargs) -> bool:
        """加载 transformers 模型。"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self._model_type = self.detect_model_type(model_path)
            self._device = device
            
            print(f"正在加载模型: {model_path}")
            print(f"检测到模型类型: {self._model_type}")
            
            device_map = self._get_device_map(device)
            self._tokenizer = self._load_tokenizer(model_path)
            self._model = self._load_model(model_path, device_map, kwargs)
            
            self._model.eval()
            print("模型加载完成")
            
            return True
            
        except ImportError as e:
            print(f"缺少依赖: {e}")
            print("请安装: pip install torch transformers accelerate")
            return False
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_device_map(self, device: str) -> Dict:
        """获取设备映射配置。"""
        if device == "auto":
            return "auto"
        elif device == "cuda":
            return "cuda"
        else:
            return {"": "cpu"}
    
    def _load_tokenizer(self, model_path: str) -> Any:
        """加载 tokenizer，根据模型类型进行特殊配置。"""
        from transformers import AutoTokenizer
        
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": False,
        }
        # Decoder-only 模型都需要 left padding 才能正确进行批量推理
        decoder_only_types = ["qwen", "llama", "mistral", "baichuan", "internlm"]
        
        if self._model_type in decoder_only_types:
            tokenizer_kwargs["padding_side"] = "left"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<|extra_0|>"
        
        return tokenizer
    
    def _load_model(self, model_path: str, device_map: Any, kwargs: Dict) -> Any:
        """加载模型，处理量化等配置。"""
        import torch
        from transformers import AutoModelForCausalLM
        
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
        }
        
        if kwargs.get("load_in_8bit"):
            load_kwargs["load_in_8bit"] = True
        elif kwargs.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16
        
        return AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    def build_prompt(self, user_prompt: str) -> str:
        """根据模型类型构建格式化的 prompt。"""
        if self._model_type == "qwen":
            messages = [{"role": "user", "content": user_prompt}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif self._model_type == "llama":
            return (
                "A chat between a curious user and an artificial intelligence "
                "assistant. The assistant gives helpful, detailed, and polite "
                "answers to the user's questions.\n\n"
                f"USER: {user_prompt}\n\nASSISTANT:"
            )
        elif self._model_type == "chatglm":
            return f"[Round 0]\n问：{user_prompt}\n答："
        else:
            return user_prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """生成模型响应。"""
        import torch
        
        if not self.is_loaded():
            raise RuntimeError("模型未加载")
        
        formatted_prompt = self.build_prompt(prompt)
        
        inputs = self._tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=kwargs.get("max_length", 10000000)
        )
        
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # temperature=temperature,
                do_sample=False,
                # top_p=top_p,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        input_token_length = inputs["input_ids"].shape[1]
        # print("输入长度")
        # print(input_token_length)
        generated_part = outputs[0][input_token_length:]
        response = self._tokenizer.decode(generated_part, skip_special_tokens=True).strip()
        
        return self._clean_response(response)
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        """
        批量生成模型响应。
        
        批量推理可以大幅提升 GPU 利用率和吞吐量（3-5倍）。
        """
        import torch
        
        if not self.is_loaded():
            raise RuntimeError("模型未加载")
        
        if len(prompts) == 0:
            return []
        
        if len(prompts) == 1:
            return [self.generate(prompts[0], max_new_tokens, temperature, top_p, **kwargs)]
        
        formatted_prompts = [self.build_prompt(p) for p in prompts]
        
        inputs = self._tokenizer(
            formatted_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=kwargs.get("max_length", 10000000000),
            padding=True,
        )
        
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # temperature=temperature,
                do_sample=False,
                # top_p=top_p,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        input_lengths = inputs["input_ids"].shape[1]
        # print("输入长度")
        # print(input_lengths)
        responses = []
        for output in outputs:
            generated_part = output[input_lengths:]
            response = self._tokenizer.decode(generated_part, skip_special_tokens=True).strip()
            responses.append(self._clean_response(response))
        
        return responses
    
    def _clean_response(self, response: str) -> str:
        """清理响应中的特殊标记。"""
        if self._model_type == "qwen" and "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        return response.strip()
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self._model is not None and self._tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息。"""
        return {
            "model_type": self._model_type,
            "device": self._device,
            "is_loaded": self.is_loaded(),
        }


class OllamaModelHandler(BaseModelHandler):
    """Ollama 模型处理器。"""
    
    def __init__(self, api_base: str = "http://localhost:11434"):
        self._api_base = api_base
        self._client = None
        self._model_name = None
    
    def load(self, model_name: str, device: str = "auto", **kwargs) -> bool:
        try:
            import ollama
            self._client = ollama.Client(host=self._api_base)
            self._model_name = model_name
            return True
        except ImportError:
            print("请安装 ollama: pip install ollama")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded():
            raise RuntimeError("Ollama 未连接")
        
        response = self._client.chat(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": kwargs.get("temperature", 0.1)}
        )
        return response["message"]["content"]
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        # Ollama 不支持批量，逐条处理
        return [self.generate(p, **kwargs) for p in prompts]
    
    def is_loaded(self) -> bool:
        return self._client is not None


class OpenAICompatibleHandler(BaseModelHandler):
    """OpenAI 兼容 API 处理器。"""
    
    def __init__(self, api_base: str = "http://localhost:8000/v1"):
        self._api_base = api_base
        self._client = None
        self._model_name = None
    
    def load(self, model_name: str, device: str = "auto", **kwargs) -> bool:
        try:
            import openai
            self._client = openai.OpenAI(
                base_url=self._api_base,
                api_key=kwargs.get("api_key", "dummy")
            )
            self._model_name = model_name
            return True
        except ImportError:
            print("请安装 openai: pip install openai")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded():
            raise RuntimeError("API 客户端未初始化")
        
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
        )
        return response.choices[0].message.content
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        # OpenAI API 不支持批量，逐条处理
        return [self.generate(p, **kwargs) for p in prompts]
    
    def is_loaded(self) -> bool:
        return self._client is not None


def create_model_handler(handler_type: str = "transformers", **kwargs) -> BaseModelHandler:
    """创建模型处理器。"""
    if handler_type == "transformers":
        return TransformersModelHandler()
    elif handler_type == "ollama":
        return OllamaModelHandler(api_base=kwargs.get("api_base", "http://localhost:11434"))
    elif handler_type == "openai":
        return OpenAICompatibleHandler(api_base=kwargs.get("api_base", "http://localhost:8000/v1"))
    else:
        raise ValueError(f"未知的处理器类型: {handler_type}")