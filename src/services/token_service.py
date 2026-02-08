# Token计数服务

from typing import Dict, Any, List
from dataclasses import dataclass

from src.core.interfaces import ITokenCounter, Message


@dataclass
class TokenConfig:
    """Token配置"""
    default_encoding: str = "cl100k_base"  # GPT-4 encoding


class TokenCounter(ITokenCounter):
    """Token计数服务"""
    
    def __init__(self, config: Optional[TokenConfig] = None):
        self.config = config or TokenConfig()
        self._encoding = None
    
    def _get_encoding(self):
        """获取编码器"""
        if self._encoding is None:
            try:
                import tiktoken
                self._encoding = tiktoken.get_encoding(self.config.default_encoding)
            except ImportError:
                self._encoding = None
        return self._encoding
    
    def count(self, text: str) -> int:
        """计算文本Token数"""
        encoding = self._get_encoding()
        
        if encoding:
            return len(encoding.encode(text))
        
        # 估算: 平均1 token ≈ 4字符
        return max(1, len(text) // 4)
    
    def count_messages(self, messages: List[Message]) -> int:
        """计算消息列表Token数"""
        encoding = self._get_encoding()
        
        total = 0
        
        for msg in messages:
            # 格式化消息内容
            content = f"{msg.role}: {msg.content}"
            
            if encoding:
                total += len(encoding.encode(content))
            else:
                total += self.count(content)
            
            # 添加每条消息的固定开销
            total += 4  # role标记 + 内容分隔符
        
        return total
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> float:
        """估算成本（基于常见定价）"""
        pricing = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},
            "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
            "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
            "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
            "gemini-pro": {"prompt": 0.0005, "completion": 0.0015},
            "deepseek-chat": {"prompt": 0.00014, "completion": 0.00028},
        }
        
        # 获取模型定价
        model_pricing = pricing.get(model.lower(), pricing["gpt-3.5-turbo"])
        
        prompt_cost = prompt_tokens / 1000 * model_pricing["prompt"]
        completion_cost = completion_tokens / 1000 * model_pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """截断文本以符合Token限制"""
        encoding = self._get_encoding()
        
        if encoding:
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)
        
        # 估算截断
        max_chars = max_tokens * 4
        return text[:max_chars]
    
    def count_batch(self, texts: List[str]) -> List[int]:
        """批量计算Token数"""
        return [self.count(text) for text in texts]


# 定价常量
MODEL_PRICING = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    "claude-3-opus-20240229": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet-20240229": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku-20240307": {"prompt": 0.00025, "completion": 0.00125},
    "gemini-1.5-pro": {"prompt": 0.00035, "completion": 0.00105},
    "gemini-1.5-flash": {"prompt": 0.000035, "completion": 0.000105},
    "deepseek-chat": {"prompt": 0.00014, "completion": 0.00028},
    "glm-4-plus": {"prompt": 0.001, "completion": 0.001},
}


from typing import Optional
