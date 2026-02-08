# OpenAI供应商实现
# 展示如何实现一个完整的LLM供应商

import asyncio
from typing import AsyncIterator, Dict, Any, Optional, List

from src.core.interfaces import AIRequest, AIResponse, UsageInfo, Message
from src.providers.llm.factory import BaseLLMProvider, register_provider


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """OpenAI供应商实现"""
    
    # 兼容的API列表
    COMPATIBLE_APIS = [
        "openai",
        "azure", 
        "localai",
        "ollama",
        "together",
        "groq",
    ]
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self._async_client = None
    
    async def _get_client(self):
        """获取同步客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                )
            except ImportError:
                self._logger.warning("OpenAI package not installed")
        return self._client
    
    async def _get_async_client(self):
        """获取异步客户端"""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                )
            except ImportError:
                self._logger.warning("OpenAI package not installed")
        return self._async_client
    
    async def generate(self, request: AIRequest) -> AIResponse:
        """生成回复"""
        client = await self._get_async_client()
        if client is None:
            raise ImportError("OpenAI package not installed")
        
        # 转换消息格式
        messages = self._convert_messages(request.messages)
        
        # 构建请求参数
        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        
        if request.json_mode:
            params["response_format"] = {"type": "json_object"}
        
        if request.tools:
            params["tools"] = request.tools
            params["tool_choice"] = "auto"
        
        try:
            response = await client.chat.completions.create(**params)
            
            return AIResponse(
                content=response.choices[0].message.content or "",
                usage=UsageInfo(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in response.choices[0].message.tool_calls or []
                ],
            )
        except Exception as e:
            self._logger.error(f"OpenAI API error: {e}")
            raise
    
    async def stream_generate(self, request: AIRequest) -> AsyncIterator[str]:
        """流式生成"""
        client = await self._get_async_client()
        if client is None:
            raise ImportError("OpenAI package not installed")
        
        messages = self._convert_messages(request.messages)
        
        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        
        if request.json_mode:
            params["response_format"] = {"type": "json_object"}
        
        try:
            stream = await client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self._logger.error(f"OpenAI streaming error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """计算Token数 - 使用tiktoken"""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len.encode(text).n_tokens
        except ImportError:
            # 估算: 平均1个Token约4个字符
            return len(text) // 4
        except Exception:
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Message]) -> int:
        """计算消息列表的Token数"""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            total = 0
            
            for msg in messages:
                # 加上角色和内容格式化的开销
                content = f"{msg.role}: {msg.content}"
                total += len.encode(content).n_tokens
                total += 4  # 每条消息的固定开销
            
            return total
        except ImportError:
            return sum(self.count_tokens(msg.content) for msg in messages)
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """转换消息格式"""
        result = []
        for msg in messages:
            msg_dict = {
                "role": msg.role.value if hasattr(msg.role, 'value') else msg.role,
                "content": msg.content,
            }
            
            if msg.name:
                msg_dict["name"] = msg.name
            
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            
            result.append(msg_dict)
        
        return result
    
    def get_provider_name(self) -> str:
        """获取供应商名称"""
        return "openai"
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return bool(self.api_key)


class OpenAICompatibleProvider(OpenAIProvider):
    """OpenAI兼容接口供应商 - 用于其他使用OpenAI API格式的供应商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.custom_base_url = config.get("base_url", "")
        self.custom_model = config.get("model", "")
    
    async def generate(self, request: AIRequest) -> AIResponse:
        """生成回复 - 使用自定义Base URL"""
        # 如果请求没有指定模型，使用默认模型
        if not request.model and self.custom_model:
            request.model = self.custom_model
        
        return await super().generate(request)
    
    def get_provider_name(self) -> str:
        """获取供应商名称"""
        return f"openai-compatible ({self.custom_base_url})"


# 注册兼容的供应商
@register_provider("custom")
class CustomOpenAIProvider(OpenAICompatibleProvider):
    """自定义OpenAI兼容接口"""
    pass


@register_provider("localai")
class LocalAIProvider(OpenAICompatibleProvider):
    """LocalAI实现"""
    pass


@register_provider("ollama")
class OllamaProvider(OpenAICompatibleProvider):
    """Ollama实现"""
    
    def __init__(self, config: Dict[str, Any]):
        # Ollama默认使用11434端口
        if not config.get("base_url"):
            config["base_url"] = "http://localhost:11434/v1"
        super().__init__(config)
    
    def get_provider_name(self) -> str:
        return "ollama"
