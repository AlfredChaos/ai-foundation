# DeepSeek供应商实现

from typing import AsyncIterator, Dict, Any, List

from src.core.interfaces import AIRequest, AIResponse, UsageInfo, Message
from src.providers.llm.factory import BaseLLMProvider, register_provider


@register_provider("deepseek")
class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek供应商实现"""
    
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._async_client = None
    
    async def _get_async_client(self):
        """获取异步客户端"""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                self._logger.warning("OpenAI package not installed")
        return self._async_client
    
    async def generate(self, request: AIRequest) -> AIResponse:
        """生成回复"""
        client = await self._get_async_client()
        if client is None:
            raise ImportError("OpenAI package not installed")
        
        messages = self._convert_messages(request.messages)
        
        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        
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
            )
        except Exception as e:
            self._logger.error(f"DeepSeek API error: {e}")
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
        
        try:
            stream = await client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self._logger.error(f"DeepSeek streaming error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """计算Token数"""
        # 估算
        return len(text) // 4
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """转换消息格式"""
        result = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else msg.role
            result.append({
                "role": role,
                "content": msg.content,
            })
        return result
    
    def get_provider_name(self) -> str:
        """获取供应商名称"""
        return "deepseek"
