# Anthropic供应商实现

from typing import AsyncIterator, Dict, Any, Optional, List

from src.core.interfaces import AIRequest, AIResponse, UsageInfo, Message
from src.providers.llm.factory import BaseLLMProvider, register_provider


@register_provider("anthropic")
class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude供应商实现"""
    
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self._async_client = None
    
    async def _get_async_client(self):
        """获取异步客户端"""
        if self._async_client is None:
            try:
                from anthropic import AsyncAnthropic
                self._async_client = AsyncAnthropic(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                )
            except ImportError:
                self._logger.warning("Anthropic package not installed")
        return self._async_client
    
    async def generate(self, request: AIRequest) -> AIResponse:
        """生成回复"""
        client = await self._get_async_client()
        if client is None:
            raise ImportError("Anthropic package not installed")
        
        # 转换消息格式
        messages = self._convert_messages(request.messages)
        
        # Anthropic使用max_tokens参数
        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 4096,
        }
        
        if request.json_mode:
            # Anthropic需要明确指定JSON格式
            params["system"] = params.get("system", "") + "\n\nYou must respond in valid JSON."
        
        try:
            response = await client.messages.create(**params)
            
            # 解析stop_reason
            finish_reason = response.stop_reason
            if finish_reason == "end_turn":
                finish_reason = "stop"
            elif finish_reason == "max_tokens":
                finish_reason = "length"
            
            # 提取工具调用
            tool_calls = None
            if response.content and hasattr(response.content[0], 'type'):
                if response.content[0].type == "tool_use":
                    tool_calls = [{
                        "id": response.content[0].id,
                        "type": "tool_use",
                        "function": {
                            "name": response.content[0].name,
                            "arguments": response.content[0].input,
                        }
                    }]
            
            content = ""
            if response.content:
                for block in response.content:
                    if block.type == "text":
                        content += block.text
            
            return AIResponse(
                content=content,
                usage=UsageInfo(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                ),
                model=response.model,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )
        except Exception as e:
            self._logger.error(f"Anthropic API error: {e}")
            raise
    
    async def stream_generate(self, request: AIRequest) -> AsyncIterator[str]:
        """流式生成"""
        client = await self._get_async_client()
        if client is None:
            raise ImportError("Anthropic package not installed")
        
        messages = self._convert_messages(request.messages)
        
        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 4096,
            "stream": True,
        }
        
        try:
            stream = await client.messages.create(**params)
            
            async for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                    yield chunk.delta.text
        except Exception as e:
            self._logger.error(f"Anthropic streaming error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """计算Token数"""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            response = client.count_tokens(text)
            return response
        except ImportError:
            # 估算
            return len(text) // 4
        except Exception:
            return len(text) // 4
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """转换消息格式 - Anthropic要求最后一条必须是user或assistant"""
        result = []
        
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else msg.role
            
            # Anthropic只支持user和assistant角色
            if role == "system":
                # System消息需要特殊处理
                if result and result[-1]["role"] == "user":
                    # 将system附加到最后一个user消息
                    result[-1]["content"] = f"{msg.content}\n\n" + result[-1]["content"]
                else:
                    # 添加一个user消息作为system
                    result.append({
                        "role": "user",
                        "content": msg.content,
                    })
            else:
                result.append({
                    "role": role,
                    "content": msg.content,
                })
        
        return result
    
    def get_provider_name(self) -> str:
        """获取供应商名称"""
        return "anthropic"
