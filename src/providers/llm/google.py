# Google Gemini供应商实现

from typing import AsyncIterator, Dict, Any, Optional, List

from src.core.interfaces import AIRequest, AIResponse, UsageInfo, Message
from src.providers.llm.factory import BaseLLMProvider, register_provider


@register_provider("google")
class GoogleProvider(BaseLLMProvider):
    """Google Gemini供应商实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
    
    async def _get_client(self):
        """获取客户端"""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                self._logger.warning("Google GenAI package not installed")
        return self._client
    
    async def generate(self, request: AIRequest) -> AIResponse:
        """生成回复"""
        client = await self._get_client()
        if client is None:
            raise ImportError("Google GenAI package not installed")
        
        # 合并消息为一个提示
        prompt = self._build_prompt(request.messages)
        
        # Gemini使用不同的API
        try:
            response = client.models.generate_content(
                model=request.model,
                contents=prompt,
                config={
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_tokens,
                }
            )
            
            # 解析响应
            content = ""
            usage = UsageInfo()
            
            if response.text:
                content = response.text
            
            # 获取使用统计
            if hasattr(response, 'usage_metadata'):
                usage = UsageInfo(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    completion_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=(
                        response.usage_metadata.prompt_token_count +
                        response.usage_metadata.candidates_token_count
                    ),
                )
            
            finish_reason = "stop"
            if hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'finish_reason'):
                    finish_reason = str(response.candidates[0].finish_reason)
            
            return AIResponse(
                content=content,
                usage=usage,
                model=request.model,
                finish_reason=finish_reason,
            )
        except Exception as e:
            self._logger.error(f"Google Gemini API error: {e}")
            raise
    
    async def stream_generate(self, request: AIRequest) -> AsyncIterator[str]:
        """流式生成"""
        client = await self._get_client()
        if client is None:
            raise ImportError("Google GenAI package not installed")
        
        prompt = self._build_prompt(request.messages)
        
        try:
            stream = client.models.generate_content_stream(
                model=request.model,
                contents=prompt,
                config={
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_tokens,
                }
            )
            
            for chunk in stream:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
        except Exception as e:
            self._logger.error(f"Google Gemini streaming error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """计算Token数"""
        # Google的tokenizer实现
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            response = client.models.count_tokens(
                model=self.models.get("default", "gemini-1.5-pro"),
                contents=text,
            )
            return response.total_tokens
        except ImportError:
            return len(text) // 4
        except Exception:
            return len(text) // 4
    
    def _build_prompt(self, messages: List[Message]) -> str:
        """构建提示 - Gemini使用不同的消息格式"""
        parts = []
        
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else msg.role
            
            if role == "system":
                parts.append(f"System: {msg.content}")
            elif role == "user":
                parts.append(f"User: {msg.content}")
            elif role == "assistant":
                parts.append(f"Assistant: {msg.content}")
            else:
                parts.append(msg.content)
        
        return "\n\n".join(parts)
    
    def get_provider_name(self) -> str:
        """获取供应商名称"""
        return "google"
