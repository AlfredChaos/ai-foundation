# [Input] Minimax API 配置（api_key/base_url）与 anthropic SDK 依赖。
# [Output] 提供基于 anthropic 协议的 Minimax 生成与流式能力。
# [Pos] LLM provider 层 minimax 实现，兼容 anthropic messages 协议。

from collections.abc import AsyncIterator
from typing import Any

from src.core.interfaces import AIRequest, AIResponse, Message, UsageInfo
from src.providers.llm.factory import BaseLLMProvider, register_provider


@register_provider("minimax")
class MinimaxProvider(BaseLLMProvider):
    """Minimax供应商实现（Anthropic协议兼容）"""

    DEFAULT_BASE_URL = "https://api.minimax.chat/v1"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._async_client = None

    async def _get_async_client(self):
        """获取异步客户端"""
        if self._async_client is None:
            api_key = (self.api_key or "").strip()
            if api_key.startswith("${") and api_key.endswith("}"):
                api_key = ""
            if not api_key:
                raise ValueError(
                    "Minimax API key is missing. Set MINIMAX_API_KEY or providers.minimax.api_key."
                )
            try:
                from anthropic import AsyncAnthropic

                self._async_client = AsyncAnthropic(
                    api_key=api_key,
                    auth_token=api_key,
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

        messages = self._convert_messages(request.messages)

        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 4096,
        }
        if request.json_mode:
            params["system"] = "You must respond in valid JSON."

        try:
            response = await client.messages.create(**params)

            finish_reason = response.stop_reason
            if finish_reason == "end_turn":
                finish_reason = "stop"
            elif finish_reason == "max_tokens":
                finish_reason = "length"

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
            )
        except Exception as error:
            self._logger.error(f"Minimax API error: {error}")
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
        except Exception as error:
            self._logger.error(f"Minimax streaming error: {error}")
            raise

    def count_tokens(self, text: str) -> int:
        """计算Token数"""
        return len(text) // 4

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """转换消息格式（兼容 anthropic 角色约束）"""
        result = []
        for message in messages:
            role = message.role.value if hasattr(message.role, "value") else message.role
            if role == "system":
                if result and result[-1]["role"] == "user":
                    result[-1]["content"] = f"{message.content}\n\n{result[-1]['content']}"
                else:
                    result.append({"role": "user", "content": message.content})
            else:
                result.append({"role": role, "content": message.content})
        return result

    def get_provider_name(self) -> str:
        """获取供应商名称"""
        return "minimax"
