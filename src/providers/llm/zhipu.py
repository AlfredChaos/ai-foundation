# [Input] 供应商配置（api_key/base_url）与 zhipuai SDK。
# [Output] 提供智谱模型的异步生成与流式生成能力。
# [Pos] LLM provider 层智谱实现，兼容新旧 SDK 客户端差异。

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from src.core.interfaces import AIRequest, AIResponse, Message, UsageInfo
from src.providers.llm.factory import BaseLLMProvider, register_provider


@register_provider("zhipu")
class ZhipuProvider(BaseLLMProvider):
    """智谱ZAI供应商实现"""

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._async_client = None
        self._client_init_error = None

    async def _get_async_client(self):
        """获取客户端，优先异步实现，回退到同步实现"""
        if self._async_client is None:
            try:
                from zhipuai import AsyncZhipuAI

                self._async_client = AsyncZhipuAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                )
            except ImportError as async_import_error:
                try:
                    from zhipuai import ZhipuAI

                    self._async_client = ZhipuAI(
                        api_key=self.api_key,
                        base_url=self.base_url if self.base_url else None,
                    )
                    # 新版 zhipuai 移除了 AsyncZhipuAI，使用同步客户端回退
                    self._logger.info(
                        f"AsyncZhipuAI unavailable, fallback to ZhipuAI: {async_import_error}"
                    )
                except ImportError as sync_import_error:
                    self._client_init_error = sync_import_error
                    self._logger.warning(
                        f"ZhipuAI package import failed: {sync_import_error}"
                    )
        return self._async_client

    async def _create_completion(self, client, params: dict[str, Any]):
        """统一调用 create，兼容同步/异步客户端"""
        create_fn = client.chat.completions.create
        if asyncio.iscoroutinefunction(create_fn):
            return await create_fn(**params)
        return await asyncio.to_thread(create_fn, **params)

    def _raise_client_import_error(self) -> None:
        """抛出更准确的客户端初始化异常"""
        if self._client_init_error is not None:
            raise ImportError(
                f"ZhipuAI package import failed: {self._client_init_error}"
            ) from self._client_init_error
        raise ImportError("ZhipuAI package not installed")

    async def generate(self, request: AIRequest) -> AIResponse:
        """生成回复"""
        client = await self._get_async_client()
        if client is None:
            self._raise_client_import_error()

        messages = self._convert_messages(request.messages)

        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        try:
            response = await self._create_completion(client, params)

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
            self._logger.error(f"Zhipu API error: {e}")
            raise

    async def stream_generate(self, request: AIRequest) -> AsyncIterator[str]:
        """流式生成"""
        client = await self._get_async_client()
        if client is None:
            self._raise_client_import_error()

        messages = self._convert_messages(request.messages)

        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        try:
            stream = await self._create_completion(client, params)

            if hasattr(stream, "__aiter__"):
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return

            iterator = iter(stream)
            while True:
                try:
                    chunk = await asyncio.to_thread(next, iterator)
                except StopIteration:
                    break
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self._logger.error(f"Zhipu streaming error: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """计算Token数"""
        # 估算
        return len(text) // 4

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """转换消息格式"""
        result = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else msg.role
            result.append(
                {
                    "role": role,
                    "content": msg.content,
                }
            )
        return result

    def get_provider_name(self) -> str:
        """获取供应商名称"""
        return "zhipu"
