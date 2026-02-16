# [Input] `zhipuai` SDK 导入行为与 ZhipuProvider 的客户端初始化逻辑。
# [Output] 验证同步回退与流式生成兼容性。
# [Pos] LLM provider 单元测试，覆盖 zhipu SDK 版本兼容回归。

import sys
import types

import pytest

from src.core.interfaces import AIRequest, Message, MessageRole
from src.providers.llm.zhipu import ZhipuProvider


class _DummyUsage:
    prompt_tokens = 3
    completion_tokens = 4
    total_tokens = 7


class _DummyMessage:
    content = "ok"


class _DummyChoice:
    message = _DummyMessage()
    finish_reason = "stop"


class _DummyResponse:
    usage = _DummyUsage()
    choices = [_DummyChoice()]
    model = "GLM-4.7"


class _DummyDelta:
    def __init__(self, content: str):
        self.content = content


class _DummyStreamChoice:
    def __init__(self, content: str):
        self.delta = _DummyDelta(content)


class _DummyStreamChunk:
    def __init__(self, content: str):
        self.choices = [_DummyStreamChoice(content)]


class _DummyStream:
    def __init__(self, contents: list[str]):
        self._iterator = iter(contents)

    def __iter__(self):
        return self

    def __next__(self):
        content = next(self._iterator)
        return _DummyStreamChunk(content)


class _DummyCompletions:
    def create(self, **kwargs):
        if kwargs.get("stream"):
            return _DummyStream(["a", "b"])
        return _DummyResponse()


class _DummyChat:
    completions = _DummyCompletions()


class _DummyZhipuAI:
    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = _DummyChat()


def _install_fake_zhipu_module(monkeypatch):
    fake_module = types.ModuleType("zhipuai")
    fake_module.ZhipuAI = _DummyZhipuAI
    monkeypatch.setitem(sys.modules, "zhipuai", fake_module)


@pytest.mark.asyncio
async def test_fallback_to_sync_zhipu_client_when_async_class_missing(monkeypatch):
    _install_fake_zhipu_module(monkeypatch)
    provider = ZhipuProvider({"api_key": "test-key", "provider_name": "zhipu"})

    client = await provider._get_async_client()

    assert client is not None
    assert isinstance(client, _DummyZhipuAI)


@pytest.mark.asyncio
async def test_generate_with_sync_zhipu_client_fallback(monkeypatch):
    _install_fake_zhipu_module(monkeypatch)
    provider = ZhipuProvider({"api_key": "test-key", "provider_name": "zhipu"})
    request = AIRequest(
        model="GLM-4.7",
        messages=[Message(role=MessageRole.USER, content="hello")],
    )

    response = await provider.generate(request)

    assert response.content == "ok"
    assert response.usage.total_tokens == 7


@pytest.mark.asyncio
async def test_stream_generate_with_sync_stream(monkeypatch):
    _install_fake_zhipu_module(monkeypatch)
    provider = ZhipuProvider({"api_key": "test-key", "provider_name": "zhipu"})
    request = AIRequest(
        model="GLM-4.7",
        messages=[Message(role=MessageRole.USER, content="hello")],
    )

    chunks = []
    async for chunk in provider.stream_generate(request):
        chunks.append(chunk)

    assert chunks == ["a", "b"]
