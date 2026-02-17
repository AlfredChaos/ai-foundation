# [Input] MinimaxProvider 与 anthropic SDK 行为模拟。
# [Output] 验证 Minimax 通过 anthropic 协议完成生成与流式输出。
# [Pos] unit 测试层 minimax provider 协议兼容回归。

import sys
import types

import pytest

from src.core.interfaces import AIRequest, Message, MessageRole
from src.providers.llm.minimax import MinimaxProvider


class _Usage:
    input_tokens = 3
    output_tokens = 4


class _TextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _GenerateResponse:
    stop_reason = "end_turn"
    model = "abab6.5s-chat"
    usage = _Usage()

    def __init__(self):
        self.content = [_TextBlock("hello from minimax")]


class _StreamDelta:
    type = "text_delta"

    def __init__(self, text: str):
        self.text = text


class _StreamChunk:
    type = "content_block_delta"

    def __init__(self, text: str):
        self.delta = _StreamDelta(text)


class _Stream:
    def __init__(self, texts):
        self._texts = list(texts)

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._texts):
            raise StopAsyncIteration
        text = self._texts[self._index]
        self._index += 1
        return _StreamChunk(text)


class _FakeMessages:
    def __init__(self):
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return _Stream(["A", "B"])
        return _GenerateResponse()


class _FakeAsyncAnthropic:
    def __init__(self, api_key=None, auth_token=None, base_url=None, default_headers=None, **kwargs):
        self.api_key = api_key
        self.auth_token = auth_token
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.messages = _FakeMessages()


def _install_fake_anthropic_module(monkeypatch):
    fake_module = types.ModuleType("anthropic")
    fake_module.AsyncAnthropic = _FakeAsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)


@pytest.mark.asyncio
async def test_minimax_generate_uses_anthropic_messages_api(monkeypatch):
    _install_fake_anthropic_module(monkeypatch)
    provider = MinimaxProvider({"api_key": "k", "provider_name": "minimax", "base_url": "https://mock"})

    response = await provider.generate(
        AIRequest(
            model="abab6.5s-chat",
            messages=[Message(role=MessageRole.USER, content="hello")],
        )
    )

    client = await provider._get_async_client()
    assert response.content == "hello from minimax"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 7
    assert client.api_key == "k"
    assert client.auth_token == "k"
    assert client.messages.last_kwargs is not None
    assert client.messages.last_kwargs["model"] == "abab6.5s-chat"


@pytest.mark.asyncio
async def test_minimax_stream_generate_uses_anthropic_stream(monkeypatch):
    _install_fake_anthropic_module(monkeypatch)
    provider = MinimaxProvider({"api_key": "k", "provider_name": "minimax", "base_url": "https://mock"})

    chunks = []
    async for chunk in provider.stream_generate(
        AIRequest(
            model="abab6.5s-chat",
            messages=[Message(role=MessageRole.USER, content="hello")],
            stream=True,
        )
    ):
        chunks.append(chunk)

    assert chunks == ["A", "B"]


@pytest.mark.asyncio
async def test_minimax_generate_raises_when_api_key_missing(monkeypatch):
    _install_fake_anthropic_module(monkeypatch)
    provider = MinimaxProvider({"api_key": "", "provider_name": "minimax", "base_url": "https://mock"})

    with pytest.raises(ValueError, match="Minimax API key is missing"):
        await provider.generate(
            AIRequest(
                model="abab6.5s-chat",
                messages=[Message(role=MessageRole.USER, content="hello")],
            )
        )


@pytest.mark.asyncio
async def test_minimax_generate_raises_when_api_key_is_placeholder(monkeypatch):
    _install_fake_anthropic_module(monkeypatch)
    provider = MinimaxProvider(
        {"api_key": "${MINIMAX_API_KEY}", "provider_name": "minimax", "base_url": "https://mock"}
    )

    with pytest.raises(ValueError, match="Minimax API key is missing"):
        await provider.generate(
            AIRequest(
                model="abab6.5s-chat",
                messages=[Message(role=MessageRole.USER, content="hello")],
            )
        )
