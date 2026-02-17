# [Input] minimax 使用示例入口与依赖对象的 mock 行为。
# [Output] 验证 minimax 示例脚本主流程可完整执行。
# [Pos] unit 测试层示例脚本回归，覆盖 minimax 全流程路径。

from types import SimpleNamespace

import pytest


class _FakeAI:
    async def chat(self, message):
        return "hello"

    async def stream(self, message):
        for chunk in ["A", "B"]:
            yield chunk


class _FakeReActAgent:
    def __init__(self, config):
        self.config = config

    async def execute(self, task):
        return SimpleNamespace(success=True, output="Paris", error=None)


class _FakeConversationalAgent:
    def __init__(self, config):
        self.config = config

    async def execute(self, task, conversation_id):
        if "名字是什么" in task:
            return SimpleNamespace(success=True, output="你叫张三", error=None)
        return SimpleNamespace(success=True, output="我记住了", error=None)


class _FakeToolManager:
    async def execute_tool(self, name, **kwargs):
        if name == "calculator":
            return 14
        return "2026-01-01 00:00:00"

    def list_tools(self):
        return [{"name": "calculator"}, {"name": "get_datetime"}]


class _FakeBuiltinTools:
    @staticmethod
    def register_builtins(tool_manager):
        return None


class _FakeProvider:
    def get_model_info(self):
        return {"available": True, "models": {"default": "abab6.5s-chat"}}


class _FakeContextManager:
    def __init__(self):
        self._messages = []

    async def add_message(self, conversation_id, role, content):
        self._messages.append((conversation_id, role, content))

    async def get_messages(self, conversation_id):
        return self._messages

    def calculate_tokens(self, messages):
        return 12

    async def get_context_info(self, conversation_id):
        return {"conversation_id": conversation_id, "message_count": len(self._messages), "token_count": 12}


@pytest.mark.asyncio
async def test_minimax_usage_main_full_flow(monkeypatch, capsys):
    import examples.minimax_usage as minimax_usage

    created_ai = {}

    def _fake_create_ai(provider, model):
        created_ai["provider"] = provider
        created_ai["model"] = model
        return _FakeAI()

    monkeypatch.setattr(minimax_usage, "create_ai", _fake_create_ai)
    monkeypatch.setattr(minimax_usage, "ReActAgent", _FakeReActAgent)
    monkeypatch.setattr(minimax_usage, "ConversationalAgent", _FakeConversationalAgent)
    monkeypatch.setattr(minimax_usage, "ToolManager", _FakeToolManager)
    monkeypatch.setattr(minimax_usage, "BuiltinTools", _FakeBuiltinTools)
    monkeypatch.setattr(minimax_usage, "get_llm_provider", lambda provider_name: _FakeProvider())
    monkeypatch.setattr(minimax_usage, "ContextManager", _FakeContextManager)

    await minimax_usage.main()

    output = capsys.readouterr().out
    assert created_ai["provider"] == "minimax"
    assert created_ai["model"] == minimax_usage.MODEL
    assert "Minimax LLM 使用示例" in output
    assert "1. 基础对话示例" in output
    assert "2. 流式对话示例" in output
    assert "3. Agent执行示例" in output
    assert "5. 多轮对话示例" in output
    assert "7. 上下文管理示例" in output
