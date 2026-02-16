# [Input] ConversationalAgent 与 ContextManager 的异步上下文行为。
# [Output] 验证对话执行与上下文信息获取在 asyncio 下可用。
# [Pos] unit 测试层异步回归用例，覆盖会话与上下文管理关键路径。

import pytest

from src.agents.base import AgentConfig, AgentType
from src.agents.conversational_agent import ConversationalAgent
from src.context.manager import ContextManager
from src.core.interfaces import AIResponse, UsageInfo


class _FakeLLMProvider:
    async def generate(self, request):
        return AIResponse(
            content="你好张三，我记住你了。",
            usage=UsageInfo(total_tokens=5),
            model=request.model,
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_conversational_agent_execute_records_history():
    agent = ConversationalAgent(
        AgentConfig(
            name="assistant",
            agent_type=AgentType.CONVERSATIONAL,
            model="GLM-4.7",
        )
    )
    agent._set_llm_provider(_FakeLLMProvider())

    result = await agent.execute("我叫张三", conversation_id="cid-1")

    assert result.success is True
    assert result.output

    history = await agent._context_manager.get_messages("cid-1")
    assert len(history) == 2
    assert history[0].role.value == "user"
    assert history[1].role.value == "assistant"


@pytest.mark.asyncio
async def test_context_manager_get_context_info_in_async_loop():
    manager = ContextManager()
    await manager.add_message("cid-ctx", "user", "hello")

    info = await manager.get_context_info("cid-ctx")

    assert info["conversation_id"] == "cid-ctx"
    assert info["message_count"] == 1
    assert isinstance(info["token_count"], int)
