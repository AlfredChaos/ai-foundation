# [Input] ReActAgent 执行链路与 LLM 输出文本。
# [Output] 验证 ReAct 在非严格格式输出下仍能正确收敛。
# [Pos] unit 测试层 ReAct 行为回归用例。

import pytest

from src.agents.base import AgentConfig, AgentType
from src.agents.react_agent import ReActAgent
from src.core.interfaces import AIResponse, UsageInfo


class _PlainAnswerProvider:
    async def generate(self, request):
        return AIResponse(
            content="The capital of France is Paris.",
            usage=UsageInfo(total_tokens=6),
            model=request.model,
            finish_reason="stop",
        )


class _LowerCaseActionProvider:
    async def generate(self, request):
        return AIResponse(
            content="Thought: done\nAction: none\nAction Input: {}",
            usage=UsageInfo(total_tokens=4),
            model=request.model,
            finish_reason="stop",
        )


class _ToolActionWithoutToolsProvider:
    async def generate(self, request):
        return AIResponse(
            content='THOUGHT: I can answer directly.\nACTION: search\nACTION INPUT: {"q": "France capital"}',
            usage=UsageInfo(total_tokens=6),
            model=request.model,
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_react_agent_finishes_when_model_returns_plain_answer():
    agent = ReActAgent(
        AgentConfig(
            name="react",
            agent_type=AgentType.REACT,
            model="chat-max-001",
            max_iterations=3,
        )
    )
    agent._set_llm_provider(_PlainAnswerProvider())

    result = await agent.execute("What is the capital of France?")

    assert result.success is True
    assert "Paris" in result.output
    assert len(result.intermediate_steps) == 1


@pytest.mark.asyncio
async def test_react_agent_parses_action_case_insensitively():
    agent = ReActAgent(
        AgentConfig(
            name="react",
            agent_type=AgentType.REACT,
            model="chat-max-001",
            max_iterations=3,
        )
    )
    agent._set_llm_provider(_LowerCaseActionProvider())

    result = await agent.execute("Any task")

    assert result.success is True
    assert result.output
    assert len(result.intermediate_steps) == 1


@pytest.mark.asyncio
async def test_react_agent_finishes_when_action_requested_but_no_tools_configured():
    agent = ReActAgent(
        AgentConfig(
            name="react",
            agent_type=AgentType.REACT,
            model="chat-max-001",
            max_iterations=3,
        )
    )
    agent._set_llm_provider(_ToolActionWithoutToolsProvider())

    result = await agent.execute("Any task")

    assert result.success is True
    assert "directly" in result.output
    assert len(result.intermediate_steps) == 1
