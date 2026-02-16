# [Input] AgentConfig 模型名、Provider 配置与 BaseAgent 解析逻辑。
# [Output] 验证 Agent 按配置精确匹配模型到 provider，并处理歧义错误。
# [Pos] unit 测试层回归用例，覆盖 Agent provider 解析与执行路径。

from types import SimpleNamespace

import pytest

from src.agents.base import AgentConfig, AgentType
from src.agents.conversational_agent import ConversationalAgent
from src.agents.react_agent import ReActAgent
from src.core.interfaces import AIResponse, UsageInfo
from src.providers.llm.factory import LLMProviderFactory


class _DummyProvider:
    async def generate(self, request):
        return AIResponse(
            content="ok",
            usage=UsageInfo(total_tokens=2),
            model=request.model,
            finish_reason="stop",
        )


@pytest.mark.parametrize(
    ("model", "models_map", "expected_provider"),
    [
        (
            "chat-max-001",
            {
                "openai": {"default": "gpt-4o"},
                "zhipu": {"default": "chat-max-001"},
            },
            "zhipu",
        ),
        (
            "x-company-model",
            {
                "openai": {"default": "x-company-model"},
                "zhipu": {"default": "GLM-4.7"},
            },
            "openai",
        ),
    ],
)
def test_base_agent_resolves_provider_from_config_exact_match(
    monkeypatch, model, models_map, expected_provider
):
    captured = {}

    monkeypatch.setattr(
        LLMProviderFactory,
        "list_providers",
        staticmethod(lambda: ["openai", "zhipu"]),
    )

    def fake_get_provider_config(provider_name: str):
        models = models_map.get(provider_name, {})
        return SimpleNamespace(models=models)

    monkeypatch.setattr("src.agents.base.get_provider_config", fake_get_provider_config)

    def fake_create_from_config(provider_name: str):
        captured["provider_name"] = provider_name
        return _DummyProvider()

    monkeypatch.setattr(
        LLMProviderFactory,
        "create_from_config",
        staticmethod(fake_create_from_config),
    )

    agent = ReActAgent(
        AgentConfig(
            name="resolver",
            agent_type=AgentType.REACT,
            model=model,
            max_iterations=1,
        )
    )

    _ = agent._get_llm_provider()

    assert captured["provider_name"] == expected_provider


def test_base_agent_resolve_provider_raises_when_model_is_ambiguous(monkeypatch):
    monkeypatch.setattr(
        LLMProviderFactory,
        "list_providers",
        staticmethod(lambda: ["openai", "zhipu"]),
    )

    def fake_get_provider_config(provider_name: str):
        if provider_name in {"openai", "zhipu"}:
            return SimpleNamespace(models={"default": "shared-model"})
        return SimpleNamespace(models={})

    monkeypatch.setattr("src.agents.base.get_provider_config", fake_get_provider_config)

    agent = ReActAgent(
        AgentConfig(
            name="resolver",
            agent_type=AgentType.REACT,
            model="shared-model",
            max_iterations=1,
        )
    )

    with pytest.raises(ValueError, match="multiple providers"):
        agent._get_llm_provider()


def test_base_agent_resolve_provider_raises_when_model_not_configured(monkeypatch):
    monkeypatch.setattr(
        LLMProviderFactory,
        "list_providers",
        staticmethod(lambda: ["openai", "zhipu"]),
    )
    monkeypatch.setattr(
        "src.agents.base.get_provider_config",
        lambda provider_name: SimpleNamespace(models={}),
    )

    agent = ReActAgent(
        AgentConfig(
            name="resolver",
            agent_type=AgentType.REACT,
            model="unknown-model",
            max_iterations=1,
        )
    )

    with pytest.raises(ValueError, match="Unable to resolve provider"):
        agent._get_llm_provider()


@pytest.mark.asyncio
async def test_conversational_agent_execute_with_glm_model(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        LLMProviderFactory,
        "list_providers",
        staticmethod(lambda: ["openai", "zhipu"]),
    )

    def fake_get_provider_config(provider_name: str):
        if provider_name == "openai":
            return SimpleNamespace(models={"default": "gpt-4o"})
        if provider_name == "zhipu":
            return SimpleNamespace(models={"default": "chat-max-001"})
        return SimpleNamespace(models={})

    monkeypatch.setattr("src.agents.base.get_provider_config", fake_get_provider_config)

    def fake_create_from_config(provider_name: str):
        captured["provider_name"] = provider_name
        return _DummyProvider()

    monkeypatch.setattr(
        LLMProviderFactory,
        "create_from_config",
        staticmethod(fake_create_from_config),
    )

    agent = ConversationalAgent(
        AgentConfig(
            name="assistant",
            agent_type=AgentType.CONVERSATIONAL,
            model="chat-max-001",
        )
    )

    result = await agent.execute("你好", conversation_id="cid-provider")

    assert captured["provider_name"] == "zhipu"
    assert result.success is True
    assert result.output == "ok"
