# 单元测试 - LLM供应商

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


class TestOpenAIProvider:
    """测试OpenAI供应商"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI客户端"""
        with patch('src.providers.llm.openai.OpenAI') as mock:
            yield mock
    
    @pytest.fixture
    def provider_config(self):
        """供应商配置"""
        return {
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "models": {"default": "gpt-4o"},
            "provider_name": "openai"
        }


class TestLLMProviderFactory:
    """测试LLM工厂"""
    
    def test_factory_creates_provider(self):
        """测试工厂创建供应商"""
        from src.providers.llm.factory import LLMProviderFactory
        
        # 注册测试供应商
        class TestProvider:
            name = "test"
        
        LLMProviderFactory.register("test", TestProvider)
        
        provider = LLMProviderFactory.get_provider("test", {})
        assert provider is not None
    
    def test_factory_list_providers(self):
        """测试工厂列出供应商"""
        from src.providers.llm.factory import LLMProviderFactory
        
        providers = LLMProviderFactory.list_providers()
        assert isinstance(providers, list)
        assert "openai" in providers


class TestConfigManager:
    """测试配置管理器"""
    
    def test_config_loads(self):
        """测试配置加载"""
        from src.config.manager import ConfigManager
        
        manager = ConfigManager()
        # 应该能加载默认配置
        config = manager.load()
        assert config is not None
    
    def test_get_nested_config(self):
        """测试获取嵌套配置"""
        from src.config.manager import get_config
        
        # 应该能获取配置
        result = get_config()
        assert result is not None


class TestTokenCounter:
    """测试Token计数"""
    
    def test_count_text(self):
        """测试文本计数"""
        from src.services.token_service import TokenCounter
        
        counter = TokenCounter()
        tokens = counter.count("Hello, world!")
        
        assert tokens >= 1
    
    def test_estimate_cost(self):
        """测试成本估算"""
        from src.services.token_service import TokenCounter
        
        counter = TokenCounter()
        cost = counter.estimate_cost(
            prompt_tokens=100,
            completion_tokens=100,
            model="gpt-4o"
        )
        
        assert cost >= 0


class TestToolManager:
    """测试工具管理器"""
    
    @pytest.fixture
    def tool_manager(self):
        """创建工具管理器"""
        from src.tools.tool_manager import ToolManager
        return ToolManager()
    
    def test_register_tool(self, tool_manager):
        """测试注册工具"""
        async def dummy_func(x: int, y: int) -> int:
            return x + y
        
        tool_manager.register_tool(
            name="add",
            func=dummy_func,
            description="Add two numbers"
        )
        
        tools = tool_manager.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "add"
    
    def test_execute_tool(self, tool_manager):
        """测试执行工具"""
        async def add(x: int, y: int) -> int:
            return x + y
        
        tool_manager.register_tool(
            name="add",
            func=add,
            description="Add two numbers"
        )
        
        result = asyncio.run(tool_manager.execute_tool("add", x=2, y=3))
        assert result == 5
    
    def test_builtin_tools(self, tool_manager):
        """测试内置工具"""
        from src.tools.tool_manager import BuiltinTools
        
        BuiltinTools.register_builtins(tool_manager)
        
        tools = tool_manager.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "search" in tool_names
        assert "calculator" in tool_names
        assert "file_reader" in tool_names


class TestContextManager:
    """测试上下文管理器"""
    
    @pytest.fixture
    def context_manager(self):
        """创建上下文管理器"""
        from src.context.manager import ContextManager
        return ContextManager()
    
    @pytest.mark.asyncio
    async def test_add_message(self, context_manager):
        """测试添加消息"""
        await context_manager.add_message(
            conversation_id="test-123",
            role="user",
            content="Hello"
        )
        
        messages = await context_manager.get_messages("test-123")
        assert len(messages) == 1
    
    @pytest.mark.asyncio
    async def test_clear_context(self, context_manager):
        """测试清空上下文"""
        await context_manager.add_message("test-123", "user", "Hello")
        await context_manager.add_message("test-123", "assistant", "Hi")
        
        await context_manager.clear_context("test-123")
        
        messages = await context_manager.get_messages("test-123")
        assert len(messages) == 0
    
    def test_calculate_tokens(self, context_manager):
        """测试Token计算"""
        from src.core.interfaces import Message, MessageRole
        
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
        
        tokens = context_manager.calculate_tokens(messages)
        assert tokens >= 1


class TestMemoryProvider:
    """测试记忆存储"""
    
    @pytest.fixture
    def in_memory_provider(self):
        """创建内存存储"""
        from src.memory.providers import InMemoryProvider
        return InMemoryProvider()
    
    @pytest.mark.asyncio
    async def test_store_memory(self, in_memory_provider):
        """测试存储记忆"""
        result = await in_memory_provider.store(
            key="test-key",
            content="test content",
            metadata={"type": "test"}
        )
        assert result is True
    
    @pytest.mark.asyncio
    async def test_retrieve_memory(self, in_memory_provider):
        """测试检索记忆"""
        await in_memory_provider.store("key1", "content1")
        
        memory = await in_memory_provider.retrieve("key1")
        assert memory is not None
        assert memory.content == "content1"
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, in_memory_provider):
        """测试删除记忆"""
        await in_memory_provider.store("key1", "content1")
        result = await in_memory_provider.delete("key1")
        
        assert result is True
        
        memory = await in_memory_provider.retrieve("key1")
        assert memory is None
    
    @pytest.mark.asyncio
    async def test_search_memory(self, in_memory_provider):
        """测试搜索记忆"""
        await in_memory_provider.store("key1", "apple fruit")
        await in_memory_provider.store("key2", "banana fruit")
        await in_memory_provider.store("key3", "carrot vegetable")
        
        results = await in_memory_provider.search("fruit", top_k=10)
        
        assert len(results) == 2


class TestReActAgent:
    """测试ReAct Agent"""
    
    @pytest.fixture
    def react_agent(self):
        """创建ReAct Agent"""
        from src.agents import ReActAgent, AgentConfig, AgentType
        
        config = AgentConfig(
            name="test-agent",
            agent_type=AgentType.REACT,
            model="gpt-4o-mini",
            max_iterations=3,
        )
        
        return ReActAgent(config)
    
    @pytest.mark.asyncio
    async def test_agent_execute(self, react_agent):
        """测试Agent执行"""
        result = await react_agent.execute("What is 2 + 2?")
        
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'output')
    
    def test_agent_info(self, react_agent):
        """测试Agent信息"""
        info = react_agent.get_info()
        
        assert info["name"] == "test-agent"
        assert info["type"] == "react"
        assert info["model"] == "gpt-4o-mini"


class TestConversationalAgent:
    """测试对话Agent"""
    
    @pytest.fixture
    def conv_agent(self):
        """创建对话Agent"""
        from src.agents import ConversationalAgent, AgentConfig
        
        config = AgentConfig(
            name="chatbot",
            agent_type=AgentType.CONVERSATIONAL,
            model="gpt-4o-mini",
        )
        
        return ConversationalAgent(config)
    
    @pytest.mark.asyncio
    async def test_conversation(self, conv_agent):
        """测试对话"""
        result = await conv_agent.execute(
            "Hello, my name is Alice",
            conversation_id="test-conv"
        )
        
        assert result.success is True
        assert len(result.output) > 0


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
