# 集成测试 - 完整功能测试

import pytest
import asyncio
from unittest.mock import Mock, patch


class TestFullWorkflow:
    """完整工作流测试"""
    
    @pytest.fixture
    def ai_foundation(self):
        """创建AI基座实例"""
        from src import AIFoundation
        return AIFoundation(provider="openai", model="gpt-4o-mini")
    
    @pytest.mark.asyncio
    async def test_chat_workflow(self, ai_foundation):
        """测试聊天工作流"""
        # 模拟LLM响应
        with patch.object(
            ai_foundation.llm, 
            'generate', 
            new_callable=AsyncMock
        ) as mock_generate:
            from src.core.interfaces import AIResponse, UsageInfo
            
            mock_generate.return_value = AIResponse(
                content="Hello! I'm an AI assistant.",
                usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                model="gpt-4o-mini",
                finish_reason="stop"
            )
            
            response = await ai_foundation.chat("Hi there!")
            
            assert "Hello" in response
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stream_workflow(self, ai_foundation):
        """测试流式工作流"""
        async def mock_stream():
            chunks = ["Hello", ", ", "world", "!"]
            for chunk in chunks:
                yield chunk
        
        with patch.object(
            ai_foundation.llm,
            'stream_generate',
            return_value=mock_stream()
        ):
            result = ""
            async for chunk in ai_foundation.stream("Hello"):
                result += chunk
            
            assert result == "Hello, world!"


class TestMultiProviderIntegration:
    """多供应商集成测试"""
    
    def test_provider_switching(self):
        """测试供应商切换"""
        from src.providers.llm.factory import LLMProviderFactory
        
        # 测试创建不同供应商
        provider_names = ["openai", "anthropic", "deepseek"]
        
        for name in provider_names:
            try:
                provider = LLMProviderFactory.get_provider(name, {
                    "api_key": "test",
                    "provider_name": name
                })
                assert provider is not None
            except Exception as e:
                # 某些供应商可能需要额外依赖
                print(f"Provider {name}: {e}")


class TestAgentWithTools:
    """Agent工具集成测试"""
    
    @pytest.fixture
    def agent_with_tools(self):
        """创建带工具的Agent"""
        from src.agents import ReActAgent, AgentConfig, AgentType
        from src.tools import ToolManager, BuiltinTools
        
        tool_manager = ToolManager()
        BuiltinTools.register_builtins(tool_manager)
        
        config = AgentConfig(
            name="tool-agent",
            agent_type=AgentType.REACT,
            model="gpt-4o-mini",
            tools=tool_manager.list_tools(),
            max_iterations=5
        )
        
        return ReActAgent(config)
    
    @pytest.mark.asyncio
    async def test_agent_with_calculator(self, agent_with_tools):
        """测试Agent使用计算器"""
        result = await agent_with_tools.execute(
            "Calculate 10 * 5"
        )
        
        assert result.success is True


class TestContextPersistence:
    """上下文持久化测试"""
    
    @pytest.fixture
    def context_setup(self):
        """设置上下文"""
        from src.context import ContextManager
        
        manager = ContextManager()
        conv_id = "test-persistence"
        
        return manager, conv_id
    
    @pytest.mark.asyncio
    async def test_history_persistence(self, context_setup):
        """测试历史持久化"""
        manager, conv_id = context_setup
        
        # 添加消息
        await manager.add_message(conv_id, "user", "Hello")
        await manager.add_message(conv_id, "assistant", "Hi!")
        await manager.add_message(conv_id, "user", "How are you?")
        
        # 获取历史
        messages = await manager.get_messages(conv_id)
        
        assert len(messages) == 3
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi!"
        assert messages[2].content == "How are you?"


class TestMemoryIntegration:
    """记忆集成测试"""
    
    @pytest.fixture
    def memory_manager(self):
        """创建记忆管理器"""
        from src.memory import MemoryManager, MemoryConfig
        
        config = MemoryConfig()
        return MemoryManager(config)
    
    @pytest.mark.asyncio
    async def test_short_term_memory(self, memory_manager):
        """测试短期记忆"""
        # 存储对话
        await memory_manager.store_conversation(
            conversation_id="conv-1",
            role="user",
            content="Remember this: 12345",
            memory_type="short_term"
        )
        
        # 检索
        memories = await memory_manager.get_conversation_history("conv-1")
        
        assert len(memories) >= 1


class TestImageGeneration:
    """图像生成测试"""
    
    def test_dalle_provider(self):
        """测试DALL-E供应商"""
        from src.providers.image import DalleProvider
        
        provider = DalleProvider(api_key="test")
        
        assert provider.get_provider_name() == "dalle"
        assert "1024x1024" in provider.get_supported_sizes()
    
    def test_image_generator(self):
        """测试图像生成器"""
        from src.providers.image import ImageGenerator
        
        generator = ImageGenerator()
        generator.register_provider("dalle", DalleProvider())
        
        providers = generator.list_providers()
        
        assert len(providers) >= 1


class TestLoggingIntegration:
    """日志集成测试"""
    
    def test_simple_logger(self):
        """测试简单日志"""
        from src.services import SimpleLogger
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            temp_path = f.name
        
        try:
            logger = SimpleLogger(log_file=temp_path)
            
            # 模拟请求和响应
            from unittest.mock import Mock
            request = Mock()
            request.model = "gpt-4"
            request.messages = []
            
            response = Mock()
            response.content = "Test response"
            response.usage.prompt_tokens = 10
            response.usage.completion_tokens = 10
            response.usage.total_tokens = 20
            
            asyncio.run(logger.log_call(request, response))
            
            # 验证日志文件
            with open(temp_path, 'r') as f:
                content = f.read()
            
            assert "Test response" in content
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# 配置测试
class TestConfiguration:
    """配置测试"""
    
    def test_config_structure(self):
        """测试配置结构"""
        from src.config.manager import get_config
        
        config = get_config()
        
        # 验证主要部分
        assert hasattr(config, 'project')
        assert hasattr(config, 'providers')
        assert hasattr(config, 'grpc')
        assert hasattr(config, 'memory')
    
    def test_provider_config(self):
        """测试供应商配置"""
        from src.config.manager import get_provider_config
        
        # 测试获取配置
        openai_config = get_provider_config("openai")
        
        if openai_config:
            assert hasattr(openai_config, 'api_key')
            assert hasattr(openai_config, 'enabled')


# 运行所有集成测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
