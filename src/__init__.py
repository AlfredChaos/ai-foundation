# AI Foundation 主包导出
# 提供简洁的API入口

# 版本
__version__ = "1.0.0"

# 核心组件
from src.config.manager import get_config, get_provider_config, get_model
from src.providers.llm import (
    LLMProviderFactory,
    get_llm_provider,
    list_available_providers,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    ZhipuProvider,
    DeepSeekProvider,
    DoubaoProvider,
    MinimaxProvider,
    OpenRouterProvider,
)

# Agent组件
from src.agents import (
    BaseAgent,
    AgentConfig,
    AgentResult,
    AgentType,
    ReActAgent,
    SimpleReActAgent,
    ConversationalAgent,
)

# 工具组件
from src.tools import (
    ToolManager,
    ToolInfo,
    BaseTool,
    BuiltinTools,
    MCPClient,
    MCPClientManager,
    MCPServerConfig,
)

# 上下文组件
from src.context import (
    ContextManager,
    ContextConfig,
)

# 记忆组件
from src.memory import (
    MemoryManager,
    MemoryConfig,
    Memory,
    InMemoryProvider,
    MongoDBProvider,
    RedisProvider,
)

# 服务组件
from src.services import (
    LoggingService,
    LangfuseLogger,
    TokenCounter,
    TokenConfig,
    HumanInLoop,
    HumanAction,
    HumanReview,
    ApprovalManager,
)

# 图像生成
from src.providers.image import (
    ImageGenerator,
    ImageConfig,
    DalleProvider,
    StableDiffusionProvider,
)

# gRPC服务
from src.grpc_service.server import serve, AICoreServicer

__all__ = [
    # 版本
    "__version__",
    # 配置
    "get_config",
    "get_provider_config",
    "get_model",
    # LLM供应商
    "LLMProviderFactory",
    "get_llm_provider",
    "list_available_providers",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "ZhipuProvider",
    "DeepSeekProvider",
    "DoubaoProvider",
    "MinimaxProvider",
    "OpenRouterProvider",
    # Agent
    "BaseAgent",
    "AgentConfig",
    "AgentResult",
    "AgentType",
    "ReActAgent",
    "SimpleReActAgent",
    "ConversationalAgent",
    # 工具
    "ToolManager",
    "ToolInfo",
    "BaseTool",
    "BuiltinTools",
    "MCPClient",
    "MCPClientManager",
    "MCPServerConfig",
    # 上下文
    "ContextManager",
    "ContextConfig",
    # 记忆
    "MemoryManager",
    "MemoryConfig",
    "Memory",
    "InMemoryProvider",
    "MongoDBProvider",
    "RedisProvider",
    # 服务
    "LoggingService",
    "LangfuseLogger",
    "TokenCounter",
    "TokenConfig",
    "HumanInLoop",
    "HumanAction",
    "HumanReview",
    "ApprovalManager",
    # 图像
    "ImageGenerator",
    "ImageConfig",
    "DalleProvider",
    "StableDiffusionProvider",
    # gRPC
    "serve",
    "AICoreServicer",
]


class AIFoundation:
    """AI基座主类 - 提供统一的API入口"""
    
    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider
        self.model = model
        self._llm = None
        self._agent = None
        self._tool_manager = None
        self._context_manager = None
    
    @property
    def llm(self):
        """获取LLM供应商"""
        if self._llm is None:
            self._llm = get_llm_provider(self.provider)
            if self.model:
                pass
        return self._llm
    
    @property
    def agent(self):
        """获取Agent"""
        if self._agent is None:
            self._agent = ReActAgent(AgentConfig(
                name="ai-foundation-agent",
                agent_type=AgentType.REACT,
                model=self.model or "gpt-4o",
            ))
        return self._agent
    
    @property
    def tools(self):
        """获取工具管理器"""
        if self._tool_manager is None:
            self._tool_manager = ToolManager()
            BuiltinTools.register_builtins(self._tool_manager)
        return self._tool_manager
    
    @property
    def context(self):
        """获取上下文管理器"""
        if self._context_manager is None:
            self._context_manager = ContextManager()
        return self._context_manager
    
    async def chat(self, message: str, conversation_id: str = "default") -> str:
        """简单对话"""
        from src.core.interfaces import AIRequest, Message, MessageRole
        
        messages = [Message(role=MessageRole.USER, content=message)]
        
        request = AIRequest(
            model=self.model or "gpt-4o",
            messages=messages,
        )
        
        response = await self.llm.generate(request)
        return response.content
    
    async def stream(self, message: str):
        """流式对话"""
        from src.core.interfaces import AIRequest, Message, MessageRole
        
        messages = [Message(role=MessageRole.USER, content=message)]
        
        request = AIRequest(
            model=self.model or "gpt-4o",
            messages=messages,
            stream=True,
        )
        
        async for chunk in self.llm.stream_generate(request):
            yield chunk
    
    async def agent_execute(self, task: str) -> dict:
        """执行Agent任务"""
        return await self.agent.execute(task)


def create_ai(provider: str = "openai", model: str = None) -> AIFoundation:
    """创建AI基座实例"""
    return AIFoundation(provider=provider, model=model)


def quick_chat(message: str, provider: str = "openai", model: str = None) -> str:
    """快速对话 - 简单的单次调用"""
    ai = create_ai(provider, model)
    return asyncio.run(ai.chat(message))


import asyncio


def example_usage():
    """示例用法"""
    ai = create_ai(provider="openai", model="gpt-4o")
    
    async def demo():
        response = await ai.chat("Hello, how are you?")
        print(f"Response: {response}")
    
    asyncio.run(demo())


if __name__ == "__main__":
    example_usage()
