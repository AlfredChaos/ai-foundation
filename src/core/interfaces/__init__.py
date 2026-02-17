# AI基座核心接口定义
# 所有模块交互通过接口进行，确保解耦和可扩展性

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class MessageRole(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """消息类"""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AIRequest:
    """AI请求基类"""
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    json_mode: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class UsageInfo:
    """Token使用信息"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class AIResponse:
    """AI响应基类"""
    content: str
    usage: UsageInfo
    model: str
    finish_reason: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None  # 思考模型返回的推理过程


class ICoreAI(ABC):
    """核心AI接口 - 所有AI功能的基础接口"""

    @abstractmethod
    async def single_call(self, request: AIRequest) -> AIResponse:
        """单次调用 - 不维护对话状态的简单调用"""
        pass

    @abstractmethod
    async def chat(self, conversation_id: str, request: AIRequest) -> AIResponse:
        """Chat调用 - 维护对话状态的多轮对话"""
        pass

    @abstractmethod
    async def stream(self, request: AIRequest) -> AsyncIterator[str]:
        """流式回复 - 返回生成器，支持流式输出"""
        pass

    @abstractmethod
    async def agent_execute(self, task: str, tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Agent执行 - 执行带有工具调用的Agent"""
        pass

    @abstractmethod
    async def react_mode(self, task: str, max_iterations: int = 10) -> Dict[str, Any]:
        """ReAct模式 - 实现思考-行动-观察循环"""
        pass


class ILLMProvider(ABC):
    """LLM供应商接口 - 所有LLM实现必须实现的接口"""

    @abstractmethod
    async def generate(self, request: AIRequest) -> AIResponse:
        """生成回复"""
        pass

    @abstractmethod
    async def stream_generate(self, request: AIRequest) -> AsyncIterator[str]:
        """流式生成"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算Token数"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查供应商是否可用"""
        pass


class IImageProvider(ABC):
    """图像生成供应商接口"""

    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> str:
        """生成图像，返回图像URL或base64"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """获取供应商名称"""
        pass

    @abstractmethod
    def get_supported_sizes(self) -> List[str]:
        """获取支持的图像尺寸"""
        pass


class IMemoryProvider(ABC):
    """记忆模块接口 - 支持短期和长期记忆"""

    @abstractmethod
    async def store(self, key: str, content: Any, metadata: Optional[Dict] = None) -> bool:
        """存储记忆"""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """检索记忆"""
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """语义搜索"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除记忆"""
        pass

    @abstractmethod
    async def clear(self, conversation_id: Optional[str] = None) -> bool:
        """清空记忆"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查记忆是否存在"""
        pass


class IContextManager(ABC):
    """上下文管理接口 - 管理对话上下文和Token"""

    @abstractmethod
    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """添加消息到上下文"""
        pass

    @abstractmethod
    async def get_messages(self, conversation_id: str) -> List[Message]:
        """获取对话消息历史"""
        pass

    @abstractmethod
    async def clear_context(self, conversation_id: str) -> None:
        """清空上下文"""
        pass

    @abstractmethod
    async def summarize(self, conversation_id: str, max_tokens: int = 1000) -> str:
        """总结上下文"""
        pass

    @abstractmethod
    def calculate_tokens(self, messages: List[Message]) -> int:
        """计算Token数"""
        pass

    @abstractmethod
    def truncate_context(self, messages: List[Message], max_tokens: int) -> List[Message]:
        """截断过长的上下文"""
        pass


class IToolManager(ABC):
    """工具管理接口 - 管理工具注册和执行"""

    @abstractmethod
    def register_tool(self, name: str, func: callable, description: str, 
                     parameters: Dict[str, Any]) -> None:
        """注册工具"""
        pass

    @abstractmethod
    def unregister_tool(self, name: str) -> bool:
        """注销工具"""
        pass

    @abstractmethod
    async def execute_tool(self, name: str, **kwargs) -> Any:
        """执行工具"""
        pass

    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        pass

    @abstractmethod
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具信息"""
        pass

    @abstractmethod
    async def execute_react(self, task: str, max_iterations: int) -> Dict[str, Any]:
        """执行ReAct循环"""
        pass


class IMCPClient(ABC):
    """MCP客户端接口 - 连接和管理MCP服务器"""

    @abstractmethod
    async def connect(self, server_url: str, api_key: Optional[str] = None) -> bool:
        """连接MCP服务器"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出MCP服务器上的工具"""
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str, **params) -> Any:
        """调用MCP工具"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass

    @abstractmethod
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        pass


class IAgent(ABC):
    """Agent接口 - 所有Agent实现的基础接口"""

    @abstractmethod
    async def execute(self, task: str, **kwargs) -> Dict[str, Any]:
        """执行Agent任务"""
        pass

    @abstractmethod
    async def plan(self, task: str) -> List[str]:
        """制定执行计划"""
        pass

    @abstractmethod
    async def evaluate(self, result: Any) -> bool:
        """评估执行结果"""
        pass

    @abstractmethod
    def get_agent_type(self) -> str:
        """获取Agent类型"""
        pass


class ILoggingService(ABC):
    """日志服务接口"""

    @abstractmethod
    async def log_call(self, request: AIRequest, response: AIResponse, **kwargs) -> None:
        """记录API调用"""
        pass

    @abstractmethod
    async def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """记录错误"""
        pass

    @abstractmethod
    async def get_stats(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """获取统计信息"""
        pass


class ITokenCounter(ABC):
    """Token计数接口"""

    @abstractmethod
    def count(self, text: str) -> int:
        """计算文本Token数"""
        pass

    @abstractmethod
    def count_messages(self, messages: List[Message]) -> int:
        """计算消息列表Token数"""
        pass

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, 
                     model: str) -> float:
        """估算成本"""
        pass
