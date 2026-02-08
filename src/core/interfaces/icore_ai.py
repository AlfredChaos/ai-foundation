"""
AI Foundation - 核心接口定义模块

本模块定义了AI基座的所有核心接口(Interface)，
遵循接口隔离原则，确保模块间松耦合。
所有功能模块通过接口进行交互，便于扩展和维护。

设计原则：
1. 接口隔离 - 每个接口专注于单一职责
2. 依赖倒置 - 依赖抽象而非具体实现
3. 开闭原则 - 对扩展开放，对修改关闭

模块包含以下核心接口：
- ICoreAI: 核心AI功能入口
- ILLMProvider: LLM供应商接口
- IMemoryProvider: 记忆存储接口
- IContextManager: 上下文管理接口
- IToolManager: 工具管理接口
- IMCPClient: MCP客户端接口
- IAgent: Agent基础接口
- ILoggingService: 日志服务接口
- IImageProvider: 图像生成接口
- ITokenCounter: Token计数接口
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class MessageRole(str, Enum):
    """
    消息角色枚举
    
    定义对话中不同参与者的角色：
    - SYSTEM: 系统提示词，定义AI行为准则
    - USER: 用户输入
    - ASSISTANT: AI回复
    - TOOL: 工具调用结果
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """
    消息数据类
    
    Attributes:
        role: 消息角色，定义消息发送者类型
        content: 消息内容文本
        name: 可选，发送者名称
        tool_calls: 可选，工具调用列表
        tool_call_id: 可选，工具调用ID
        timestamp: 消息创建时间戳
    """
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AIRequest:
    """
    AI请求数据类
    
    用于封装AI调用的所有参数。
    
    Attributes:
        model: 使用的模型名称
        messages: 消息列表
        temperature: 生成温度(0.0-2.0)，越高越有创意
        max_tokens: 最大生成Token数
        stream: 是否使用流式输出
        json_mode: 是否强制JSON格式输出
        tools: 可选，工具定义列表
        extra_params: 可选，额外参数字典
    """
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
    """
    Token使用统计信息
    
    Attributes:
        prompt_tokens: 输入Token数
        completion_tokens: 输出Token数
        total_tokens: 总Token数
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class AIResponse:
    """
    AI响应数据类
    
    Attributes:
        content: AI生成的内容
        usage: Token使用统计
        model: 实际使用的模型
        finish_reason: 结束原因(stop/length/tool_calls等)
        tool_calls: 可选，工具调用列表
    """
    content: str
    usage: UsageInfo
    model: str
    finish_reason: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ICoreAI(ABC):
    """
    核心AI接口 - 所有AI功能的基础接口
    
    提供四种主要的AI调用模式：
    1. single_call: 单次调用，不维护状态
    2. chat: 多轮对话，维护对话状态
    3. stream: 流式输出，边生成边返回
    4. agent_execute: Agent模式，支持工具调用
    5. react_mode: ReAct模式，思考-行动-观察循环
    """
    
    @abstractmethod
    async def single_call(self, request: AIRequest) -> AIResponse:
        """
        单次调用 - 不维护对话状态的简单调用
        
        Args:
            request: AI请求对象，包含模型和消息
            
        Returns:
            AIResponse: AI生成的响应
            
        Example:
            >>> request = AIRequest(model="gpt-4", messages=[...])
            >>> response = await ai.single_call(request)
        """
        pass

    @abstractmethod
    async def chat(self, conversation_id: str, request: AIRequest) -> AIResponse:
        """
        Chat调用 - 维护对话状态的多轮对话
        
        Args:
            conversation_id: 对话会话ID，用于追踪对话历史
            request: AI请求对象
            
        Returns:
            AIResponse: AI生成的响应
            
        Example:
            >>> response = await ai.chat("user-123", request)
        """
        pass

    @abstractmethod
    async def stream(self, request: AIRequest) -> AsyncIterator[str]:
        """
        流式回复 - 返回生成器，支持流式输出
        
        Args:
            request: AI请求对象
            
        Returns:
            AsyncIterator[str]: 异步生成器，逐块返回生成的内容
            
        Example:
            >>> async for chunk in ai.stream(request):
            ...     print(chunk, end="", flush=True)
        """
        pass

    @abstractmethod
    async def agent_execute(self, task: str, 
                           tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Agent执行 - 执行带有工具调用的Agent
        
        Args:
            task: 要执行的任务描述
            tools: 可选，工具定义列表
            
        Returns:
            Dict包含执行结果和中间步骤
            
        Example:
            >>> result = await ai.agent_execute("查询天气", tools=[...])
        """
        pass

    @abstractmethod
    async def react_mode(self, task: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        ReAct模式 - 实现思考-行动-观察循环
        
        ReAct (Reasoning and Acting) 是一种Agent设计模式：
        1. 思考(Thought): 分析问题
        2. 行动(Action): 决定执行什么工具
        3. 观察(Observation): 获取工具返回结果
        重复直到任务完成
        
        Args:
            task: 要解决的任务
            max_iterations: 最大迭代次数，防止无限循环
            
        Returns:
            Dict包含最终答案和执行轨迹
        """
        pass


class ILLMProvider(ABC):
    """
    LLM供应商接口 - 所有LLM实现必须实现的接口
    
    采用策略模式，允许自由切换不同的LLM供应商。
    所有供应商实现必须实现以下方法。
    
    设计要点：
    - generate: 同步/异步生成回复
    - stream_generate: 流式生成
    - count_tokens: Token计数
    - get_model_info: 获取模型信息
    """
    
    @abstractmethod
    async def generate(self, request: AIRequest) -> AIResponse:
        """
        生成回复
        
        Args:
            request: AI请求对象
            
        Returns:
            AIResponse: 包含生成内容和Token使用统计
            
        Raises:
            ProviderError: 供应商调用失败时抛出
        """
        pass

    @abstractmethod
    async def stream_generate(self, request: AIRequest) -> AsyncIterator[str]:
        """
        流式生成
        
        Args:
            request: AI请求对象
            
        Returns:
            异步生成器，逐块返回内容
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        计算文本Token数
        
        Args:
            text: 输入文本
            
        Returns:
            int: Token数量
            
        Note:
            不同模型使用不同的编码器，结果可能不同
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict包含:
            - provider: 供应商名称
            - models: 可用模型列表
            - available: 是否可用
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查供应商是否可用
        
        Returns:
            bool: API密钥是否配置有效
        """
        pass


class IImageProvider(ABC):
    """
    图像生成供应商接口
    
    统一图像生成接口，支持不同后端(DALL-E, Stable Diffusion等)。
    """
    
    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> str:
        """
        生成图像
        
        Args:
            prompt: 图像描述提示词
            **kwargs: 额外参数(size, quality等)
            
        Returns:
            str: 图像URL或base64编码
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """获取供应商名称"""
        pass

    @abstractmethod
    def get_supported_sizes(self) -> List[str]:
        """
        获取支持的图像尺寸列表
        
        Returns:
            List[str]: 支持的尺寸列表，如["256x256", "512x512"]
        """
        pass


class IMemoryProvider(ABC):
    """
    记忆模块接口 - 支持短期和长期记忆
    
    提供统一的记忆存储和检索接口。
    实现类可以基于不同存储后端（内存、MongoDB、Redis等）。
    """
    
    @abstractmethod
    async def store(self, key: str, content: Any, 
                   metadata: Optional[Dict] = None) -> bool:
        """
        存储记忆
        
        Args:
            key: 记忆唯一标识符
            content: 要存储的内容
            metadata: 可选，元数据字典
            
        Returns:
            bool: 是否存储成功
        """
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """
        检索记忆
        
        Args:
            key: 记忆标识符
            
        Returns:
            存储的内容，不存在返回None
        """
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义搜索
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量上限
            
        Returns:
            List[Dict]: 匹配的记忆列表，按相关性排序
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除记忆"""
        pass

    @abstractmethod
    async def clear(self, conversation_id: Optional[str] = None) -> bool:
        """
        清空记忆
        
        Args:
            conversation_id: 可选，只清空指定会话的记忆
            
        Returns:
            bool: 是否清空成功
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查记忆是否存在"""
        pass


class IContextManager(ABC):
    """
    上下文管理接口 - 管理对话上下文和Token
    
    职责：
    1. 维护对话历史
    2. 计算Token数量
    3. 截断过长上下文
    4. 总结对话内容
    """
    
    @abstractmethod
    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """
        添加消息到上下文
        
        Args:
            conversation_id: 对话ID
            role: 消息角色
            content: 消息内容
        """
        pass

    @abstractmethod
    async def get_messages(self, conversation_id: str) -> List[Message]:
        """
        获取对话消息历史
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            List[Message]: 消息历史列表
        """
        pass

    @abstractmethod
    async def clear_context(self, conversation_id: str) -> None:
        """清空指定对话的上下文"""
        pass

    @abstractmethod
    async def summarize(self, conversation_id: str, max_tokens: int = 1000) -> str:
        """
        总结上下文
        
        使用AI对对话历史进行总结，减少Token消耗。
        
        Args:
            conversation_id: 对话ID
            max_tokens: 总结的最大Token数
            
        Returns:
            str: 总结后的内容
        """
        pass

    @abstractmethod
    def calculate_tokens(self, messages: List[Message]) -> int:
        """
        计算消息列表的Token数
        
        Args:
            messages: 消息列表
            
        Returns:
            int: 总Token数
        """
        pass

    @abstractmethod
    def truncate_context(self, messages: List[Message], max_tokens: int) -> List[Message]:
        """
        截断过长的上下文
        
        保留系统提示和最新的消息，截断中间部分。
        
        Args:
            messages: 原始消息列表
            max_tokens: 最大Token数
            
        Returns:
            List[Message]: 截断后的消息列表
        """
        pass


class IToolManager(ABC):
    """
    工具管理接口 - 管理工具注册和执行
    
    提供工具的注册、发现和执行功能。
    支持同步和异步工具。
    """
    
    @abstractmethod
    def register_tool(self, name: str, func: Callable, description: str,
                     parameters: Dict[str, Any]) -> None:
        """
        注册工具
        
        Args:
            name: 工具唯一名称
            func: 可调用的函数
            description: 工具功能描述
            parameters: JSON Schema格式的参数定义
        """
        pass

    @abstractmethod
    def unregister_tool(self, name: str) -> bool:
        """注销工具"""
        pass

    @abstractmethod
    async def execute_tool(self, name: str, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            Any: 工具执行结果
            
        Raises:
            ToolExecutionError: 工具执行失败
        """
        pass

    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        列出所有可用工具
        
        Returns:
            List[Dict]: 工具定义列表
        """
        pass

    @abstractmethod
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取工具信息
        
        Args:
            name: 工具名称
            
        Returns:
            Dict工具信息，不存在返回None
        """
        pass

    @abstractmethod
    async def execute_react(self, task: str, max_iterations: int) -> Dict[str, Any]:
        """
        执行ReAct循环
        
        与Agent配合使用，实现思考-行动-观察循环。
        """
        pass


class IMCPClient(ABC):
    """
    MCP客户端接口 - 连接和管理MCP服务器
    
    MCP (Model Context Protocol) 是一种工具协议标准。
    本接口定义与MCP服务器交互的方法。
    """
    
    @abstractmethod
    async def connect(self, server_url: str, 
                     api_key: Optional[str] = None) -> bool:
        """
        连接MCP服务器
        
        Args:
            server_url: MCP服务器URL (ws:// 或 http://)
            api_key: 可选，API密钥
            
        Returns:
            bool: 是否连接成功
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开与MCP服务器的连接"""
        pass

    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        列出MCP服务器上的工具
        
        Returns:
            List[Dict]: 可用工具列表
        """
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str, **params) -> Any:
        """
        调用MCP工具
        
        Args:
            tool_name: 工具名称
            **params: 工具参数
            
        Returns:
            Any: 工具返回结果
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass

    @abstractmethod
    def get_server_info(self) -> Dict[str, Any]:
        """
        获取服务器信息
        
        Returns:
            Dict包含服务器URL、连接状态、工具数量等
        """
        pass


class IAgent(ABC):
    """
    Agent接口 - 所有Agent实现的基础接口
    
    Agent是能够自主决策和执行任务的高级AI实体。
    本接口定义Agent的核心能力。
    """
    
    @abstractmethod
    async def execute(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        执行Agent任务
        
        Args:
            task: 任务描述
            **kwargs: 额外参数
            
        Returns:
            Dict包含执行结果和中间信息
        """
        pass

    @abstractmethod
    async def plan(self, task: str) -> List[str]:
        """
        制定执行计划
        
        将复杂任务分解为多个步骤。
        
        Args:
            task: 任务描述
            
        Returns:
            List[str]: 执行步骤列表
        """
        pass

    @abstractmethod
    async def evaluate(self, result: Any) -> bool:
        """
        评估执行结果
        
        判断当前结果是否满足要求。
        
        Args:
            result: 执行结果
            
        Returns:
            bool: 结果是否可接受
        """
        pass

    @abstractmethod
    def get_agent_type(self) -> str:
        """
        获取Agent类型
        
        Returns:
            str: Agent类型标识
        """
        pass


class ILoggingService(ABC):
    """
    日志服务接口 - 记录AI调用和监控指标
    
    支持Langfuse等可观测性平台集成。
    """
    
    @abstractmethod
    async def log_call(self, request: AIRequest, 
                      response: AIResponse, **kwargs) -> None:
        """
        记录API调用
        
        记录请求、响应、Token消耗等信息。
        
        Args:
            request: AI请求
            response: AI响应
            **kwargs: 额外信息
        """
        pass

    @abstractmethod
    async def log_error(self, error: Exception, 
                       context: Dict[str, Any]) -> None:
        """
        记录错误
        
        Args:
            error: 异常对象
            context: 错误上下文信息
        """
        pass

    @abstractmethod
    async def get_stats(self, start_time: datetime, 
                       end_time: datetime) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            start_time: 统计开始时间
            end_time: 统计结束时间
            
        Returns:
            Dict包含调用次数、Token消耗、错误率等
        """
        pass


class ITokenCounter(ABC):
    """
    Token计数接口 - 计算Token消耗和估算成本
    
    职责：
    1. 计算文本Token数
    2. 估算API调用成本
    3. 支持批量计数
    """
    
    @abstractmethod
    def count(self, text: str) -> int:
        """
        计算文本Token数
        
        Args:
            text: 输入文本
            
        Returns:
            int: Token数量
        """
        pass

    @abstractmethod
    def count_messages(self, messages: List[Message]) -> int:
        """
        计算消息列表Token数
        
        包含消息格式化的Token开销。
        
        Args:
            messages: 消息列表
            
        Returns:
            int: 总Token数
        """
        pass

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int,
                     model: str) -> float:
        """
        估算API调用成本
        
        基于各模型的定价计算预估费用。
        
        Args:
            prompt_tokens: 输入Token数
            completion_tokens: 输出Token数
            model: 模型名称
            
        Returns:
            float: 预估成本（美元）
        """
        pass
