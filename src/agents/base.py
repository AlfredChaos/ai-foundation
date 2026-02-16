# [Input] AgentConfig、模型名、Provider 工厂与配置管理依赖。
# [Output] 提供 Agent 基类能力（按配置精确解析 provider、请求构建、历史管理）。
# [Pos] agents 层基础抽象，供各类 Agent 复用通用行为。
"""
AI Foundation - Agent系统基础模块

本模块定义Agent框架的基础架构，
遵循SOLID原则，提供灵活的Agent扩展机制。

设计模式：
1. 模板方法模式 - BaseAgent定义执行骨架
2. 策略模式 - 支持不同类型的Agent
3. 组合模式 - Agent可组合工具和记忆

Agent类型：
- ConversationalAgent: 对话Agent
- ReActAgent: 思考-行动-观察Agent
- ToolAgent: 工具Agent
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from src.core.interfaces import IAgent, AIRequest, AIResponse, Message, MessageRole
from src.config.manager import get_provider_config
from src.providers.llm.factory import LLMProviderFactory, ILLMProvider
from src.tools.tool_manager import ToolManager


class AgentType(str, Enum):
    """
    Agent类型枚举
    
    定义支持的Agent类型：
    - CONVERSATIONAL: 基础对话Agent
    - REACT: ReAct模式Agent
    - TOOL: 工具Agent
    - PLANNING: 规划Agent
    - MULTI_AGENT: 多智能体协作
    """
    CONVERSATIONAL = "conversational"
    REACT = "react"
    TOOL = "tool"
    PLANNING = "planning"
    MULTI_AGENT = "multi_agent"


@dataclass
class AgentConfig:
    """
    Agent配置数据类
    
    Attributes:
        name: Agent名称
        agent_type: Agent类型
        system_prompt: 系统提示词
        model: 使用的模型
        temperature: 生成温度
        max_iterations: 最大迭代次数
        max_tokens: 最大输出Token数
        tools: 可用工具列表
        memory_enabled: 是否启用记忆
        human_in_loop: 是否启用人在回路
    """
    name: str
    agent_type: AgentType
    system_prompt: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_iterations: int = 10
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    memory_enabled: bool = True
    human_in_loop: bool = False


@dataclass
class AgentResult:
    """
    Agent执行结果数据类
    
    Attributes:
        success: 是否执行成功
        output: 执行输出内容
        intermediate_steps: 执行中间步骤列表
        tool_calls: 工具调用列表
        tokens_used: 使用的Token数
        error: 错误信息（如果失败）
    """
    success: bool
    output: str
    intermediate_steps: List[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = None
    tokens_used: int = 0
    error: Optional[str] = None


class BaseAgent(IAgent, ABC):
    """
    Agent基类 - 提供通用Agent功能
    
    遵循开闭原则，定义Agent的通用行为框架，
    具体Agent通过继承并实现抽象方法来实现特定逻辑。
    
    Attributes:
        config: Agent配置
        name: Agent名称
        agent_type: Agent类型
        _llm_provider: LLM供应商实例
        _tool_manager: 工具管理器实例
        _conversation_history: 对话历史
    
    Example:
        >>> class MyAgent(BaseAgent):
        ...     async def execute(self, task, **kwargs):
        ...         return AgentResult(success=True, output="Done")
        ...
        ...     async def plan(self, task):
        ...         return ["Step 1", "Step 2"]
        ...
        ...     async def evaluate(self, result):
        ...         return True
    """
    
    def __init__(self, config: AgentConfig):
        """
        初始化Agent
        
        Args:
            config: Agent配置对象
        """
        self.config = config
        self.name = config.name
        self.agent_type = config.agent_type
        self._llm_provider: Optional[ILLMProvider] = None
        self._tool_manager: Optional[ToolManager] = None
        self._conversation_history: List[Message] = []
    
    def _get_llm_provider(self) -> ILLMProvider:
        """
        获取LLM供应商实例
        
        懒加载方式创建供应商实例。
        
        Returns:
            ILLMProvider: LLM供应商实例
        """
        if self._llm_provider is None:
            provider_name = self._resolve_provider_name(self.config.model)
            self._llm_provider = LLMProviderFactory.create_from_config(provider_name)
        return self._llm_provider

    def _resolve_provider_name(self, model: str) -> str:
        """根据配置文件中的模型名精确解析 provider 名称"""
        normalized_model = model.strip().lower()
        if not normalized_model:
            raise ValueError("Model name cannot be empty")

        registered_providers = [name.lower() for name in LLMProviderFactory.list_providers()]
        if normalized_model in registered_providers:
            return normalized_model

        matches = []
        for provider_name in registered_providers:
            provider_config = get_provider_config(provider_name)
            if provider_config is None:
                continue

            configured_models = provider_config.models or {}
            for configured_model in configured_models.values():
                if not isinstance(configured_model, str):
                    continue
                if configured_model.strip().lower() == normalized_model:
                    matches.append(provider_name)
                    break

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            providers = ", ".join(sorted(matches))
            raise ValueError(
                f"Model '{model}' is configured in multiple providers: [{providers}]. "
                "Please specify provider explicitly."
            )

        raise ValueError(
            f"Unable to resolve provider for model '{model}' from configured providers. "
            "Please ensure it exists in `config/default.yaml` under `providers.*.models`."
        )
    
    def _set_llm_provider(self, provider: ILLMProvider) -> None:
        """
        设置LLM供应商实例
        
        支持手动注入，便于测试和替换。
        
        Args:
            provider: LLM供应商实例
        """
        self._llm_provider = provider
    
    def _get_tool_manager(self) -> ToolManager:
        """
        获取工具管理器实例
        
        懒加载方式创建，并注册配置中指定的工具。
        
        Returns:
            ToolManager: 工具管理器实例
        """
        if self._tool_manager is None:
            self._tool_manager = ToolManager()
            
            # 注册配置中的工具
            if self.config.tools:
                for tool in self.config.tools:
                    self._tool_manager.register_tool(
                        name=tool["name"],
                        func=tool["function"],
                        description=tool.get("description", ""),
                        parameters=tool.get("parameters", {})
                    )
        return self._tool_manager
    
    def _build_system_message(self) -> Message:
        """
        构建系统消息
        
        Returns:
            Message: 系统消息对象
        """
        return Message(
            role=MessageRole.SYSTEM,
            content=self.config.system_prompt or self._get_default_system_prompt()
        )
    
    def _get_default_system_prompt(self) -> str:
        """
        获取默认系统提示词
        
        Returns:
            str: 默认系统提示词
        """
        return f"""You are {self.name}, an AI assistant.

Your characteristics:
- Type: {self.agent_type.value}
- Purpose: {self._get_purpose()}
- Always be helpful and accurate.
- Think step by step before answering."""
    
    def _get_purpose(self) -> str:
        """
        获取Agent目的描述
        
        Returns:
            str: Agent目的描述
        """
        return "assist users with various tasks"
    
    def _create_request(
        self, 
        user_input: str, 
        additional_messages: Optional[List[Message]] = None
    ) -> AIRequest:
        """
        创建AI请求对象
        
        构建包含系统消息、历史消息和用户输入的完整请求。
        
        Args:
            user_input: 用户输入
            additional_messages: 可选，额外的消息列表
            
        Returns:
            AIRequest: 完整的AI请求对象
        """
        messages = [self._build_system_message()]
        
        # 添加对话历史
        messages.extend(self._conversation_history)
        
        # 添加额外的消息
        if additional_messages:
            messages.extend(additional_messages)
        
        # 添加用户输入
        messages.append(Message(role=MessageRole.USER, content=user_input))
        
        return AIRequest(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tools=self.config.tools,
        )
    
    def _update_history(self, role: MessageRole, content: str) -> None:
        """
        更新对话历史
        
        自动限制历史长度，防止Token超出限制。
        
        Args:
            role: 消息角色
            content: 消息内容
        """
        self._conversation_history.append(
            Message(role=role, content=content)
        )
        
        # 限制历史长度
        max_history = 20
        if len(self._conversation_history) > max_history:
            self._conversation_history = self._conversation_history[-max_history:]
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self._conversation_history = []
    
    def get_history(self) -> List[Message]:
        """
        获取对话历史副本
        
        Returns:
            List[Message]: 对话历史列表
        """
        return self._conversation_history.copy()
    
    @abstractmethod
    async def execute(self, task: str, **kwargs) -> AgentResult:
        """
        执行Agent任务 - 子类必须实现
        
        Args:
            task: 任务描述
            **kwargs: 额外参数
            
        Returns:
            AgentResult: 执行结果
        """
        pass
    
    @abstractmethod
    async def plan(self, task: str) -> List[str]:
        """
        制定执行计划 - 子类必须实现
        
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
        评估执行结果 - 子类必须实现
        
        判断当前结果是否满足要求。
        
        Args:
            result: 执行结果
            
        Returns:
            bool: 结果是否可接受
        """
        pass
    
    def get_agent_type(self) -> str:
        """
        获取Agent类型
        
        Returns:
            str: Agent类型字符串
        """
        return self.agent_type.value
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取Agent信息
        
        Returns:
            Dict包含Agent配置和状态信息
        """
        return {
            "name": self.name,
            "type": self.agent_type.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "tools_count": len(self.config.tools) if self.config.tools else 0,
            "history_length": len(self._conversation_history),
        }
