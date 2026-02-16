# [Input] AgentConfig、对话任务文本与 ContextManager/LLM provider 依赖。
# [Output] 提供可维护上下文的多轮对话 Agent 执行结果。
# [Pos] agents 层对话 Agent 实现，负责会话历史管理与总结触发。

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.agents.base import BaseAgent, AgentConfig, AgentResult, AgentType
from src.core.interfaces import AIRequest, Message, MessageRole
from src.context.manager import ContextManager


@dataclass
class ConversationConfig:
    """对话配置"""
    max_history: int = 10
    max_tokens_per_message: int = 4000
    enable_summarization: bool = False
    summarization_threshold: int = 20
    system_prompt: str = ""


class ConversationalAgent(BaseAgent):
    """对话Agent - 维护对话上下文的多轮对话"""
    
    def __init__(self, config: AgentConfig, conversation_config: Optional[ConversationConfig] = None):
        super().__init__(config)
        self.agent_type = AgentType.CONVERSATIONAL
        self.conversation_config = conversation_config or ConversationConfig()
        self._context_manager = ContextManager()
    
    def _get_default_system_prompt(self) -> str:
        """获取对话Agent默认系统提示"""
        return f"""You are {self.name}, a helpful and friendly AI assistant.

## Your Characteristics
- You engage in natural, conversational dialogue
- You remember context from the conversation
- You provide clear and helpful responses
- You ask clarifying questions when needed
- You are honest about limitations

## Guidelines
- Be conversational but professional
- Keep responses concise and relevant
- Remember important details from the conversation
- Adapt your communication style to the user"""
    
    async def execute(self, task: str, **kwargs) -> AgentResult:
        """执行对话"""
        try:
            conversation_id = kwargs.get('conversation_id', self.name)
            
            # 获取对话历史
            history = await self._context_manager.get_messages(conversation_id)
            
            # 构建消息
            messages = [self._build_system_message()]
            messages.extend(history)
            messages.append(Message(role=MessageRole.USER, content=task))
            
            # 计算Token
            token_count = self._context_manager.calculate_tokens(messages)
            
            # 如果需要截断
            if token_count > self.conversation_config.max_tokens_per_message * 3:
                messages = self._context_manager.truncate_context(
                    messages, 
                    self.conversation_config.max_tokens_per_message * 2
                )
            
            # 创建请求
            request = AIRequest(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            # 调用LLM
            llm = self._get_llm_provider()
            response = await llm.generate(request)
            
            # 添加到历史
            await self._context_manager.add_message(conversation_id, "user", task)
            await self._context_manager.add_message(
                conversation_id, 
                "assistant", 
                response.content
            )
            
            # 检查是否需要总结
            if (
                self.conversation_config.enable_summarization and
                len(history) > self.conversation_config.summarization_threshold
            ):
                summary = await self._summarize_conversation(conversation_id)
                if summary:
                    # 替换旧的历史
                    await self._context_manager.clear_context(conversation_id)
                    await self._context_manager.add_message(
                        conversation_id, 
                        "system", 
                        f"Conversation Summary: {summary}"
                    )
            
            return AgentResult(
                success=True,
                output=response.content,
                tokens_used=response.usage.total_tokens,
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                error=str(e),
            )
    
    async def plan(self, task: str) -> List[str]:
        """制定对话计划"""
        return [
            f"Understand the user's intent: {task}",
            "Retrieve relevant context",
            "Formulate response",
            "Consider follow-up topics",
        ]
    
    async def evaluate(self, result: Any) -> bool:
        """评估对话结果"""
        if isinstance(result, AgentResult):
            return result.success and len(result.output) > 0
        return False
    
    async def _summarize_conversation(self, conversation_id: str) -> str | None:
        """总结对话"""
        messages = await self._context_manager.get_messages(conversation_id)
        
        # 提取主要内容
        contents = [m.content for m in messages if m.role != MessageRole.SYSTEM]
        full_text = "\n".join(contents[:10])  # 只取前10条
        
        # 创建总结请求
        summary_request = AIRequest(
            model=self.config.model,
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content="Summarize this conversation concisely in 200 words or less."
                ),
                Message(role=MessageRole.USER, content=full_text),
            ],
            max_tokens=300,
        )
        
        try:
            llm = self._get_llm_provider()
            response = await llm.generate(summary_request)
            return response.content
        except Exception:
            return None
    
    async def clear_conversation(self, conversation_id: str) -> None:
        """清空对话"""
        await self._context_manager.clear_context(conversation_id)
    
    def _get_purpose(self) -> str:
        """获取Agent目的"""
        return "engage in natural multi-turn conversations"
