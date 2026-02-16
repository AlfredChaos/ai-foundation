# [Input] 对话 ID、消息列表与 LLM token 计数依赖。
# [Output] 提供上下文增删查、截断、总结与统计信息。
# [Pos] context 层管理器实现，统一维护会话上下文状态。

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.core.interfaces import IContextManager, Message, MessageRole, UsageInfo
from src.providers.llm.factory import LLMProviderFactory, ILLMProvider


@dataclass
class ContextConfig:
    """上下文配置"""
    max_messages: int = 20
    max_tokens: int = 8000
    summarization_model: str = "gpt-4o-mini"
    preserve_system_prompt: bool = True


class MemoryContextStore:
    """内存上下文存储"""
    
    def __init__(self):
        self._contexts: Dict[str, List[Message]] = {}
    
    async def get(self, conversation_id: str) -> List[Message]:
        """获取上下文"""
        return self._contexts.get(conversation_id, [])
    
    async def set(self, conversation_id: str, messages: List[Message]) -> None:
        """设置上下文"""
        self._contexts[conversation_id] = messages
    
    async def add(self, conversation_id: str, message: Message) -> None:
        """添加消息"""
        if conversation_id not in self._contexts:
            self._contexts[conversation_id] = []
        self._contexts[conversation_id].append(message)
    
    async def clear(self, conversation_id: str) -> None:
        """清空上下文"""
        if conversation_id in self._contexts:
            del self._contexts[conversation_id]
    
    async def exists(self, conversation_id: str) -> bool:
        """检查上下文是否存在"""
        return conversation_id in self._contexts


class ContextManager(IContextManager):
    """上下文管理器"""
    
    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
        self._store = MemoryContextStore()
        self._token_counter: Optional[ILLMProvider] = None
    
    def _get_token_counter(self) -> ILLMProvider:
        """获取Token计数器"""
        if self._token_counter is None:
            self._token_counter = LLMProviderFactory.get_provider("openai")
        return self._token_counter
    
    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """添加消息到上下文"""
        message_role = MessageRole(role) if isinstance(role, str) else role
        
        message = Message(
            role=message_role,
            content=content,
            timestamp=datetime.utcnow(),
        )
        
        await self._store.add(conversation_id, message)
        
        # 检查是否需要截断
        messages = await self._store.get(conversation_id)
        if len(messages) > self.config.max_messages:
            # 保留系统提示（如果配置）
            if self.config.preserve_system_prompt:
                system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
                other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]
                # 保留最近的 max_messages - len(system_messages) 条
                other_messages = other_messages[-(self.config.max_messages - len(system_messages)):]
                messages = system_messages + other_messages
            else:
                messages = messages[-self.config.max_messages:]
            
            await self._store.set(conversation_id, messages)
    
    async def get_messages(self, conversation_id: str) -> List[Message]:
        """获取对话消息历史"""
        return await self._store.get(conversation_id)
    
    async def clear_context(self, conversation_id: str) -> None:
        """清空上下文"""
        await self._store.clear(conversation_id)
    
    async def summarize(self, conversation_id: str, max_tokens: int = 1000) -> str:
        """总结上下文"""
        messages = await self._store.get(conversation_id)
        
        if not messages:
            return ""
        
        # 提取消息内容
        contents = [f"{m.role.value}: {m.content}" for m in messages]
        full_text = "\n\n".join(contents)
        
        # 如果已经很短，直接返回
        if len(full_text) < max_tokens * 4:  # 假设平均1 token = 4字符
            return full_text
        
        # 创建总结请求
        from src.core.interfaces import AIRequest, Message as CoreMessage
        
        summary_prompt = f"""Summarize this conversation in {max_tokens} tokens or less. Focus on:
1. Main topics discussed
2. Key decisions made
3. Important information to remember

Conversation:
{{content}}"""

        try:
            llm = self._get_token_counter()
            request = AIRequest(
                model=self.config.summarization_model,
                messages=[
                    CoreMessage(role=MessageRole.SYSTEM, content=summary_prompt.format(content=full_text[:10000])),
                ],
                max_tokens=max_tokens,
            )
            
            response = await llm.generate(request)
            return response.content
            
        except Exception as e:
            # 如果总结失败，返回摘要
            return f"Conversation with {len(messages)} messages. (Summary unavailable: {str(e)})"
    
    def calculate_tokens(self, messages: List[Message]) -> int:
        """计算Token数"""
        try:
            llm = self._get_token_counter()
            text = "\n".join([f"{m.role.value}: {m.content}" for m in messages])
            return llm.count_tokens(text)
        except Exception:
            # 估算
            return sum(len(m.content) for m in messages) // 4
    
    def truncate_context(self, messages: List[Message], max_tokens: int) -> List[Message]:
        """截断过长的上下文"""
        if not messages:
            return messages
        
        # 计算当前Token数
        current_tokens = self.calculate_tokens(messages)
        
        if current_tokens <= max_tokens:
            return messages
        
        # 保留系统提示和最近的对话
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]
        
        # 计算系统消息的Token
        system_tokens = self.calculate_tokens(system_messages)
        
        # 保留的空间
        remaining_tokens = max_tokens - system_tokens
        
        # 从后向前添加消息，直到达到限制
        truncated = system_messages
        current_tokens = system_tokens
        
        for message in reversed(other_messages):
            message_tokens = self.calculate_tokens([message])
            if current_tokens + message_tokens <= max_tokens:
                truncated.insert(len(system_messages), message)
                current_tokens += message_tokens
            else:
                break
        
        return truncated
    
    def _build_context_info(self, conversation_id: str, messages: List[Message]) -> Dict[str, Any]:
        """基于消息列表构建上下文统计信息"""
        token_count = self.calculate_tokens(messages)
        return {
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "token_count": token_count,
            "is_truncated": len(messages) >= self.config.max_messages or token_count >= self.config.max_tokens,
            "oldest_message": messages[0].timestamp.isoformat() if messages else None,
            "latest_message": messages[-1].timestamp.isoformat() if messages else None,
        }

    async def get_context_info(self, conversation_id: str) -> Dict[str, Any]:
        """异步获取上下文信息（可在运行中的事件循环内安全调用）"""
        messages = await self._store.get(conversation_id)
        return self._build_context_info(conversation_id, messages)

    def get_context_info_sync(self, conversation_id: str) -> Dict[str, Any]:
        """同步获取上下文信息，仅用于非异步调用场景"""
        import asyncio

        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Event loop is already running. Use `await get_context_info(...)` in async context."
            )
        except RuntimeError as error:
            if "already running" in str(error):
                raise
            return asyncio.run(self.get_context_info(conversation_id))
    
    async def copy_context(self, source_id: str, target_id: str) -> None:
        """复制上下文"""
        messages = await self._store.get(source_id)
        await self._store.set(target_id, messages.copy())
