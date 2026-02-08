# 记忆管理服务

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.memory.interfaces import IMemoryProvider, Memory, MemoryType
from src.memory.providers import InMemoryProvider, MongoDBProvider, RedisProvider


@dataclass
class MemoryConfig:
    """记忆配置"""
    short_term_provider: str = "memory"  # memory, redis
    long_term_provider: str = "mongodb"  # mongodb
    max_short_term_memories: int = 100
    max_conversation_history: int = 20


class MemoryManager:
    """记忆管理器 - 统一管理短期和长期记忆"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._short_term: Optional[IMemoryProvider] = None
        self._long_term: Optional[IMemoryProvider] = None
    
    @property
    def short_term(self) -> IMemoryProvider:
        """获取短期记忆"""
        if self._short_term is None:
            if self.config.short_term_provider == "redis":
                self._short_term = RedisProvider()
            else:
                self._short_term = InMemoryProvider()
        return self._short_term
    
    @property
    def long_term(self) -> IMemoryProvider:
        """获取长期记忆"""
        if self._long_term is None:
            if self.config.long_term_provider == "mongodb":
                # 需要配置连接字符串
                self._long_term = MongoDBProvider("mongodb://localhost:27017")
            else:
                self._long_term = InMemoryProvider()
        return self._long_term
    
    async def store_conversation(
        self,
        conversation_id: str,
        role: str,
        content: str,
        memory_type: str = "short_term"
    ) -> bool:
        """存储对话"""
        key = f"conv:{conversation_id}:{role}:{datetime.utcnow().timestamp()}"
        
        provider = self.short_term if memory_type == "short_term" else self.long_term
        
        return await provider.store(
            key=key,
            content=content,
            metadata={
                "conversation_id": conversation_id,
                "role": role,
                "memory_type": memory_type,
            }
        )
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """获取对话历史"""
        memories = await self.short_term.search(
            query=conversation_id,
            top_k=limit
        )
        
        # 按时间排序
        memories.sort(key=lambda x: x.get("metadata", {}).get("created_at", ""), reverse=True)
        
        return memories
    
    async def store_memory(
        self,
        key: str,
        content: Any,
        memory_type: str = "long_term",
        metadata: Optional[Dict] = None
    ) -> bool:
        """存储记忆"""
        provider = self.long_term if memory_type == "long_term" else self.short_term
        
        return await provider.store(
            key=key,
            content=content,
            metadata=metadata
        )
    
    async def retrieve_memory(self, key: str) -> Optional[Memory]:
        """检索记忆"""
        # 先查短期，再查长期
        memory = await self.short_term.retrieve(key)
        if memory:
            return memory
        
        return await self.long_term.retrieve(key)
    
    async def search_memories(
        self,
        query: str,
        memory_type: str = "long_term",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索记忆"""
        provider = self.long_term if memory_type == "long_term" else self.short_term
        return await provider.search(query, top_k)
    
    async def delete_memory(self, key: str) -> bool:
        """删除记忆"""
        deleted = await self.short_term.delete(key)
        if not deleted:
            deleted = await self.long_term.delete(key)
        return deleted
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """清空对话"""
        await self.short_term.clear(conversation_id)
        await self.long_term.clear(conversation_id)
        return True
    
    async def clear_all(self) -> bool:
        """清空所有记忆"""
        await self.short_term.clear()
        await self.long_term.clear()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "short_term_count": (
                self._short_term.count() 
                if hasattr(self._short_term, 'count') else "N/A"
            ),
            "long_term_type": self.config.long_term_provider,
        }


from datetime import datetime
