# 记忆模块接口

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass


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


@dataclass
class Memory:
    """记忆数据类"""
    key: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    memory_type: str = "short_term"
    embedding: Optional[List[float]] = None


class MemoryType:
    """记忆类型常量"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
