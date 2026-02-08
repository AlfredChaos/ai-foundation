# 记忆模块实现

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from src.memory.interfaces import IMemoryProvider, Memory


class InMemoryProvider(IMemoryProvider):
    """内存记忆存储 - 适用于短期记忆"""
    
    def __init__(self):
        self._storage: Dict[str, Memory] = {}
    
    async def store(self, key: str, content: Any, metadata: Optional[Dict] = None) -> bool:
        """存储记忆"""
        now = datetime.utcnow()
        
        memory = Memory(
            key=key,
            content=str(content),
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            memory_type="short_term",
        )
        
        self._storage[key] = memory
        return True
    
    async def retrieve(self, key: str) -> Optional[Memory]:
        """检索记忆"""
        return self._storage.get(key)
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """简单关键词搜索"""
        results = []
        query_lower = query.lower()
        
        for key, memory in self._storage.items():
            if query_lower in memory.content.lower():
                results.append({
                    "key": memory.key,
                    "content": memory.content,
                    "score": 1.0,
                    "metadata": memory.metadata,
                })
        
        return results[:top_k]
    
    async def delete(self, key: str) -> bool:
        """删除记忆"""
        if key in self._storage:
            del self._storage[key]
            return True
        return False
    
    async def clear(self, conversation_id: Optional[str] = None) -> bool:
        """清空记忆"""
        if conversation_id:
            # 只删除指定会话的记忆
            keys_to_delete = [
                k for k, m in self._storage.items()
                if m.metadata.get("conversation_id") == conversation_id
            ]
            for k in keys_to_delete:
                del self._storage[k]
        else:
            self._storage.clear()
        return True
    
    async def exists(self, key: str) -> bool:
        """检查记忆是否存在"""
        return key in self._storage
    
    def count(self) -> int:
        """获取记忆数量"""
        return len(self._storage)
    
    def get_all(self) -> List[Memory]:
        """获取所有记忆"""
        return list(self._storage.values())


class MongoDBProvider(IMemoryProvider):
    """MongoDB记忆存储 - 适用于长期记忆"""
    
    def __init__(self, connection_string: str, database: str = "ai_foundation"):
        self.connection_string = connection_string
        self.database_name = database
        self._client = None
        self._db = None
        self._collection = None
    
    async def _get_collection(self):
        """获取MongoDB集合"""
        if self._collection is None:
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
                self._client = AsyncIOMotorClient(self.connection_string)
                self._db = self._client[self.database_name]
                self._collection = self._db["memories"]
            except ImportError:
                raise ImportError("Please install motor: pip install motor")
        return self._collection
    
    async def store(self, key: str, content: Any, metadata: Optional[Dict] = None) -> bool:
        """存储记忆"""
        collection = await self._get_collection()
        now = datetime.utcnow()
        
        document = {
            "key": key,
            "content": str(content),
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "memory_type": "long_term",
        }
        
        await collection.update_one(
            {"key": key},
            {"$set": document},
            upsert=True
        )
        return True
    
    async def retrieve(self, key: str) -> Optional[Memory]:
        """检索记忆"""
        collection = await self._get_collection()
        
        doc = await collection.find_one({"key": key})
        
        if doc:
            return Memory(
                key=doc["key"],
                content=doc["content"],
                metadata=doc.get("metadata", {}),
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
                memory_type=doc.get("memory_type", "long_term"),
            )
        return None
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """文本搜索"""
        collection = await self._get_collection()
        
        cursor = collection.find({
            "$text": {"$search": query}
        }).limit(top_k)
        
        results = []
        async for doc in cursor:
            results.append({
                "key": doc["key"],
                "content": doc["content"],
                "score": doc.get("score", 0),
                "metadata": doc.get("metadata", {}),
            })
        
        return results
    
    async def delete(self, key: str) -> bool:
        """删除记忆"""
        collection = await self._get_collection()
        
        result = await collection.delete_one({"key": key})
        return result.deleted_count > 0
    
    async def clear(self, conversation_id: Optional[str] = None) -> bool:
        """清空记忆"""
        collection = await self._get_collection()
        
        if conversation_id:
            await collection.delete_many({"metadata.conversation_id": conversation_id})
        else:
            await collection.delete_many({})
        return True
    
    async def exists(self, key: str) -> bool:
        """检查记忆是否存在"""
        collection = await self._get_collection()
        
        doc = await collection.find_one({"key": key})
        return doc is not None
    
    async def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()


class RedisProvider(IMemoryProvider):
    """Redis记忆存储 - 适用于会话缓存"""
    
    def __init__(self, connection_string: str = "redis://localhost:6379", 
                 prefix: str = "memory:"):
        self.connection_string = connection_string
        self.prefix = prefix
        self._client = None
    
    async def _get_client(self):
        """获取Redis客户端"""
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.connection_string)
            except ImportError:
                raise ImportError("Please install redis: pip install redis")
        return self._client
    
    async def store(self, key: str, content: Any, metadata: Optional[Dict] = None) -> bool:
        """存储记忆"""
        client = await self._get_client()
        
        data = {
            "content": str(content),
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.utcnow().isoformat(),
        }
        
        await client.set(
            f"{self.prefix}{key}",
            json.dumps(data),
            ex=86400  # 24小时过期
        )
        return True
    
    async def retrieve(self, key: str) -> Optional[Memory]:
        """检索记忆"""
        client = await self._get_client()
        
        data = await client.get(f"{self.prefix}{key}")
        
        if data:
            parsed = json.loads(data)
            return Memory(
                key=key,
                content=parsed["content"],
                metadata=json.loads(parsed["metadata"]),
                created_at=datetime.fromisoformat(parsed["created_at"]),
                updated_at=datetime.fromisoformat(parsed["created_at"]),
                memory_type="short_term",
            )
        return None
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """简单搜索（Redis不擅长文本搜索，返回空）"""
        return []
    
    async def delete(self, key: str) -> bool:
        """删除记忆"""
        client = await self._get_client()
        
        result = await client.delete(f"{self.prefix}{key}")
        return result > 0
    
    async def clear(self, conversation_id: Optional[str] = None) -> bool:
        """清空记忆"""
        client = await self._get_client()
        
        pattern = f"{self.prefix}*"
        keys = await client.keys(pattern)
        
        if keys:
            await client.delete(*keys)
        return True
    
    async def exists(self, key: str) -> bool:
        """检查记忆是否存在"""
        client = await self._get_client()
        
        return await client.exists(f"{self.prefix}{key}") > 0
    
    async def close(self):
        """关闭连接"""
        if self._client:
            await self._client.close()
