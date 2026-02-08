# 基础抽象类
# 提供通用功能，减少重复代码

from abc import ABC
from typing import Any, Dict, Optional
from datetime import datetime
import logging


class BaseProvider(ABC):
    """所有供应商的基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def _get_config(self, key: str, default: Any = None) -> Any:
        """安全获取配置"""
        return self.config.get(key, default)
    
    def _validate_config(self, required_keys: list) -> bool:
        """验证必需配置"""
        for key in required_keys:
            if not self._get_config(key):
                self._logger.error(f"Missing required config: {key}")
                return False
        return True
    
    def get_provider_name(self) -> str:
        """获取供应商名称"""
        return self.__class__.__name__.lower().replace('provider', '')


class BaseService(ABC):
    """所有服务的基类"""
    
    def __init__(self, name: str):
        self.name = name
        self._logger = logging.getLogger(name)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """初始化服务"""
        self._initialized = True
        self._logger.info(f"Service {self.name} initialized")
        return True
    
    async def shutdown(self) -> None:
        """关闭服务"""
        self._initialized = False
        self._logger.info(f"Service {self.name} shutdown")
    
    def is_ready(self) -> bool:
        """检查是否就绪"""
        return self._initialized


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.requests = []
    
    async def acquire(self) -> bool:
        """获取许可"""
        from datetime import timedelta
        
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # 清理过期的请求记录
        self.requests = [t for t in self.requests if t > minute_ago]
        
        if len(self.requests) >= self.rpm:
            return False
        
        self.requests.append(now)
        return True
    
    async def wait_for_permit(self) -> None:
        """等待获得许可"""
        import asyncio
        while not await self.acquire():
            await asyncio.sleep(1)


class TokenBucket:
    """Token桶 - 用于API调用计数"""
    
    def __init__(self, capacity: int, refill_rate: int):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = datetime.utcnow()
    
    def consume(self, tokens: int = 1) -> bool:
        """消费Token"""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self) -> None:
        """补充Token"""
        from datetime import timedelta
        
        now = datetime.utcnow()
        seconds_passed = (now - self.last_refill).total_seconds()
        self.tokens = min(self.capacity, self.tokens + seconds_passed * self.refill_rate)
        self.last_refill = now


class AsyncCache:
    """异步缓存"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl  # seconds
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key not in self.cache:
            return None
        
        data, timestamp = self.cache[key]
        if datetime.utcnow().timestamp() - timestamp > self.ttl:
            del self.cache[key]
            return None
        
        return data
    
    async def set(self, key: str, value: Any) -> None:
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            # 简单的LRU: 删除最旧的条目
            oldest_key = min(self.cache.keys(), 
                            key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.utcnow().timestamp())
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()


class CircuitBreaker:
    """熔断器 - 防止级联故障"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure = None
        self.state = "closed"  # closed, open, half_open
    
    def record_success(self) -> None:
        """记录成功"""
        self.failures = 0
        self.state = "closed"
    
    def record_failure(self) -> None:
        """记录失败"""
        self.failures += 1
        self.last_failure = datetime.utcnow()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"
    
    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if (datetime.utcnow() - self.last_failure).total_seconds() > self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        
        return True  # half_open
    
    def reset(self) -> None:
        """重置熔断器"""
        self.failures = 0
        self.state = "closed"
