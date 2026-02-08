# Langfuse集成
# 可观测性平台集成

import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.core.interfaces import ILoggingService, AIRequest, AIResponse


@dataclass
class LangfuseConfig:
    """Langfuse配置"""
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    timeout: int = 10
    enabled: bool = True


class LangfuseLogger(ILoggingService):
    """Langfuse日志服务"""
    
    def __init__(self, config: Optional[LangfuseConfig] = None):
        self.config = config or LangfuseConfig()
        self._client = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """初始化Langfuse"""
        if not self.config.enabled:
            return False
        
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.host,
            )
            self._initialized = True
            return True
        except ImportError:
            print("Langfuse not installed. Run: pip install langfuse")
            return False
        except Exception as e:
            print(f"Langfuse initialization error: {e}")
            return False
    
    async def log_call(
        self,
        request: AIRequest,
        response: AIResponse,
        **kwargs
    ) -> None:
        """记录API调用"""
        if not self._initialized:
            return
        
        try:
            generation = self._client.generation(
                name=kwargs.get("name", "ai_call"),
                input=self._format_messages(request.messages),
                output=response.content,
                metadata={
                    "model": request.model,
                    "temperature": request.temperature,
                    "tokens": {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens,
                    },
                    "finish_reason": response.finish_reason,
                }
            )
            
            generation.end()
            
        except Exception as e:
            print(f"Langfuse logging error: {e}")
    
    async def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """记录错误"""
        if not self._initialized:
            return
        
        try:
            self._client.trace(
                name="error",
                input=str(context.get("input", "")),
                output=str(error),
                metadata=context,
            )
        except Exception as e:
            print(f"Langfuse error logging error: {e}")
    
    async def get_stats(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """获取统计信息"""
        # Langfuse API调用获取统计数据
        return {
            "total_calls": 0,
            "total_tokens": 0,
            "average_latency": 0,
        }
    
    def _format_messages(self, messages: List) -> str:
        """格式化消息"""
        return "\n".join([
            f"{m.role}: {m.content[:100]}..."
            for m in messages
        ])
    
    async def flush(self) -> None:
        """刷新日志"""
        if self._client:
            self._client.flush()


class SimpleLogger(ILoggingService):
    """简单日志服务 - 不依赖Langfuse"""
    
    def __init__(self, log_file: str = "ai_calls.log"):
        self.log_file = log_file
    
    async def log_call(
        self,
        request: AIRequest,
        response: AIResponse,
        **kwargs
    ) -> None:
        """记录API调用到文件"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": request.model,
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            },
            "input": self._format_messages(request.messages),
            "output": response.content[:200] + "...",
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    async def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """记录错误"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "error",
            "error": str(error),
            "context": context,
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    async def get_stats(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """从日志文件获取统计"""
        stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "errors": 0,
        }
        
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("type") == "error":
                        stats["errors"] += 1
                    else:
                        stats["total_calls"] += 1
                        stats["total_tokens"] += entry.get("tokens", {}).get("total", 0)
        except FileNotFoundError:
            pass
        
        return stats
    
    def _format_messages(self, messages: List) -> str:
        """格式化消息"""
        return "\n".join([
            f"{m.role}: {m.content[:100]}"
            for m in messages
        ])


class LoggingService:
    """日志服务管理器"""
    
    def __init__(self, config: Optional[LangfuseConfig] = None):
        self._logger = None
        self._config = config
    
    async def initialize(self) -> bool:
        """初始化日志服务"""
        # 尝试使用Langfuse
        self._logger = LangfuseLogger(self._config)
        success = await self._logger.initialize()
        
        if not success:
            # 回退到简单日志
            self._logger = SimpleLogger()
        
        return True
    
    async def log_call(
        self,
        request: AIRequest,
        response: AIResponse,
        **kwargs
    ) -> None:
        """记录API调用"""
        if self._logger:
            await self._logger.log_call(request, response, **kwargs)
    
    async def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """记录错误"""
        if self._logger:
            await self._logger.log_error(error, context)
    
    async def get_stats(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """获取统计"""
        if self._logger:
            return await self._logger.get_stats(start_time, end_time)
        return {}


import json
