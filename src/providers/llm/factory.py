"""
AI Foundation - LLM供应商工厂模块

本模块实现LLM供应商的抽象工厂模式，
遵循开闭原则，支持灵活扩展新的供应商。

设计模式：
1. 工厂模式 - 统一创建供应商实例
2. 单例模式 - 工厂实例全局唯一
3. 策略模式 - 自由切换不同供应商

支持的供应商：
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- 智谱AI (ChatGLM)
- DeepSeek
- 豆包 (Doubao)
- Minimax
- OpenRouter
"""

from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod
import json

from src.core.interfaces import ILLMProvider, AIRequest, AIResponse
from src.core.abstracts.base_provider import BaseProvider
from src.config.manager import get_provider_config, get_model


class LLMProviderFactory:
    """
    LLM供应商工厂类
    
    采用单例模式，确保全局只有一个工厂实例。
    提供供应商注册、创建、查询等功能。
    
    Attributes:
        _instance: 单例实例
        _providers: 已注册的供应商类字典
        _provider_instances: 已创建的供应商实例字典
    
    Example:
        >>> factory = LLMProviderFactory()
        >>> provider = factory.get_provider("openai", config)
        >>> providers = factory.list_providers()
    """
    
    _instance: Optional['LLMProviderFactory'] = None
    _providers: Dict[str, Type[ILLMProvider]] = {}
    
    def __new__(cls):
        """
        单例模式的工厂实例创建
        
        Returns:
            LLMProviderFactory: 工厂唯一实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._provider_instances = {}
        return cls._instance
    
    @classmethod
    def register(cls, name: str, provider_class: Type[ILLMProvider]) -> None:
        """
        注册LLM供应商类
        
        供应商实现类通过此方法注册到工厂。
        注册后可以通过工厂创建实例。
        
        Args:
            name: 供应商名称（小写）
            provider_class: 供应商实现类
            
        Example:
            >>> class MyProvider(BaseLLMProvider):
            ...     pass
            >>> LLMProviderFactory.register("myprovider", MyProvider)
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def get_provider(cls, name: str, 
                   config: Optional[Dict[str, Any]] = None) -> ILLMProvider:
        """
        获取供应商实例
        
        如果实例已存在则返回缓存，否则创建新实例。
        
        Args:
            name: 供应商名称
            config: 可选，供应商配置字典
            
        Returns:
            ILLMProvider: 供应商实例
            
        Raises:
            ValueError: 未知供应商名称
            ImportError: 缺少必要的依赖包
        """
        name = name.lower()
        
        if name not in cls._providers:
            # 尝试使用通用OpenAI兼容接口
            if name in ["custom", "compatible"]:
                from .openai import OpenAIProvider
                provider_class = OpenAIProvider
            else:
                raise ValueError(
                    f"Unknown LLM provider: '{name}'. "
                    f"Available: {cls.list_providers()}"
                )
        else:
            provider_class = cls._providers[name]
        
        # 使用配置或默认配置
        if config is None:
            config = {}
        
        # 检查缓存
        cache_key = f"{name}_{hash(json.dumps(config, sort_keys=True))}"
        # 确保实例存在
        if cls._instance is None:
            cls()
        if cache_key in cls._instance._provider_instances:
            return cls._instance._provider_instances[cache_key]
        
        # 创建新实例
        instance = provider_class(config)
        cls._instance._provider_instances[cache_key] = instance
        
        return instance
    
    @classmethod
    def list_providers(cls) -> list:
        """
        列出所有已注册的供应商名称
        
        Returns:
            list: 供应商名称列表
        """
        return list(cls._providers.keys())
    
    @classmethod
    def create_from_config(cls, provider_name: str) -> ILLMProvider:
        """
        从配置文件创建供应商实例
        
        从config/default.yaml读取配置信息。
        
        Args:
            provider_name: 供应商名称
            
        Returns:
            ILLMProvider: 配置好的供应商实例
            
        Raises:
            ValueError: 配置不存在或供应商未启用
        """
        provider_config = get_provider_config(provider_name)
        
        if provider_config is None:
            raise ValueError(
                f"Provider '{provider_name}' not found in configuration"
            )
        
        if not provider_config.enabled:
            raise ValueError(
                f"Provider '{provider_name}' is disabled in configuration"
            )
        
        config_dict = {
            "api_key": provider_config.api_key,
            "base_url": provider_config.base_url,
            "models": provider_config.models,
            "provider_name": provider_name,
        }
        
        return cls.get_provider(provider_name, config_dict)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        获取当前可用的供应商列表
        
        检查每个已注册供应商的API密钥是否配置有效。
        
        Returns:
            list: 可用的供应商名称列表
        """
        available = []
        for name in cls.list_providers():
            try:
                provider = cls.create_from_config(name)
                if provider.is_available():
                    available.append(name)
            except (ValueError, ImportError):
                # 配置缺失或依赖缺失，跳过
                continue
        return available


class BaseLLMProvider(BaseProvider, ILLMProvider, ABC):
    """
    LLM供应商基类 - 提供通用功能
    
    所有具体供应商实现应继承此类。
    此类提供通用属性和方法，减少重复代码。
    
    Attributes:
        provider_name: 供应商名称
        api_key: API密钥
        base_url: API基础URL
        models: 可用模型字典
    
    Example:
        >>> class MyProvider(BaseLLMProvider):
        ...     async def generate(self, request):
        ...         pass
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化供应商
        
        Args:
            config: 配置字典，应包含api_key、base_url、models等
        """
        super().__init__(config)
        self.provider_name = config.get("provider_name", "unknown")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "")
        self.models = config.get("models", {})
    
    @abstractmethod
    async def generate(self, request: AIRequest) -> AIResponse:
        """
        生成回复 - 子类必须实现
        
        Args:
            request: AI请求对象
            
        Returns:
            AIResponse: AI响应对象
        """
        pass
    
    @abstractmethod
    async def stream_generate(self, request: AIRequest):
        """
        流式生成 - 子类必须实现
        
        Args:
            request: AI请求对象
            
        Returns:
            AsyncIterator[str]: 异步生成器
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        计算Token数 - 子类必须实现
        
        Args:
            text: 输入文本
            
        Returns:
            int: Token数量
        """
        pass
    
    def get_model(self, model_type: str = "default") -> str:
        """
        获取指定类型的模型名称
        
        Args:
            model_type: 模型类型（如'default'、'chat'、'reasoning'）
            
        Returns:
            str: 模型名称，未配置则返回空字符串
        """
        return self.models.get(model_type, "")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict包含:
            - provider: 供应商名称
            - models: 可用模型字典
            - available: 是否可用
        """
        return {
            "provider": self.provider_name,
            "models": self.models,
            "available": self.is_available(),
        }
    
    def is_available(self) -> bool:
        """
        检查供应商是否可用
        
        检查API密钥是否已配置。
        
        Returns:
            bool: API密钥有效返回True
        """
        return bool(self.api_key)


def register_provider(name: str):
    """
    装饰器：注册LLM供应商类
    
    使用此装饰器可以方便地注册供应商类。
    
    Args:
        name: 供应商名称（小写）
        
    Returns:
        decorator: 装饰器函数
        
    Example:
        >>> @register_provider("myprovider")
        ... class MyProvider(BaseLLMProvider):
        ...     pass
    """
    def decorator(cls):
        LLMProviderFactory.register(name, cls)
        return cls
    return decorator


def get_llm_provider(provider_name: str) -> ILLMProvider:
    """
    便捷函数：获取LLM供应商实例
    
    从配置创建供应商实例的快捷方式。
    
    Args:
        provider_name: 供应商名称
        
    Returns:
        ILLMProvider: 配置好的供应商实例
        
    Example:
        >>> provider = get_llm_provider("openai")
    """
    return LLMProviderFactory.create_from_config(provider_name)


def list_available_providers() -> list:
    """
    便捷函数：列出可用的供应商
    
    Returns:
        list: 当前可用的供应商名称列表
    """
    return LLMProviderFactory.get_available_providers()
