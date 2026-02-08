# 配置管理器
# 统一管理所有配置文件，支持环境变量覆盖

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """供应商配置"""
    enabled: bool = False
    api_key: str = ""
    base_url: str = ""
    models: Dict[str, str] = field(default_factory=dict)


class LLMConfig(BaseModel):
    """LLM配置"""
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    google: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    doubao: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)


class ImageConfig(BaseModel):
    """图像生成配置"""
    dalle: Dict[str, Any] = Field(default_factory=dict)
    stable_diffusion: Dict[str, Any] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    """记忆模块配置"""
    short_term: Dict[str, Any] = Field(default_factory=dict)
    long_term: Dict[str, Any] = Field(default_factory=dict)


class LangfuseConfig(BaseModel):
    """Langfuse配置"""
    enabled: bool = False
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    timeout: int = 10


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class MongoDBConfig(BaseModel):
    """MongoDB配置（可选）"""
    enabled: bool = False
    connection_string: str = ""
    database: str = "ai_foundation"
    collection: str = "ai_logs"


class GrpcConfig(BaseModel):
    """gRPC配置"""
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10


class ToolsConfig(BaseModel):
    """工具配置"""
    mcp: Dict[str, Any] = Field(default_factory=dict)
    builtins: Dict[str, Any] = Field(default_factory=dict)


class ProjectConfig(BaseModel):
    """项目配置"""
    name: str = "ai-foundation"
    version: str = "1.0.0"
    environment: str = "development"


class AppConfig(BaseModel):
    """应用配置"""
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    providers: LLMConfig = Field(default_factory=LLMConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    mongodb_logging: MongoDBConfig = Field(default_factory=MongoDBConfig)
    grpc: GrpcConfig = Field(default_factory=GrpcConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


class ConfigManager:
    """配置管理器"""
    
    _instance: Optional['ConfigManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = None
            cls._instance._config_path = None
        return cls._instance
    
    def __init__(self):
        pass
    
    def load(self, config_path: Optional[str] = None) -> AppConfig:
        """加载配置"""
        if self._config is not None:
            return self._config
        
        # 确定配置文件路径
        if config_path is None:
            config_path = self._find_config_file()
        
        self._config_path = config_path
        
        # 加载YAML配置
        raw_config = self._load_yaml(config_path)
        
        # 应用环境变量覆盖
        raw_config = self._apply_env_overrides(raw_config)
        
        # 创建配置对象
        self._config = AppConfig(**raw_config)
        
        return self._config
    
    def _find_config_file(self) -> str:
        """查找配置文件"""
        possible_paths = [
            "config/default.yaml",
            "config.yaml",
            "/opt/ai-foundation/config/default.yaml",
            os.path.expanduser("~/.ai-foundation/config.yaml"),
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(Path(path).absolute())
        
        # 返回默认路径
        return possible_paths[0]
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """加载YAML文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        env_mappings = {
            "OPENAI_API_KEY": ("providers", "openai", "api_key"),
            "ANTHROPIC_API_KEY": ("providers", "anthropic", "api_key"),
            "GOOGLE_API_KEY": ("providers", "google", "api_key"),
            "ZHIPU_API_KEY": ("providers", "zhipu", "api_key"),
            "DEEPSEEK_API_KEY": ("providers", "deepseek", "api_key"),
            "DOUBAN_API_KEY": ("providers", "doubao", "api_key"),
            "MINIMAX_API_KEY": ("providers", "minimax", "api_key"),
            "OPENROUTER_API_KEY": ("providers", "openrouter", "api_key"),
            "DALLE_API_KEY": ("image", "dalle", "api_key"),
            "MONGODB_URI": ("mongodb_logging", "connection_string"),
            "LANGFUSE_PUBLIC_KEY": ("langfuse", "public_key"),
            "LANGFUSE_SECRET_KEY": ("langfuse", "secret_key"),
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested(config, path, value)
        
        return config
    
    def _set_nested(self, d: Dict, path: tuple, value: Any) -> None:
        """设置嵌套字典值"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """获取配置值"""
        if self._config is None:
            self.load()
        
        result = self._config
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key)
            elif hasattr(result, key):
                result = getattr(result, key)
            else:
                return default
            if result is None:
                return default
        return result
    
    def get_provider(self, provider_name: str) -> Optional[ProviderConfig]:
        """获取特定供应商配置"""
        providers = self.get("providers")
        if providers is None:
            return None
        
        return getattr(providers, provider_name, None)
    
    def get_model(self, provider_name: str, model_type: str = "default") -> str:
        """获取特定供应商的模型"""
        provider = self.get_provider(provider_name)
        if provider is None:
            return ""
        
        models = provider.models or {}
        return models.get(model_type, "")
    
    def is_provider_enabled(self, provider_name: str) -> bool:
        """检查供应商是否启用"""
        provider = self.get_provider(provider_name)
        if provider is None:
            return False
        return provider.enabled
    
    def reload(self) -> AppConfig:
        """重新加载配置"""
        self._config = None
        return self.load()


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """获取配置"""
    return config_manager.load()


def get_provider_config(provider_name: str) -> Optional[ProviderConfig]:
    """获取供应商配置"""
    return config_manager.get_provider(provider_name)


def get_model(provider_name: str, model_type: str = "default") -> str:
    """获取模型名称"""
    return config_manager.get_model(provider_name, model_type)
