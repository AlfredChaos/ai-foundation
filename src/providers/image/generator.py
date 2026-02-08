# 图像生成模块
# 统一图像生成接口

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.core.interfaces import IImageProvider


@dataclass
class ImageConfig:
    """图像生成配置"""
    provider: str = "dalle"
    size: str = "1024x1024"
    quality: str = "standard"
    style: str = "vivid"


class BaseImageProvider(IImageProvider, ABC):
    """图像生成供应商基类"""
    
    def __init__(self, config: Optional[ImageConfig] = None):
        self.config = config or ImageConfig()
    
    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> str:
        """生成图像"""
        pass
    
    def _validate_prompt(self, prompt: str) -> bool:
        """验证提示词"""
        if not prompt or len(prompt) < 10:
            raise ValueError("Prompt too short (minimum 10 characters)")
        return True


class DalleProvider(BaseImageProvider):
    """DALL-E 图像生成"""
    
    def __init__(self, api_key: str = "", config: Optional[ImageConfig] = None):
        super().__init__(config or ImageConfig())
        self.api_key = api_key
    
    async def generate_image(self, prompt: str, **kwargs) -> str:
        """生成图像"""
        self._validate_prompt(prompt)
        
        # 这里调用OpenAI DALL-E API
        # 简化版本：返回占位符URL
        return f"https://placeholder.com/dalle/{len(prompt)}.jpg"
    
    def get_provider_name(self) -> str:
        """获取供应商名称"""
        return "dalle"
    
    def get_supported_sizes(self) -> List[str]:
        """获取支持的尺寸"""
        return ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]


class StableDiffusionProvider(BaseImageProvider):
    """Stable Diffusion 图像生成"""
    
    def __init__(self, api_url: str = "http://localhost:7860", 
                 config: Optional[ImageConfig] = None):
        super().__init__(config)
        self.api_url = api_url
    
    async def generate_image(self, prompt: str, **kwargs) -> str:
        """生成图像"""
        self._validate_prompt(prompt)
        
        # 这里调用Stable Diffusion API
        # 简化版本：返回占位符URL
        return f"https://placeholder.com/sd/{len(prompt)}.jpg"
    
    def get_provider_name(self) -> str:
        """获取供应商名称"""
        return "stable_diffusion"
    
    def get_supported_sizes(self) -> List[str]:
        """获取支持的尺寸"""
        return ["512x512", "768x768", "1024x1024"]


class ImageGenerator:
    """图像生成器 - 统一入口"""
    
    def __init__(self, config: Optional[ImageConfig] = None):
        self.config = config or ImageConfig()
        self._providers: Dict[str, IImageProvider] = {}
    
    def register_provider(self, name: str, provider: IImageProvider):
        """注册供应商"""
        self._providers[name] = provider
    
    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成图像"""
        provider_name = provider or self.config.provider
        
        if provider_name not in self._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        image_provider = self._providers[provider_name]
        
        # 获取尺寸
        size = kwargs.get("size", self.config.size)
        
        # 生成
        image_url = await image_provider.generate_image(prompt, **kwargs)
        
        return {
            "image_url": image_url,
            "provider": provider_name,
            "size": size,
            "prompt": prompt,
        }
    
    async def generate_multiple(
        self,
        prompt: str,
        count: int = 4,
        provider: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """生成多张图像"""
        provider_name = provider or self.config.provider
        
        if provider_name not in self._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        # 生成多张
        results = []
        for i in range(count):
            result = await self.generate(prompt, provider_name, **kwargs)
            results.append(result)
        
        return results
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """列出可用供应商"""
        return [
            {
                "name": name,
                "sizes": provider.get_supported_sizes(),
            }
            for name, provider in self._providers.items()
        ]


# 便捷函数
def create_image_generator(provider: str = "dalle", **kwargs) -> ImageGenerator:
    """创建图像生成器"""
    generator = ImageGenerator()
    
    if provider == "dalle":
        generator.register_provider("dalle", DalleProvider(kwargs.get("api_key")))
    elif provider == "stable_diffusion":
        generator.register_provider(
            "stable_diffusion", 
            StableDiffusionProvider(kwargs.get("api_url"))
        )
    
    return generator
