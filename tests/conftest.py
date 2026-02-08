# conftest.py - pytest配置

import pytest
import asyncio
import sys
from pathlib import Path


# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config):
    """Pytest配置"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM响应"""
    from src.core.interfaces import AIResponse, UsageInfo
    
    return AIResponse(
        content="This is a test response",
        usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        model="gpt-4o-mini",
        finish_reason="stop",
        tool_calls=None
    )


@pytest.fixture
def sample_messages():
    """示例消息"""
    from src.core.interfaces import Message, MessageRole
    
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello!"),
    ]


@pytest.fixture
def sample_config():
    """示例配置"""
    return {
        "api_key": "test-api-key",
        "base_url": "https://api.openai.com/v1",
        "models": {
            "default": "gpt-4o-mini",
            "chat": "gpt-4o-mini",
        }
    }
