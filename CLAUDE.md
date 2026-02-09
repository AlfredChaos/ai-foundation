# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 安装依赖（首次使用或依赖更新时）
pip install -e ".[dev]"

# 运行所有测试
pytest tests/
# 或使用脚本
./scripts/run_tests.sh

# 运行单个测试文件
pytest tests/unit/test_providers.py -v

# 运行特定测试
pytest tests/unit/test_providers.py::test_openai_provider -v

# 代码质量检查（格式化 + 类型检查）
./scripts/code_quality_check.py

# 手动格式化
black src/ tests/ --line-length 100
isort src/ tests/ --profile black

# 类型检查
mypy src/

# Linting
ruff check src/ tests/

# gRPC Proto 编译（修改 .proto 文件后需要）
python -m grpc_tools.protoc -I=src/grpc_service --python_out=src/grpc_service --grpc_python_out=src/grpc_service src/grpc_service/ai_core.proto

# 启动 gRPC 服务
python -m src.grpc_service.server

# 运行示例
python examples/basic_usage.py
```

## 架构概览

本项目是一个基于 Python + LangChain 的 AI 基座框架，采用分层架构设计：

### 分层结构

```
Application Layer (gRPC Service, SDK, CLI)
         ↓
Interface Layer (ICoreAI, ILLMProvider, IMemoryProvider 等)
         ↓
Service Layer (LLM Service, Agent Service, Memory Service, Tool Service)
         ↓
Provider Layer (LLM Providers, Image Providers - 插件式架构)
         ↓
Infrastructure Layer (Langfuse, MongoDB, Config, Utilities)
```

### 核心设计原则

**严格遵循 SOLID 原则**：
- **S**: 每个模块单一职责
- **O**: 通过抽象类和接口实现扩展开放、修改关闭
- **L**: 所有 Provider 子类可替换父类
- **I**: 接口隔离（如 ILLMProvider、IMemoryProvider、IToolManager 等）
- **D**: 依赖抽象接口，具体实现通过工厂模式注入

**设计模式应用**：
- **抽象工厂模式**: `LLMProviderFactory` 创建不同供应商实例
- **策略模式**: 切换不同 LLM/Image 供应商
- **观察者模式**: 人在回路 (HumanInLoop) 功能
- **装饰器模式**: Token 计数、日志服务
- **代理模式**: gRPC 远程调用

### 目录结构说明

- `src/core/` - 核心接口定义 ([interfaces/](src/core/interfaces/)) 和抽象类 ([abstracts/](src/core/abstracts/))
- `src/providers/` - LLM/Image 供应商实现，通过工厂模式创建
- `src/agents/` - Agent 系统（ReAct、Conversational、自定义）
- `src/tools/` - 工具管理、MCP 客户端、Skills 系统
- `src/memory/` - 记忆模块（短期内存、长期记忆、语义搜索）
- `src/context/` - 上下文管理和 Token 计数
- `src/services/` - 日志、监控、人在回路等服务
- `src/grpc_service/` - gRPC 服务端和客户端
- `src/config/` - 配置管理（YAML + 环境变量）

### 主要入口点

```python
# 简单使用
from src import create_ai

ai = create_ai(provider="openai", model="gpt-4o")
response = await ai.chat("你好")
```

```python
# 直接使用 Provider
from src.providers.llm import get_llm_provider
from src.core.interfaces import AIRequest, Message, MessageRole

provider = get_llm_provider("openai")
request = AIRequest(
    model="gpt-4o",
    messages=[Message(role=MessageRole.USER, content="Hello")]
)
response = await provider.generate(request)
```

## 开发注意事项

### 异步编程
- **全局使用 async/await**，所有 Provider 和 Agent 方法都是异步的
- 使用 `pytest-asyncio` 进行测试（已配置 `asyncio_mode = "auto"`）

### 配置管理
- 配置文件位于 `config/default.yaml`
- 环境变量优先级高于配置文件
- API 密钥通过环境变量注入（如 `OPENAI_API_KEY`）

### 添加新 LLM 供应商
1. 继承 `ILLMProvider` 接口
2. 实现 `generate()`, `stream_generate()`, `count_tokens()`, `get_model_info()` 方法
3. 在 `LLMProviderFactory` 中注册
4. 在 `config/default.yaml` 添加配置模板

### Agent 系统
- `ReActAgent`: 思维-行动-观察循环
- `ConversationalAgent`: 带记忆的对话
- 自定义 Agent 继承 `BaseAgent` 并实现 `execute()`, `plan()`, `evaluate()` 方法

### 错误处理
- 自定义异常继承自 `AIError`
- 常见异常：`ProviderError`, `TokenLimitExceededError`, `ToolExecutionError`

### 代码风格
- Black (line-length: 100)
- 中文注释，英文日志和错误信息
- 类型提示：使用 Pydantic 模型和类型注解

## 依赖说明

- **LangChain**: 核心 LLM 集成框架
- **Pydantic**: 配置验证和数据模型
- **grpcio**: gRPC 服务
- **motor/redis**: 异步存储驱动
- **langfuse**: 可观测性监控
- **mcp**: Model Context Protocol 支持
