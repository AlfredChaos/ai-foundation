<!-- [Input] 项目架构信息与目录结构说明。 -->
<!-- [Output] 描述 AI Foundation 的系统架构与实施规划。 -->
<!-- [Pos] docs 层架构设计文档。 -->
# AI基座架构设计文档

## 1. 项目概述

### 1.1 项目目标
建立一个基于Python+LangChain的AI基座，提供快速集成AI能力的开发框架。

### 1.2 核心功能
- AI单次调用
- Chat调用
- Agent实现
- ReAct模式
- AI生图
- 工具调用
- 多轮对话
- 人在回路
- MCP支持
- Skills支持
- 多智能体聊天
- 上下文管理
- 记忆模块
- Token计数
- 流式回复
- JSON格式输出

### 1.3 支持的模型供应商
- OpenAI
- OpenAI兼容接口
- Google Gemini
- 智谱ZAI
- DeepSeek
- 豆包
- Minimax
- Anthropic
- OpenRouter
- 及其他兼容接口

---

## 2. 架构设计原则

### 2.1 SOLID原则
本项目严格遵循SOLID原则：

1. **S**ingle Responsibility (单一职责) - 每个类/模块只有一个改变的理由
2. **O**pen/Closed (开闭原则) - 对扩展开放，对修改关闭
3. **L**iskov Substitution (里氏替换) - 子类可以替换父类
4. **I**nterface Segregation (接口隔离) - 多个专用接口优于一个通用接口
5. **D**ependency Inversion (依赖倒置) - 依赖于抽象而非具体

### 2.2 设计模式应用
- **抽象工厂模式** - 用于创建不同类型的AI提供者
- **策略模式** - 用于切换不同的模型供应商
- **观察者模式** - 用于实现人在回路功能
- **装饰器模式** - 用于添加额外功能（如Token计数）
- **代理模式** - 用于实现gRPC远程调用

---

## 3. 系统架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │ gRPC Service│ │  SDK/Package│ │  CLI Tool   │ │  Web API   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Interface Layer                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    ICoreAI Interface                        │ │
│  │  - single_call()  - chat()  - stream()  - agent()          │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Service Layer                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │ LLM Service │ │ Agent Service│ │ Memory     │ │ Context    │ │
│  │              │ │              │ │ Service    │ │ Service    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │ Tool Service│ │ Image Service│ │ Logging    │ │ Token      │ │
│  │              │ │              │ │ Service    │ │ Service    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Provider Layer (Plugins)                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      LLM Providers                            │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │ │
│  │  │ OpenAI │ │ Gemini │ │DeepSeek│ │  智谱  │ │  豆包  │   │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Image Providers                            │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐                           │ │
│  │  │DALL-E  │ │Stable  │ │Midjourney(API)                    │ │
│  │  └────────┘ └────────┘ └────────┘                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │  Langfuse   │ │  MongoDB    │ │  Config     │ │  Utilities │ │
│  │  (Observabil│ │  (Logging)  │ │  Manager    │ │             │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心接口设计

#### 3.2.1 ICoreAI (核心AI接口)

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class AIRequest:
    """AI请求基类"""
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    json_mode: bool = False
    tools: Optional[List[Dict]] = None
    extra_params: Optional[Dict[str, Any]] = None

@dataclass
class AIResponse:
    """AI响应基类"""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str
    tool_calls: Optional[List[Dict]] = None

class ICoreAI(ABC):
    """核心AI接口"""

    @abstractmethod
    async def single_call(self, request: AIRequest) -> AIResponse:
        """单次调用"""
        pass

    @abstractmethod
    async def chat(self, conversation_id: str, request: AIRequest) -> AIResponse:
        """Chat调用"""
        pass

    @abstractmethod
    async def stream(self, request: AIRequest) -> AsyncIterator[str]:
        """流式回复"""
        pass

    @abstractmethod
    async def agent_execute(self, task: str, tools: List[Dict]) -> Dict[str, Any]:
        """Agent执行"""
        pass

    @abstractmethod
    async def react_mode(self, task: str, max_iterations: int = 10) -> Dict[str, Any]:
        """ReAct模式"""
        pass
```

#### 3.2.2 ILLMProvider (LLM供应商接口)

```python
class ILLMProvider(ABC):
    """LLM供应商接口"""

    @abstractmethod
    async def generate(self, request: AIRequest) -> AIResponse:
        """生成回复"""
        pass

    @abstractmethod
    async def stream_generate(self, request: AIRequest) -> AsyncIterator[str]:
        """流式生成"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算Token数"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass
```

#### 3.2.3 IMemoryProvider (记忆模块接口)

```python
class IMemoryProvider(ABC):
    """记忆模块接口"""

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
```

#### 3.2.4 IContextManager (上下文管理接口)

```python
class IContextManager(ABC):
    """上下文管理接口"""

    @abstractmethod
    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """添加消息"""
        pass

    @abstractmethod
    async def get_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        """获取消息历史"""
        pass

    @abstractmethod
    async def clear_context(self, conversation_id: str) -> None:
        """清空上下文"""
        pass

    @abstractmethod
    async def summarize(self, conversation_id: str, max_tokens: int = 1000) -> str:
        """总结上下文"""
        pass

    @abstractmethod
    def calculate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """计算Token"""
        pass
```

#### 3.2.5 IToolManager (工具管理接口)

```python
class IToolManager(ABC):
    """工具管理接口"""

    @abstractmethod
    def register_tool(self, name: str, func: callable, description: str) -> None:
        """注册工具"""
        pass

    @abstractmethod
    def unregister_tool(self, name: str) -> bool:
        """注销工具"""
        pass

    @abstractmethod
    async def execute_tool(self, name: str, **kwargs) -> Any:
        """执行工具"""
        pass

    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        pass

    @abstractmethod
    async def execute_react(self, task: str, max_iterations: int) -> Dict[str, Any]:
        """执行ReAct"""
        pass
```

#### 3.2.6 IMCPClient (MCP客户端接口)

```python
class IMCPClient(ABC):
    """MCP客户端接口"""

    @abstractmethod
    async def connect(self, server_url: str) -> bool:
        """连接MCP服务器"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出MCP工具"""
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str, **params) -> Any:
        """调用MCP工具"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass
```

---

## 4. 模块设计

### 4.1 配置管理模块 (Config Manager)

**职责**：
- 统一管理所有配置文件
- 提供配置热加载功能
- 支持多环境配置（开发、测试、生产）

**设计**：
- 使用Pydantic进行配置验证
- 支持YAML/JSON配置文件
- 环境变量覆盖支持

### 4.2 LLM服务模块 (LLM Service)

**职责**：
- 统一LLM调用入口
- 实现负载均衡和故障转移
- Token计数和成本统计

**设计**：
- 抽象工厂模式创建LLM Provider
- 装饰器模式添加日志和统计功能

### 4.3 Agent服务模块 (Agent Service)

**职责**：
- 提供标准Agent实现
- 支持自定义Agent行为
- 实现ReAct模式

**设计**：
- 策略模式切换不同Agent类型
- 观察者模式支持人在回路

### 4.4 记忆模块 (Memory Module)

**职责**：
- 短期记忆管理（对话历史）
- 长期记忆管理（语义记忆）
- 记忆检索和总结

**设计**：
- 接口隔离，不同存储后端实现不同Provider
- 支持MongoDB、Redis、文件等存储

### 4.5 上下文管理模块 (Context Manager)

**职责**：
- 维护对话上下文
- Token计算和截断
- 上下文总结

**设计**：
- 责任链模式处理不同长度的上下文

### 4.6 工具管理模块 (Tool Manager)

**职责**：
- 工具注册和发现
- 工具执行
- 工具编排

**设计**：
- 注册表模式管理工具

### 4.7 MCP支持模块 (MCP Support)

**职责**：
- MCP协议实现
- MCP服务器连接管理
- 工具映射和调用

**设计**：
- 适配器模式对接MCP协议

### 4.8 图像生成模块 (Image Generation)

**职责**：
- 统一图像生成接口
- 多供应商支持（DALL-E、Stable Diffusion等）

**设计**：
- 策略模式切换不同供应商

### 4.9 日志和监控模块 (Logging & Monitoring)

**职责**：
- 调用日志记录
- Token消耗统计
- Langfuse集成

**设计**：
- 中介者模式统一管理

---

## 5. gRPC服务设计

### 5.1 Proto定义

```protobuf
service AICoreService {
    // 单次调用
    rpc SingleCall(SingleCallRequest) returns (SingleCallResponse);

    // Chat对话
    rpc Chat(ChatRequest) returns (ChatResponse);

    // 流式回复
    rpc StreamCall(StreamRequest) returns (stream StreamResponse);

    // Agent执行
    rpc ExecuteAgent(AgentRequest) returns (AgentResponse);

    // ReAct模式
    rpc ExecuteReAct(ReActRequest) returns (ReActResponse);

    // 图像生成
    rpc GenerateImage(ImageRequest) returns (ImageResponse);

    // 工具调用
    rpc CallTool(ToolCallRequest) returns (ToolCallResponse);

    // 记忆管理
    rpc StoreMemory(MemoryRequest) returns (MemoryResponse);
    rpc RetrieveMemory(MemoryQueryRequest) returns (MemoryQueryResponse);
}
```

### 5.2 消息定义

```protobuf
message SingleCallRequest {
    string model = 1;
    repeated Message messages = 2;
    float temperature = 3;
    int32 max_tokens = 4;
    bool json_mode = 5;
    map<string, string> extra_params = 6;
}

message Message {
    string role = 1;      // system, user, assistant, tool
    string content = 2;
    string name = 3;
}

message SingleCallResponse {
    string content = 1;
    Usage usage = 2;
    string model = 3;
    string finish_reason = 4;
    repeated ToolCall tool_calls = 5;
}

message Usage {
    int32 prompt_tokens = 1;
    int32 completion_tokens = 2;
    int32 total_tokens = 3;
}
```

---

## 6. 目录结构

```
ai-foundation/
├── src/
│   ├── core/
│   │   ├── interfaces/
│   │   │   ├── __init__.py
│   │   │   ├── icore_ai.py          # 核心AI接口
│   │   │   ├── illm_provider.py     # LLM供应商接口
│   │   │   ├── imemory_provider.py  # 记忆模块接口
│   │   │   ├── icontext_manager.py  # 上下文管理接口
│   │   │   ├── itool_manager.py     # 工具管理接口
│   │   │   ├── imcp_client.py        # MCP客户端接口
│   │   │   └── iimage_provider.py    # 图像生成接口
│   │   ├── abstracts/
│   │   │   ├── __init__.py
│   │   │   ├── base_ai.py           # 基础AI抽象类
│   │   │   ├── base_provider.py     # 基础供应商抽象类
│   │   │   └── base_agent.py        # 基础Agent抽象类
│   │   └── __init__.py
│   │
│   ├── providers/
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── openai.py            # OpenAI实现
│   │   │   ├── anthropic.py         # Anthropic实现
│   │   │   ├── google.py            # Google Gemini实现
│   │   │   ├── zhipu.py             # 智谱ZAI实现
│   │   │   ├── deepseek.py          # DeepSeek实现
│   │   │   ├── doubao.py            # 豆包实现
│   │   │   ├── minimax.py           # Minimax实现
│   │   │   ├── openrouter.py        # OpenRouter实现
│   │   │   └── factory.py           # LLMProvider工厂
│   │   ├── image/
│   │   │   ├── __init__.py
│   │   │   ├── dalle.py             # DALL-E实现
│   │   │   ├── stable_diffusion.py  # Stable Diffusion实现
│   │   │   └── factory.py           # ImageProvider工厂
│   │   └── tool/
│   │       ├── __init__.py
│   │       └── builtins.py          # 内置工具实现
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # Agent基类
│   │   ├── react_agent.py          # ReAct Agent
│   │   ├── conversational_agent.py  # 对话Agent
│   │   ├── tool_agent.py           # 工具Agent
│   │   └── multi_agent.py          # 多智能体系统
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py           # 短期记忆
│   │   ├── long_term.py            # 长期记忆
│   │   ├── semantic.py             # 语义记忆
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── memory_provider.py  # 记忆存储接口
│   │       ├── mongo_provider.py   # MongoDB实现
│   │       └── redis_provider.py   # Redis实现
│   │
│   ├── context/
│   │   ├── __init__.py
│   │   └── manager.py              # 上下文管理器
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── mcp/
│   │   │   ├── __init__.py
│   │   │   ├── client.py           # MCP客户端
│   │   │   ├── protocol.py         # MCP协议实现
│   │   │   └── tools.py            # MCP工具适配器
│   │   └── skills/
│   │       ├── __init__.py
│   │       ├── base.py             # Skill基类
│   │       └── registry.py         # Skill注册表
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py          # LLM服务
│   │   ├── agent_service.py        # Agent服务
│   │   ├── token_service.py        # Token计数服务
│   │   ├── logging_service.py      # 日志服务
│   │   └── human_in_loop.py        # 人在回路服务
│   │
│   ├── grpc_service/
│   │   ├── __init__.py
│   │   ├── server.py               # gRPC服务器
│   │   ├── servicer.py             # 服务实现
│   │   └── client.py               # gRPC客户端
│   │
│   └── __init__.py                 # 包导出
│
├── config/
│   ├── default.yaml               # 默认配置
│   ├── development.yaml          # 开发环境配置
│   ├── production.yaml           # 生产环境配置
│   └── providers.yaml            # 供应商配置模板
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # 测试配置
│   ├── unit/
│   │   ├── test_interfaces.py    # 接口测试
│   │   ├── test_providers.py     # 供应商测试
│   │   └── test_agents.py        # Agent测试
│   └── integration/
│       ├── test_grpc.py          # gRPC集成测试
│       └── test_full_flow.py     # 完整流程测试
│
├── docs/
│   ├── architecture.md           # 本架构文档
│   ├── api_reference.md          # API参考
│   └── usage_guide.md            # 使用指南
│
├── scripts/
│   ├── generate_proto.sh         # Proto生成脚本
│   └── setup_dev.sh              # 开发环境设置
│
├── examples/
│   ├── zhipu_usage.py            # 智谱使用示例
│   ├── minimax_usage.py          # Minimax 使用示例
│   ├── advanced_agent.py         # 高级Agent示例
│   └── grpc_client_example.py   # gRPC客户端示例
│
├── pyproject.toml               # 项目配置
├── requirements.txt              # 依赖列表
├── requirements-dev.txt         # 开发依赖
├── setup.py                     # 安装脚本
├── README.md                    # 项目说明
├── LICENSE                      # 许可证
└── .gitignore                   # Git忽略文件
```

---

## 7. 实施计划

### Phase 1: 基础设施搭建 (Week 1)
- [ ] 创建项目基础结构
- [ ] 配置Python环境和依赖
- [ ] 实现配置管理模块
- [ ] 实现核心接口定义
- [ ] 搭建CI/CD流水线

### Phase 2: LLM供应商集成 (Week 2)
- [ ] 实现LLMProvider接口
- [ ] 集成OpenAI、Google Gemini、智谱ZAI
- [ ] 集成DeepSeek、豆包、Minimax
- [ ] 集成Anthropic、OpenRouter
- [ ] 实现LLM工厂模式

### Phase 3: Agent系统 (Week 3)
- [ ] 实现基础Agent框架
- [ ] 实现ReAct模式
- [ ] 实现对话Agent
- [ ] 实现工具Agent
- [ ] 实现多智能体系统

### Phase 4: 记忆和上下文 (Week 4)
- [ ] 实现记忆模块接口
- [ ] 实现短期记忆（对话历史）
- [ ] 实现长期记忆（语义记忆）
- [ ] 实现上下文管理
- [ ] 实现Token计数服务

### Phase 5: 工具和MCP支持 (Week 5)
- [ ] 实现工具管理框架
- [ ] 实现内置工具集
- [ ] 实现MCP客户端
- [ ] 实现Skills系统
- [ ] 实现人在回路功能

### Phase 6: 图像生成和日志 (Week 6)
- [ ] 实现图像生成模块
- [ ] 集成Langfuse
- [ ] 实现MongoDB日志（可选）
- [ ] 实现JSON格式输出
- [ ] 实现流式回复

### Phase 7: gRPC服务 (Week 7)
- [ ] 定义Proto文件
- [ ] 实现gRPC服务器
- [ ] 实现gRPC客户端
- [ ] 编写集成测试
- [ ] 性能优化

### Phase 8: 文档和优化 (Week 8)
- [ ] 编写API文档
- [ ] 编写使用指南
- [ ] 代码审查和优化
- [ ] 编写示例代码
- [ ] 项目发布准备

---

## 8. 依赖清单

### 核心依赖
- langchain>=0.2.0
- langchain-core>=0.2.0
- langchain-openai>=0.1.0
- langchain-anthropic>=0.1.0
- langchain-google-genai>=0.1.0
- pydantic>=2.5.0
- pydantic-settings>=2.1.0
- pyyaml>=6.0.0

### gRPC依赖
- grpcio>=1.60.0
- grpcio-tools>=1.60.0
- protobuf>=4.25.0

### 工具依赖
- mcp>=0.9.0
- aiohttp>=3.9.0
- httpx>=0.26.0

### 存储依赖
- motor>=3.3.0  # MongoDB async driver
- redis>=5.0.0

### 日志和监控
- langfuse>=2.0.0

### 开发工具
- pytest>=7.4.0
- pytest-asyncio>=0.23.0
- pytest-cov>=4.1.0
- black>=23.12.0
- isort>=5.13.0
- mypy>=1.8.0
- ruff>=0.1.0

---

## 9. 配置示例

### config/default.yaml

```yaml
# AI基座默认配置

# 项目配置
project:
  name: "ai-foundation"
  version: "1.0.0"
  environment: "development"

# LLM供应商配置
providers:
  openai:
    enabled: true
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    models:
      default: "gpt-4o"
      chat: "gpt-4o"
      reasoning: "o1"

  anthropic:
    enabled: true
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com"
    models:
      default: "claude-sonnet-4-20250514"

  google:
    enabled: true
    api_key: "${GOOGLE_API_KEY}"
    models:
      default: "gemini-2.5-pro"

  zhipu:
    enabled: true
    api_key: "${ZHIPU_API_KEY}"
    base_url: "https://open.bigmodel.cn/api/paas/v4"
    models:
      default: "glm-4-plus"

  deepseek:
    enabled: true
    api_key: "${DEEPSEEK_API_KEY}"
    base_url: "https://api.deepseek.com"
    models:
      default: "deepseek-chat"

  doubao:
    enabled: true
    api_key: "${DOUBAN_API_KEY}"
    base_url: "https://ark.cn-beijing.volces.com/api/v3"
    models:
      default: "doubao-pro-32k"

  minimax:
    enabled: true
    api_key: "${MINIMAX_API_KEY}"
    base_url: "https://api.minimax.chat/v1"
    models:
      default: "abab6.5s-chat"

  openrouter:
    enabled: true
    api_key: "${OPENROUTER_API_KEY}"
    base_url: "https://openrouter.ai/api/v1"
    models:
      default: "anthropic/claude-3.5-sonnet"

# 图像生成配置
image:
  dalle:
    enabled: false
    api_key: "${DALLE_API_KEY}"
    model: "dall-e-3"
    size: "1024x1024"
  
  stable_diffusion:
    enabled: false
    api_url: "http://localhost:7860"

# 记忆模块配置
memory:
  short_term:
    provider: "memory"
    max_messages: 20
  
  long_term:
    provider: "mongodb"
    enabled: false
    connection_string: "${MONGODB_URI}"
    database: "ai_foundation"
    collection: "memories"

# Langfuse配置
langfuse:
  enabled: true
  public_key: "${LANGFUSE_PUBLIC_KEY}"
  secret_key: "${LANGFUSE_SECRET_KEY}"
  host: "https://cloud.langfuse.com"
  timeout: 10

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
# MongoDB日志配置（可选）
mongodb_logging:
  enabled: false
  connection_string: "${MONGODB_URI}"
  database: "ai_foundation"
  collection: "ai_logs"

# gRPC服务配置
grpc:
  host: "0.0.0.0"
  port: 50051
  max_workers: 10

# 工具配置
tools:
  mcp:
    enabled: true
    servers: []
  
  builtins:
    enabled: true
    registry:
      - name: "search"
        description: "网络搜索工具"
      - name: "calculator"
        description: "数学计算工具"
      - name: "file_reader"
        description: "文件读取工具"
      - name: "file_writer"
        description: "文件写入工具"
```

---

## 10. 错误处理

### 错误码定义

```python
class AIError(Exception):
    """AI基座基础错误类"""
    code: str
    message: str

class ProviderError(AIError):
    """供应商相关错误"""
    pass

class TokenLimitExceededError(AIError):
    """Token限制超限"""
    pass

class ToolExecutionError(AIError):
    """工具执行错误"""
    pass

class ContextLengthExceededError(AIError):
    """上下文长度超限"""
    pass

class RateLimitError(AIError):
    """速率限制错误"""
    pass

class AuthenticationError(AIError):
    """认证错误"""
    pass

class ConfigurationError(AIError):
    """配置错误"""
    pass
```

---

## 11. 版本兼容性

### Python版本
- 最低版本: 3.10
- 推荐版本: 3.12

### LangChain版本
- 最低版本: 0.2.0
- 推荐版本: 0.3.0

### 依赖兼容性
- 所有依赖保持向后兼容
- 使用>=而不是==指定版本范围

---

## 12. 性能考虑

### 异步设计
- 全局使用async/await
- 使用连接池管理HTTP连接
- 使用批量处理减少API调用

### 缓存策略
- 工具描述缓存
- 模型信息缓存
- 频繁访问的记忆缓存

### 资源管理
- 限制并发请求数量
- 设置请求超时
- 实现优雅降级

---

## 13. 安全考虑

### API密钥管理
- 使用环境变量
- 配置文件不提交到版本控制
- 实现密钥轮换机制

### 输入验证
- 所有输入进行验证
- 防止注入攻击
- 限制请求大小

### 审计日志
- 记录所有API调用
- 记录异常情况
- 定期审查日志

---

*文档版本: 1.0.0*
*最后更新: 2026-02-08*
*作者: Alfred (AI Assistant)*
