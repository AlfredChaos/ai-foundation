# AI Foundation 使用指南

## 目录

1. [快速开始](#快速开始)
2. [安装配置](#安装配置)
3. [基础用法](#基础用法)
4. [高级功能](#高级功能)
5. [Agent系统](#agent系统)
6. [工具集成](#工具集成)
7. [记忆管理](#记忆管理)
8. [gRPC服务](#grpc服务)
9. [配置说明](#配置说明)

---

## 快速开始

### 安装

```bash
# 克隆项目
cd /opt/ai-foundation

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -e ".[dev]"

# 编译gRPC
python -m grpc_tools.protoc -I=src/grpc_service --python_out=src/grpc_service --grpc_python_out=src/grpc_service src/grpc_service/ai_core.proto
```

### 第一个AI应用

```python
import asyncio
from src import create_ai

async def main():
    # 创建AI实例
    ai = create_ai(provider="openai", model="gpt-4o")
    
    # 简单对话
    response = await ai.chat("你好，请介绍一下你自己")
    print(response)

asyncio.run(main())
```

---

## 安装配置

### 环境变量配置

创建 `.env` 文件：

```env
# OpenAI
OPENAI_API_KEY=sk-your-openai-key

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key

# Google
GOOGLE_API_KEY=your-google-key

# 智谱
ZHIPU_API_KEY=your-zhipu-key

# DeepSeek
DEEPSEEK_API_KEY=your-deepseek-key

# MongoDB (可选)
MONGODB_URI=mongodb://localhost:27017

# Langfuse (可选)
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
```

### 配置文件

编辑 `config/default.yaml`：

```yaml
providers:
  openai:
    enabled: true
    api_key: "${OPENAI_API_KEY}"
    models:
      default: "gpt-4o"
      chat: "gpt-4o"
```

---

## 基础用法

### 单次调用

```python
from src.providers.llm import get_llm_provider
from src.core.interfaces import AIRequest, Message, MessageRole

async def single_call():
    provider = get_llm_provider("openai")
    
    request = AIRequest(
        model="gpt-4o",
        messages=[
            Message(role=MessageRole.USER, content="Explain quantum computing")
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    response = await provider.generate(request)
    print(response.content)
    print(f"Tokens: {response.usage.total_tokens}")
```

### 流式回复

```python
async def stream_response():
    provider = get_llm_provider("openai")
    
    request = AIRequest(
        model="gpt-4o",
        messages=[
            Message(role=MessageRole.USER, content="Write a story about AI")
        ],
        stream=True
    )
    
    async for chunk in provider.stream_generate(request):
        print(chunk, end="", flush=True)
```

### 切换供应商

```python
# 使用不同供应商
providers = ["openai", "anthropic", "deepseek", "zhipu", "google"]

for provider_name in providers:
    try:
        provider = get_llm_provider(provider_name)
        info = provider.get_model_info()
        print(f"{provider_name}: {info}")
    except Exception as e:
        print(f"{provider_name}: {e}")
```

---

## 高级功能

### JSON格式输出

```python
request = AIRequest(
    model="gpt-4o",
    messages=[Message(role=MessageRole.USER, content="Generate JSON data")],
    json_mode=True
)

response = await provider.generate(request)
import json
data = json.loads(response.content)
```

### Token计数和成本估算

```python
from src.services import TokenCounter

counter = TokenCounter()

# 计数
tokens = counter.count("Hello, world!")
print(f"Tokens: {tokens}")

# 估算成本
cost = counter.estimate_cost(
    prompt_tokens=1000,
    completion_tokens=500,
    model="gpt-4o"
)
print(f"Estimated cost: ${cost:.4f}")
```

---

## Agent系统

### ReAct Agent

```python
from src.agents import ReActAgent, AgentConfig, AgentType

agent = ReActAgent(AgentConfig(
    name="research-agent",
    agent_type=AgentType.REACT,
    model="gpt-4o",
    system_prompt="You are a research assistant.",
    max_iterations=10
))

result = await agent.execute("Research the benefits of meditation")
print(result.output)
print(f"Steps: {len(result.intermediate_steps)}")
```

### 对话Agent

```python
from src.agents import ConversationalAgent, AgentConfig

agent = ConversationalAgent(AgentConfig(
    name="chatbot",
    agent_type=AgentType.CONVERSATIONAL,
    model="gpt-4o"
))

# 第一轮
result1 = await agent.execute(
    "我叫张三",
    conversation_id="user-123"
)
print(result1.output)

# 第二轮（记住名字）
result2 = await agent.execute(
    "我的名字是什么？",
    conversation_id="user-123"
)
print(result2.output)
```

### 自定义Agent

```python
from src.agents import BaseAgent, AgentConfig, AgentResult, AgentType

class CustomAgent(BaseAgent):
    async def execute(self, task: str, **kwargs) -> AgentResult:
        # 自定义逻辑
        return AgentResult(success=True, output=f"Processed: {task}")
    
    async def plan(self, task: str):
        return ["Step 1", "Step 2"]
    
    async def evaluate(self, result):
        return True
```

---

## 工具集成

### 使用内置工具

```python
from src.tools import ToolManager, BuiltinTools

tool_manager = ToolManager()
BuiltinTools.register_builtins(tool_manager)

# 执行工具
result = await tool_manager.execute_tool("calculator", expression="2 + 3 * 4")
print(f"Result: {result}")

# 获取当前时间
datetime_result = await tool_manager.execute_tool("get_datetime")
print(f"Now: {datetime_result}")
```

### 注册自定义工具

```python
async def search_wikipedia(query: str) -> str:
    """搜索维基百科"""
    return f"Results for: {query}"

async def translate(text: str, target_lang: str) -> str:
    """翻译文本"""
    return f"Translated '{text}' to {target_lang}"

# 注册工具
tool_manager.register_tool(
    name="wikipedia_search",
    func=search_wikipedia,
    description="Search Wikipedia for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
)

tool_manager.register_tool(
    name="translate",
    func=translate,
    description="Translate text to target language",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "target_lang": {"type": "string"}
        },
        "required": ["text", "target_lang"]
    }
)
```

### MCP工具

```python
from src.tools import MCPClient, MCPServerConfig

# 连接MCP服务器
client = MCPClient()
await client.connect("ws://mcp-server.example.com")

# 列出工具
tools = await client.list_tools()
print(f"Available tools: {len(tools)}")

# 调用工具
result = await client.call_tool("mcp-tool-name", param="value")
```

---

## 记忆管理

### 短期记忆（内存）

```python
from src.memory import MemoryManager, MemoryConfig

manager = MemoryManager(MemoryConfig())

# 存储对话
await manager.store_conversation(
    conversation_id="chat-123",
    role="user",
    content="记住这个信息：我的生日是5月15日",
    memory_type="short_term"
)

# 获取历史
history = await manager.get_conversation_history("chat-123")
```

### 长期记忆（MongoDB）

```python
from src.memory import MongoDBProvider

provider = MongoDBProvider(
    connection_string="mongodb://localhost:27017",
    database="ai_foundation"
)

# 存储长期记忆
await provider.store(
    key="user-preference-123",
    content={"theme": "dark", "language": "zh-CN"},
    metadata={"type": "preferences"}
)

# 检索
memory = await provider.retrieve("user-preference-123")
```

### 语义搜索

```python
# 搜索记忆
results = await provider.search(
    query="user preferences",
    top_k=5
)
```

---

## gRPC服务

### 启动服务

```python
from src.grpc_service import serve

async def main():
    server = await serve(port="50051", max_workers=10)
    await server.wait_for_termination()

# python -m src.grpc_service.server
```

### 客户端调用

```python
import grpc

# 连接
channel = grpc.aio.insecure_channel('localhost:50051')

# 调用服务
response = await stub.SingleCall(SingleCallRequest(
    model="gpt-4o",
    messages=[...],
    temperature=0.7
))
```

---

## 配置说明

### 完整配置示例

```yaml
project:
  name: "ai-foundation"
  version: "1.0.0"
  environment: "development"

providers:
  openai:
    enabled: true
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    models:
      default: "gpt-4o"
      chat: "gpt-4o"
  
  anthropic:
    enabled: true
    api_key: "${ANTHROPIC_API_KEY}"
    models:
      default: "claude-3-sonnet-20240229"

memory:
  short_term:
    provider: "memory"
    max_messages: 20
  
  long_term:
    provider: "mongodb"
    enabled: false
    connection_string: "${MONGODB_URI}"

langfuse:
  enabled: false
  public_key: "${LANGFUSE_PUBLIC_KEY}"
  secret_key: "${LANGFUSE_SECRET_KEY}"

grpc:
  host: "0.0.0.0"
  port: 50051
  max_workers: 10

tools:
  mcp:
    enabled: true
    servers:
      - name: "example-server"
        url: "ws://localhost:8080"
```

---

## 最佳实践

1. **错误处理**
   ```python
   try:
       response = await provider.generate(request)
   except Exception as e:
       print(f"Error: {e}")
   ```

2. **资源管理**
   ```python
   # 使用上下文管理器
   async with get_provider("openai") as provider:
       response = await provider.generate(request)
   ```

3. **性能优化**
   - 使用流式输出处理长文本
   - 缓存频繁使用的工具描述
   - 限制对话历史长度

4. **安全**
   - 不要在代码中硬编码API密钥
   - 使用环境变量
   - 限制请求频率

---

## 示例代码

查看 `examples/` 目录：

- `basic_usage.py` - 基础用法示例
- `advanced_agent.py` - Agent高级示例

```bash
# 运行示例
python examples/basic_usage.py
python examples/advanced_agent.py
```
