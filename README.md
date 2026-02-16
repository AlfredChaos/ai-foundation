<!-- [Input] 项目定位、安装方式与示例运行指令。 -->
<!-- [Output] 提供仓库概览与快速开始指引。 -->
<!-- [Pos] 仓库根目录说明文档。 -->
# AI Foundation

基于Python+LangChain的AI基座，提供快速集成AI能力的开发框架。

## 特性

- 🤖 **多LLM供应商支持** - OpenAI、Anthropic、Google Gemini、智谱ZAI、DeepSeek、豆包、Minimax、OpenRouter
- 🧠 **Agent系统** - ReAct Agent、对话Agent、自定义Agent
- 🔧 **工具管理** - 灵活的工具注册和执行框架
- 📚 **记忆模块** - 短期记忆（内存）、长期记忆（MongoDB/Redis）
- 🎯 **上下文管理** - Token计算、上下文截断
- 📊 **监控集成** - Langfuse可观测性支持
- 🚀 **gRPC服务** - 跨平台调用支持
- ✅ **全面测试** - 单元测试和集成测试覆盖

## 快速开始

```bash
# 安装
cd /opt/ai-foundation
pip install -e ".[dev]"

# 配置
export ZHIPU_API_KEY=your-key

# 运行示例
python examples/zhipu_usage.py
```

## 使用示例

```python
import asyncio
from src import create_ai

async def main():
    ai = create_ai(provider="zhipu", model="GLM-4.7")

    response = await ai.chat("你好！")
    print(response)

asyncio.run(main())
```

## 项目结构

```
ai-foundation/
├── src/
│   ├── core/           # 核心接口和抽象类
│   ├── providers/      # LLM和图像供应商
│   ├── agents/         # Agent实现
│   ├── tools/          # 工具管理和MCP
│   ├── memory/         # 记忆模块
│   ├── context/        # 上下文管理
│   ├── services/       # 日志、Token、人在回路
│   ├── grpc_service/   # gRPC服务
│   └── config/         # 配置管理
├── examples/           # 使用示例
├── tests/              # 测试代码
├── docs/               # 文档
└── config/             # 配置文件
```

## 文档

- [架构文档](docs/architecture.md)
- [使用指南](docs/usage_guide.md)
- [API参考](docs/api_reference.md)

## 许可证

MIT
