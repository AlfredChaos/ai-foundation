<!-- [Input] 各 LLM provider 实现文件与工厂注册信息。 -->
<!-- [Output] 说明 llm provider 目录职责与文件角色。 -->
<!-- [Pos] src/providers/llm 目录微架构清单。 -->
# llm provider 目录说明
本目录实现 LLM 供应商抽象工厂与各供应商协议适配，统一对外提供 `generate/stream_generate` 能力。

文件清单:
- `factory.py` - Provider 抽象与注册工厂。
- `openai.py` - OpenAI 及兼容接口实现。
- `anthropic.py` - Anthropic 协议实现。
- `google.py` - Gemini 协议实现。
- `zhipu.py` - 智谱实现（兼容同步/异步 SDK）。
- `deepseek.py` - DeepSeek 实现。
- `doubao.py` - 豆包实现。
- `minimax.py` - Minimax 实现（Anthropic 协议，双头鉴权并校验缺失 API Key）。
- `openrouter.py` - OpenRouter 实现。
- `__init__.py` - provider 模块导出入口。
