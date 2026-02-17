<!-- [Input] tests/unit 目录中的单元测试文件。 -->
<!-- [Output] 说明单元测试覆盖范围与文件职责。 -->
<!-- [Pos] tests/unit 目录微架构清单。 -->
# tests/unit 目录说明
本目录存放快速执行的单元测试，用于校验核心模块的行为与兼容性。

文件清单:
- `test_core.py` - 核心模块基础行为测试集合。
- `test_zhipu_provider.py` - 智谱 provider 兼容性与流式回归测试。
- `test_minimax_provider.py` - Minimax provider（Anthropic 协议）兼容测试。
- `test_conversation_context_async.py` - 会话 Agent 与上下文管理异步回归测试。
- `test_agent_provider_resolution.py` - Agent 模型到 provider 解析回归测试。
- `test_react_agent_behavior.py` - ReAct 在非严格输出格式下的收敛行为测试。
- `test_minimax_usage.py` - Minimax 示例脚本全流程回归测试。
