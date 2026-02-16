<!-- [Input] Agent 抽象与具体实现代码。 -->
<!-- [Output] 说明 agents 目录的职责与文件分工。 -->
<!-- [Pos] src/agents 目录微架构清单。 -->
# agents 目录说明
本目录实现 Agent 抽象基类与具体策略，用于统一执行任务、维护上下文和工具调用。

文件清单:
- `base.py` - Agent 基类与通用执行能力（provider 解析、历史管理、请求构建）。
- `react_agent.py` - ReAct 推理-行动循环实现。
- `conversational_agent.py` - 多轮对话 Agent，实现上下文记忆与会话总结。
- `__init__.py` - Agent 模块导出入口。
