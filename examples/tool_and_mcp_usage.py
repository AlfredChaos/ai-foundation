# [Input] 项目根目录下的 `src` 包，DeepSeek API Key 配置。
# [Output] 提供工具调用和 MCP 使用示例入口。
# [Pos] examples 层工具调用与 MCP 示例脚本，展示 DeepSeek + 工具调用的完整流程。

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from src import (
        AgentConfig,
        AgentType,
        BuiltinTools,
        ReActAgent,
        ToolManager,
        create_ai,
        get_llm_provider,
    )
    from src.core.interfaces import AIRequest, Message, MessageRole
    from src.tools.mcp.client import MCPClient, MCPClientManager, MCPServerConfig
except ModuleNotFoundError as exc:
    if exc.name != "src":
        raise
    # 兼容 `python examples/tool_and_mcp_usage.py` 直接运行场景
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src import (
        AgentConfig,
        AgentType,
        BuiltinTools,
        ReActAgent,
        ToolManager,
        create_ai,
        get_llm_provider,
    )
    from src.core.interfaces import AIRequest, Message, MessageRole
    from src.tools.mcp.client import MCPClient, MCPClientManager, MCPServerConfig


# ============================================================================
# 自定义工具定义
# ============================================================================

async def weather_search(city: str) -> str:
    """
    模拟天气查询工具

    Args:
        city: 城市名称

    Returns:
        str: 天气信息
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，温度 15-25°C，空气质量优",
        "上海": "多云，温度 18-26°C，空气质量良",
        "深圳": "阴天，温度 22-30°C，空气质量优",
        "杭州": "小雨，温度 16-24°C，空气质量优",
    }
    return weather_data.get(city, f"{city} 暂无天气数据")


async def stock_price(symbol: str) -> str:
    """
    模拟股票价格查询工具

    Args:
        symbol: 股票代码

    Returns:
        str: 股票价格信息
    """
    # 模拟股票数据
    stock_data = {
        "AAPL": "Apple Inc. (AAPL): $178.52 +2.34%",
        "GOOGL": "Alphabet Inc. (GOOGL): $141.80 +1.15%",
        "MSFT": "Microsoft Corp (MSFT): $378.91 +3.21%",
        "TSLA": "Tesla Inc (TSLA): $248.50 -1.80%",
    }
    return stock_data.get(symbol.upper(), f"股票代码 {symbol} 暂无数据")


async def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    单位转换工具

    Args:
        value: 数值
        from_unit: 原单位
        to_unit: 目标单位

    Returns:
        str: 转换结果
    """
    # 长度单位转换
    length_units = {"m": 1, "km": 1000, "cm": 0.01, "mm": 0.001, "inch": 0.0254, "ft": 0.3048}

    if from_unit in length_units and to_unit in length_units:
        # 先转换为米，再转换为目标单位
        meters = value * length_units[from_unit]
        result = meters / length_units[to_unit]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    return f"不支持从 {from_unit} 到 {to_unit} 的转换"


# ============================================================================
# 示例函数
# ============================================================================

async def basic_tool_example():
    """示例 1: 基础工具使用"""
    print("=" * 60)
    print("示例 1: 基础工具使用")
    print("=" * 60)

    # 创建工具管理器
    tool_manager = ToolManager()

    # 注册自定义工具
    tool_manager.register_tool(
        name="weather_search",
        func=weather_search,
        description="查询指定城市的天气信息",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如：北京、上海"}
            },
            "required": ["city"],
        }
    )

    tool_manager.register_tool(
        name="stock_price",
        func=stock_price,
        description="查询股票价格信息",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如：AAPL, TSLA"}
            },
            "required": ["symbol"],
        }
    )

    # 列出所有工具
    tools = tool_manager.list_tools()
    print(f"\n可用工具: {len(tools)} 个")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

    # 执行工具
    print("\n--- 执行天气查询 ---")
    result = await tool_manager.execute_tool("weather_search", city="北京")
    print(f"result = {result}")

    print("\n--- 执行股票查询 ---")
    result = await tool_manager.execute_tool("stock_price", symbol="AAPL")
    print(f"result = {result}")
    print()


async def builtin_tools_example():
    """示例 2: 使用内置工具"""
    print("=" * 60)
    print("示例 2: 使用内置工具")
    print("=" * 60)

    # 创建工具管理器并注册内置工具
    tool_manager = ToolManager()
    BuiltinTools.register_builtins(tool_manager)

    # 列出所有内置工具
    tools = tool_manager.list_tools()
    print(f"\n内置工具: {len(tools)} 个")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

    # 使用计算器工具
    print("\n--- 计算器工具 ---")
    result = await tool_manager.execute_tool("calculator", expression="2 ** 10 + 100")
    print(f"2^10 + 100 = {result}")

    # 使用日期时间工具
    print("\n--- 日期时间工具 ---")
    result = await tool_manager.execute_tool("get_datetime", format="%Y年%m月%d日 %H:%M:%S")
    print(f"当前时间: {result}")

    # 使用文件读写工具
    print("\n--- 文件读写工具 ---")
    test_file = "/tmp/test_tool_file.txt"
    await tool_manager.execute_tool("file_writer", file_path=test_file, content="Hello from DeepSeek + Tools!")
    print(f"已写入文件: {test_file}")

    content = await tool_manager.execute_tool("file_reader", file_path=test_file)
    print(f"文件内容: {content}")
    print()


async def deepseek_tool_calling_example():
    """示例 3: DeepSeek + 工具调用（手动模拟）"""
    print("=" * 60)
    print("示例 3: DeepSeek + 工具调用")
    print("=" * 60)

    # 创建工具管理器
    tool_manager = ToolManager()
    BuiltinTools.register_builtins(tool_manager)

    # 添加自定义工具
    tool_manager.register_tool(
        name="weather_search",
        func=weather_search,
        description="查询指定城市的天气信息",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"],
        }
    )

    # 创建 DeepSeek AI
    provider = get_llm_provider("deepseek")

    # 获取工具定义
    tools = tool_manager.list_tools()
    tool_definitions = [
        tool_manager.create_tool_definition(t["name"])
        for t in tools
    ]

    print("\n可用工具定义:")
    for tool_def in tool_definitions:
        print(f"  {tool_def['function']['name']}: {tool_def['function']['description']}")

    # 构建带工具的请求
    request = AIRequest(
        model="deepseek-chat",
        messages=[
            Message(role=MessageRole.SYSTEM, content="你是一个有用的助手，可以使用提供的工具来帮助用户。"),
            Message(role=MessageRole.USER, content="北京今天的天气怎么样？"),
        ],
        tools=tool_definitions,
    )

    print("\n--- 发送请求到 DeepSeek ---")
    print(f"用户问题: 北京今天的天气怎么样？")

    try:
        response = await provider.generate(request)
        print(f"\nDeepSeek 回复:\n{response.content}")

        # 注意: 实际工具调用需要解析 response.tool_calls
        # 这里是简化示例，实际实现需要处理工具调用流程
        if response.tool_calls:
            print(f"\n检测到工具调用: {response.tool_calls}")
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                print(f"调用工具: {tool_name}")
                print(f"参数: {tool_args}")

    except Exception as e:
        print(f"请求出错: {e}")
        print("\n提示: DeepSeek 工具调用功能需要在实际 API 中启用")
    print()


async def react_agent_with_tools_example():
    """示例 4: ReAct Agent + 工具使用"""
    print("=" * 60)
    print("示例 4: ReAct Agent + 工具使用")
    print("=" * 60)

    # 创建工具管理器并注册工具
    tool_manager = ToolManager()

    # 注册自定义工具
    tool_manager.register_tool(
        name="weather_search",
        func=weather_search,
        description="查询指定城市的天气信息",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"],
        }
    )

    tool_manager.register_tool(
        name="stock_price",
        func=stock_price,
        description="查询股票价格信息",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"}
            },
            "required": ["symbol"],
        }
    )

    tool_manager.register_tool(
        name="unit_converter",
        func=unit_converter,
        description="单位转换工具",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "数值"},
                "from_unit": {"type": "string", "description": "原单位"},
                "to_unit": {"type": "string", "description": "目标单位"}
            },
            "required": ["value", "from_unit", "to_unit"],
        }
    )

    # 注册内置工具
    BuiltinTools.register_builtins(tool_manager)

    # 创建 ReAct Agent
    agent = ReActAgent(AgentConfig(
        name="tool-agent",
        agent_type=AgentType.REACT,
        system_prompt="你是一个智能助手，可以使用工具来查询信息。请根据用户需求选择合适的工具。",
        model="deepseek-chat",
        max_iterations=5,
    ))

    # 绑定工具管理器
    agent.tool_manager = tool_manager

    print("\n--- 任务 1: 查询天气 ---")
    result = await agent.execute("帮我查询一下北京的天气情况")
    print(f"成功: {result.success}")
    print(f"结果: {result.output}")
    if result.error:
        print(f"错误: {result.error}")

    print("\n--- 任务 2: 股票查询 ---")
    result = await agent.execute("查一下苹果公司(AAPL)的股价")
    print(f"成功: {result.success}")
    print(f"结果: {result.output}")
    if result.error:
        print(f"错误: {result.error}")

    print("\n--- 任务 3: 单位转换 ---")
    result = await agent.execute("把 100 英寸转换成厘米")
    print(f"成功: {result.success}")
    print(f"结果: {result.output}")
    if result.error:
        print(f"错误: {result.error}")
    print()


async def mcp_client_example():
    """示例 5: MCP 客户端使用"""
    print("=" * 60)
    print("示例 5: MCP 客户端使用")
    print("=" * 60)

    # 创建 MCP 客户端
    client = MCPClient()

    print("\n--- 连接 MCP 服务器 ---")
    # 注意: 这是一个模拟连接，实际使用时需要配置真实的 MCP 服务器地址
    connected = await client.connect(
        server_url="ws://localhost:3000/mcp",  # 示例地址
        api_key=None
    )

    if connected:
        print("✓ MCP 服务器连接成功")

        # 列出可用工具
        tools = await client.list_tools()
        print(f"\nMCP 服务器工具: {len(tools)} 个")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")

        # 调用工具
        print("\n--- 调用 MCP 工具 ---")
        result = await client.call_tool("example_tool", input="test")
        print(f"结果: {result}")

        # 获取服务器信息
        server_info = client.get_server_info()
        print(f"\n服务器信息: {server_info}")

        # 断开连接
        await client.disconnect()
        print("\n✓ 已断开 MCP 连接")
    else:
        print("✗ MCP 服务器连接失败")
        print("\n提示: MCP 需要运行 MCP 服务器才能正常工作")
        print("可以访问 https://modelcontextprotocol.io 了解更多")
    print()


async def mcp_manager_example():
    """示例 6: MCP 管理器 - 多服务器管理"""
    print("=" * 60)
    print("示例 6: MCP 管理器 - 多服务器管理")
    print("=" * 60)

    # 创建 MCP 管理器
    manager = MCPClientManager()

    # 配置多个 MCP 服务器（模拟配置）
    servers = [
        MCPServerConfig(
            name="weather-server",
            url="ws://localhost:3001/mcp",
            enabled=True
        ),
        MCPServerConfig(
            name="search-server",
            url="ws://localhost:3002/mcp",
            enabled=True
        ),
        MCPServerConfig(
            name="database-server",
            url="ws://localhost:3003/mcp",
            enabled=True
        ),
    ]

    print("\n--- 添加 MCP 服务器 ---")
    for server_config in servers:
        print(f"添加服务器: {server_config.name} ({server_config.url})")
        # 注意: 实际连接会失败（因为没有真实服务器），这里仅作演示
        success = await manager.add_server(server_config)
        status = "✓ 连接成功" if success else "✗ 连接失败"
        print(f"  {status}")

    # 列出所有服务器
    servers_list = manager.list_servers()
    print(f"\n已配置服务器: {len(servers_list)} 个")
    for server in servers_list:
        print(f"  - {server['name']}: {server['url']}")

    # 列出所有工具（来自所有服务器）
    all_tools = manager.list_all_tools()
    print(f"\n所有可用工具: {len(all_tools)} 个")
    for tool in all_tools:
        print(f"  - {tool['name']} (来自: {tool.get('server', 'unknown')})")

    print("\n提示: 配置真实的 MCP 服务器后可以实际调用工具")
    print()


async def custom_tool_with_base_class_example():
    """示例 7: 使用 BaseTool 创建自定义工具"""
    print("=" * 60)
    print("示例 7: 使用 BaseTool 创建自定义工具")
    print("=" * 60)

    from src.tools.tool_manager import BaseTool

    # 定义自定义工具类
    class EmailTool(BaseTool):
        """邮件发送工具"""

        @property
        def name(self) -> str:
            return "send_email"

        @property
        def description(self) -> str:
            return "发送邮件给指定收件人"

        @property
        def parameters(self) -> Dict:
            return {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人邮箱"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件内容"}
                },
                "required": ["to", "subject", "body"],
            }

        async def execute(self, **kwargs) -> str:
            """执行邮件发送（模拟）"""
            to = kwargs.get("to")
            subject = kwargs.get("subject")
            body = kwargs.get("body")
            return f"✓ 邮件已发送\n  收件人: {to}\n  主题: {subject}\n  内容: {body[:50]}..."

    class DatabaseTool(BaseTool):
        """数据库查询工具"""

        @property
        def name(self) -> str:
            return "query_database"

        @property
        def description(self) -> str:
            return "执行数据库查询"

        @property
        def parameters(self) -> Dict:
            return {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "表名"},
                    "conditions": {"type": "string", "description": "查询条件"}
                },
                "required": ["table"],
            }

        async def execute(self, **kwargs) -> str:
            """执行数据库查询（模拟）"""
            table = kwargs.get("table")
            conditions = kwargs.get("conditions", "")
            return f"✓ 查询成功\n  表: {table}\n  条件: {conditions or '无'}\n  结果: 找到 5 条记录"

    # 创建工具管理器
    tool_manager = ToolManager()

    # 创建工具实例
    email_tool = EmailTool()
    db_tool = DatabaseTool()

    # 注册工具
    tool_manager.register_tool(
        name=email_tool.name,
        func=email_tool.execute,
        description=email_tool.description,
        parameters=email_tool.parameters
    )

    tool_manager.register_tool(
        name=db_tool.name,
        func=db_tool.execute,
        description=db_tool.description,
        parameters=db_tool.parameters
    )

    print("\n已注册自定义工具:")
    tools = tool_manager.list_tools()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

    # 使用工具
    print("\n--- 发送邮件 ---")
    result = await tool_manager.execute_tool(
        "send_email",
        to="user@example.com",
        subject="测试邮件",
        body="这是一封测试邮件，来自 DeepSeek + Tools 示例。"
    )
    print(result)

    print("\n--- 查询数据库 ---")
    result = await tool_manager.execute_tool(
        "query_database",
        table="users",
        conditions="age > 18"
    )
    print(result)
    print()


async def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("DeepSeek + 工具调用 + MCP 使用示例")
    print("=" * 60 + "\n")

    # 运行所有示例
    await basic_tool_example()
    await builtin_tools_example()
    await deepseek_tool_calling_example()
    await react_agent_with_tools_example()
    await mcp_client_example()
    await mcp_manager_example()
    await custom_tool_with_base_class_example()

    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
    print("\n提示:")
    print("  1. 确保 DEEPSEEK_API_KEY 环境变量已设置")
    print("  2. MCP 功能需要运行实际的 MCP 服务器")
    print("  3. 更多工具定义参考 src/tools/tool_manager.py")


if __name__ == "__main__":
    asyncio.run(main())
