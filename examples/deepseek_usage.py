# [Input] 项目根目录下的 `src` 包与运行时配置（如 API Key）。
# [Output] 提供可执行的 DeepSeek LLM 使用示例入口，包含思考模型演示。
# [Pos] examples 层 DeepSeek 示例脚本，兼容直接运行与模块方式运行。

import asyncio
import sys
from pathlib import Path

try:
    from src import (
        AgentConfig,
        AgentType,
        BuiltinTools,
        ContextManager,
        ConversationalAgent,
        ReActAgent,
        ToolManager,
        create_ai,
        get_llm_provider,
    )
except ModuleNotFoundError as exc:
    if exc.name != "src":
        raise
    # 兼容 `python examples/deepseek_usage.py` 直接运行场景
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src import (
        AgentConfig,
        AgentType,
        BuiltinTools,
        ContextManager,
        ConversationalAgent,
        ReActAgent,
        ToolManager,
        create_ai,
        get_llm_provider,
    )


async def basic_chat_example():
    """基础对话示例 - 使用 deepseek-chat 非思考模型"""
    print("=" * 50)
    print("1. 基础对话示例 (deepseek-chat)")
    print("=" * 50)

    # 创建AI基座
    ai = create_ai(provider="deepseek", model="deepseek-chat")

    # 简单对话
    response = await ai.chat("你好，请介绍一下你自己")
    print(f"用户: 你好，请介绍一下你自己")
    print(f"AI: {response}")
    print()


async def stream_chat_example():
    """流式对话示例"""
    print("=" * 50)
    print("2. 流式对话示例 (deepseek-chat)")
    print("=" * 50)

    ai = create_ai(provider="deepseek", model="deepseek-chat")

    print("用户: 请写一首关于人工智能的诗")
    print("AI: ", end="", flush=True)

    async for chunk in ai.stream("请写一首关于人工智能的诗"):
        print(chunk, end="", flush=True)

    print("\n")


async def reasoning_model_example():
    """思考模型示例 - 使用 deepseek-reasoner 返回思考过程"""
    print("=" * 50)
    print("3. 思考模型示例 (deepseek-reasoner)")
    print("=" * 50)

    # 创建AI基座，使用思考模型
    ai = create_ai(provider="deepseek", model="deepseek-reasoner")

    # 复杂问题以展示思考过程
    question = "如果在一个房间里有3只猫，每只猫前面有3只猫，每只猫后面有3只猫，请问一共有几只猫？请详细推理。"

    print(f"用户: {question}")
    print("\n--- 思考过程 ---")

    # 使用 Provider 直接获取包含思考内容的响应
    provider = get_llm_provider("deepseek")
    from src.core.interfaces import AIRequest, Message, MessageRole

    request = AIRequest(
        model="deepseek-reasoner",
        messages=[Message(role=MessageRole.USER, content=question)]
    )

    response = await provider.generate(request)

    # 打印思考过程
    if response.reasoning_content:
        print(response.reasoning_content)
    else:
        print("(此模型不返回思考过程)")

    print("\n--- 最终答案 ---")
    print(response.content)
    print()


async def agent_example():
    """Agent执行示例"""
    print("=" * 50)
    print("4. Agent执行示例")
    print("=" * 50)

    # 创建Agent
    agent = ReActAgent(AgentConfig(
        name="research-agent",
        agent_type=AgentType.REACT,
        system_prompt="You are a helpful research assistant.",
        model="deepseek-chat",
        max_iterations=5,
    ))

    # 执行任务
    result = await agent.execute("What is the capital of China?")

    print("任务: What is the capital of China?")
    print(f"成功: {result.success}")
    print(f"结果: {result.output}")
    if result.error:
        print(f"错误: {result.error}")
    print()


async def tool_example():
    """工具使用示例"""
    print("=" * 50)
    print("5. 工具使用示例")
    print("=" * 50)

    # 创建工具管理器
    tool_manager = ToolManager()
    BuiltinTools.register_builtins(tool_manager)

    # 使用计算器
    result = await tool_manager.execute_tool("calculator", expression="15 * 24 + 8")
    print(f"计算 15 * 24 + 8 = {result}")

    # 获取当前时间
    datetime_result = await tool_manager.execute_tool("get_datetime")
    print(f"当前时间: {datetime_result}")

    # 列出所有工具
    tools = tool_manager.list_tools()
    print(f"\n可用工具: {[t['name'] for t in tools]}")
    print()


async def conversation_example():
    """多轮对话示例"""
    print("=" * 50)
    print("6. 多轮对话示例")
    print("=" * 50)

    agent = ConversationalAgent(AgentConfig(
        name="assistant",
        agent_type=AgentType.CONVERSATIONAL,
        model="deepseek-chat",
    ))

    conversation_id = "user-456"

    # 第一轮
    result1 = await agent.execute("我喜欢编程和阅读", conversation_id=conversation_id)
    print("用户: 我喜欢编程和阅读")
    print(f"成功: {result1.success}")
    print(f"AI: {result1.output}")
    if result1.error:
        print(f"错误: {result1.error}")

    # 第二轮（Agent会记住爱好）
    result2 = await agent.execute("根据我的爱好，推荐一本书和一个编程项目", conversation_id=conversation_id)
    print("\n用户: 根据我的爱好，推荐一本书和一个编程项目")
    print(f"成功: {result2.success}")
    print(f"AI: {result2.output}")
    if result2.error:
        print(f"错误: {result2.error}")
    print()


async def reasoning_comparison_example():
    """思考模型与非思考模型对比"""
    print("=" * 50)
    print("7. 思考模型与非思考模型对比")
    print("=" * 50)

    question = "一个人有3个苹果，吃了半个，还剩多少个？请解释原因。"

    provider = get_llm_provider("deepseek")
    from src.core.interfaces import AIRequest, Message, MessageRole

    request = AIRequest(
        model="deepseek-chat",
        messages=[Message(role=MessageRole.USER, content=question)]
    )

    # 非思考模型
    print(f"问题: {question}\n")
    print("--- 非思考模型 (deepseek-chat) ---")
    request.model = "deepseek-chat"
    response_chat = await provider.generate(request)
    print(f"回答: {response_chat.content}")
    print()

    # 思考模型
    print("--- 思考模型 (deepseek-reasoner) ---")
    request.model = "deepseek-reasoner"
    response_reasoner = await provider.generate(request)

    if response_reasoner.reasoning_content:
        print("思考过程:")
        print(response_reasoner.reasoning_content[:500] + "..." if len(response_reasoner.reasoning_content) > 500 else response_reasoner.reasoning_content)

    print(f"\n回答: {response_reasoner.content}")
    print()


async def provider_switching_example():
    """供应商信息查看"""
    print("=" * 50)
    print("8. 供应商信息查看")
    print("=" * 50)

    provider_name = "deepseek"

    try:
        provider = get_llm_provider(provider_name)
        info = provider.get_model_info()
        print(f"供应商: {provider_name}")
        print(f"  可用: {info.get('available', False)}")
        print(f"  模型: {info.get('models', {})}")
    except Exception as e:
        print(f"供应商 {provider_name}: {e}")
    print()


async def context_management_example():
    """上下文管理示例"""
    print("=" * 50)
    print("9. 上下文管理示例")
    print("=" * 50)

    manager = ContextManager()
    conversation_id = "demo-conversation-deepseek"

    # 添加消息
    await manager.add_message(conversation_id, "user", "Hello")
    await manager.add_message(conversation_id, "assistant", "Hi! How can I help you?")
    await manager.add_message(conversation_id, "user", "What is DeepSeek?")

    # 获取历史
    messages = await manager.get_messages(conversation_id)
    print(f"对话消息数: {len(messages)}")

    # 计算Token
    token_count = manager.calculate_tokens(messages)
    print(f"Token数: {token_count}")

    # 获取上下文信息
    info = await manager.get_context_info(conversation_id)
    print(f"上下文信息: {info}")
    print()


async def main():
    """运行所有示例"""
    print("\n" + "=" * 50)
    print("DeepSeek LLM 使用示例")
    print("=" * 50 + "\n")

    # 运行所有示例
    await basic_chat_example()
    await stream_chat_example()
    await reasoning_model_example()  # 重点：思考模型示例
    await agent_example()
    await tool_example()
    await conversation_example()
    await reasoning_comparison_example()  # 重点：对比示例
    await provider_switching_example()
    await context_management_example()

    print("=" * 50)
    print("所有示例运行完成!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
