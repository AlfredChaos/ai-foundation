# [Input] 项目根目录下的 `src` 包与运行时配置（如 API Key）。
# [Output] 提供可执行的 Google Gemini LLM 使用示例入口。
# [Pos] examples 层 Gemini 示例脚本，兼容直接运行与模块方式运行。

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
    # 兼容 `python examples/gemini_usage.py` 直接运行场景
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
    """基础对话示例"""
    print("=" * 50)
    print("1. 基础对话示例 (gemini-2.5-flash)")
    print("=" * 50)

    # 创建AI基座
    ai = create_ai(provider="google", model="gemini-2.5-flash")

    # 简单对话
    response = await ai.chat("你好，请用中文介绍一下你自己")
    print(f"用户: 你好，请用中文介绍一下你自己")
    print(f"AI: {response}")
    print()


async def stream_chat_example():
    """流式对话示例"""
    print("=" * 50)
    print("2. 流式对话示例")
    print("=" * 50)

    ai = create_ai(provider="google", model="gemini-2.5-flash")

    print("用户: 请写一首关于科技与未来的短诗")
    print("AI: ", end="", flush=True)

    async for chunk in ai.stream("请写一首关于科技与未来的短诗"):
        print(chunk, end="", flush=True)

    print("\n")


async def code_generation_example():
    """代码生成示例 - Gemini 擅长代码生成"""
    print("=" * 50)
    print("3. 代码生成示例")
    print("=" * 50)

    ai = create_ai(provider="google", model="gemini-2.5-flash")

    prompt = """请用 Python 写一个函数，实现快速排序算法，并添加类型注解和中文注释。"""

    print(f"用户: {prompt}")
    print("\nAI 回复:")

    response = await ai.chat(prompt)
    print(response)
    print()


async def multimodal_example():
    """多模态能力展示（文本描述）"""
    print("=" * 50)
    print("4. 多模态能力描述")
    print("=" * 50)

    ai = create_ai(provider="google", model="gemini-2.5-flash")

    prompt = "请描述一下如何识别图片中的物体，有哪些常用的计算机视觉技术？"

    print(f"用户: {prompt}")
    print("\nAI:")

    response = await ai.chat(prompt)
    print(response)
    print()


async def agent_example():
    """Agent执行示例"""
    print("=" * 50)
    print("5. Agent执行示例")
    print("=" * 50)

    # 创建Agent
    agent = ReActAgent(AgentConfig(
        name="research-agent",
        agent_type=AgentType.REACT,
        system_prompt="You are a helpful research assistant.",
        model="gemini-2.5-flash",
        max_iterations=5,
    ))

    # 执行任务
    result = await agent.execute("Explain what is machine learning in simple terms.")

    print("任务: Explain what is machine learning in simple terms.")
    print(f"成功: {result.success}")
    print(f"结果: {result.output}")
    if result.error:
        print(f"错误: {result.error}")
    print()


async def tool_example():
    """工具使用示例"""
    print("=" * 50)
    print("6. 工具使用示例")
    print("=" * 50)

    # 创建工具管理器
    tool_manager = ToolManager()
    BuiltinTools.register_builtins(tool_manager)

    # 使用计算器
    result = await tool_manager.execute_tool("calculator", expression="2 ** 10 + 100")
    print(f"计算 2^10 + 100 = {result}")

    # 获取当前时间
    datetime_result = await tool_manager.execute_tool("get_datetime")
    print(f"当前时间: {datetime_result}")

    # 列出所有工具
    tools = tool_manager.list_tools()
    print(f"\n可用工具: {[t['name'] for t in tools]}")
    print()


async def conversation_example():
    """多轮对话示例 - 展示上下文理解能力"""
    print("=" * 50)
    print("7. 多轮对话示例")
    print("=" * 50)

    agent = ConversationalAgent(AgentConfig(
        name="gemini-assistant",
        agent_type=AgentType.CONVERSATIONAL,
        model="gemini-2.5-flash",
    ))

    conversation_id = "user-gemini-001"

    # 第一轮
    result1 = await agent.execute("我想学习 Python 编程", conversation_id=conversation_id)
    print("用户: 我想学习 Python 编程")
    print(f"成功: {result1.success}")
    print(f"AI: {result1.output}")
    if result1.error:
        print(f"错误: {result1.error}")

    # 第二轮（AI 会记得上下文）
    result2 = await agent.execute("请给我推荐一些学习资源", conversation_id=conversation_id)
    print("\n用户: 请给我推荐一些学习资源")
    print(f"成功: {result2.success}")
    print(f"AI: {result2.output}")
    if result2.error:
        print(f"错误: {result2.error}")
    print()


async def provider_info_example():
    """供应商信息查看"""
    print("=" * 50)
    print("8. 供应商信息查看")
    print("=" * 50)

    provider_name = "google"

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
    conversation_id = "gemini-demo-conversation"

    # 添加消息
    await manager.add_message(conversation_id, "user", "Hello Gemini!")
    await manager.add_message(conversation_id, "assistant", "Hi! I'm Gemini, Google's AI assistant.")
    await manager.add_message(conversation_id, "user", "What can you do?")

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


async def reasoning_example():
    """推理能力示例"""
    print("=" * 50)
    print("10. 推理能力示例")
    print("=" * 50)

    ai = create_ai(provider="google", model="gemini-2.5-flash")

    question = """
    一个经典的逻辑推理题：
    有三个盒子，一个装着苹果，一个装着橘子，一个装着苹果和橘子的混合。
    三个盒子上都贴有标签，但所有标签都贴错了。
    你只能从其中一个盒子里拿出一个水果，然后确定所有盒子的内容。
    请问应该如何操作？为什么？
    """

    print(f"用户: {question.strip()}")
    print("\nAI:")

    response = await ai.chat(question)
    print(response)
    print()


async def creative_writing_example():
    """创意写作示例 - Gemini 的创意能力"""
    print("=" * 50)
    print("11. 创意写作示例")
    print("=" * 50)

    ai = create_ai(provider="google", model="gemini-2.5-flash")

    prompt = "请写一个关于 AI 与人类成为朋友的短故事，大约 200 字，风格温馨。"

    print(f"用户: {prompt}")
    print("\nAI:")

    response = await ai.chat(prompt)
    print(response)
    print()


async def main():
    """运行所有示例"""
    print("\n" + "=" * 50)
    print("Google Gemini LLM 使用示例")
    print("=" * 50 + "\n")

    # 运行所有示例
    await basic_chat_example()
    await stream_chat_example()
    await code_generation_example()
    await multimodal_example()
    await agent_example()
    await tool_example()
    await conversation_example()
    await provider_info_example()
    await context_management_example()
    await reasoning_example()
    await creative_writing_example()

    print("=" * 50)
    print("所有示例运行完成!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
