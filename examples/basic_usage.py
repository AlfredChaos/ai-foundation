# AI基座使用示例
# 展示如何快速集成AI能力

import asyncio
from src import (
    create_ai,
    get_llm_provider,
    ReActAgent,
    ConversationalAgent,
    AgentConfig,
    AgentType,
    ToolManager,
    BuiltinTools,
    ContextManager,
)


async def basic_chat_example():
    """基础对话示例"""
    print("=" * 50)
    print("1. 基础对话示例")
    print("=" * 50)
    
    # 创建AI基座
    ai = create_ai(provider="openai", model="gpt-4o")
    
    # 简单对话
    response = await ai.chat("你好，请介绍一下你自己")
    print(f"用户: 你好，请介绍一下你自己")
    print(f"AI: {response}")
    print()


async def stream_chat_example():
    """流式对话示例"""
    print("=" * 50)
    print("2. 流式对话示例")
    print("=" * 50)
    
    ai = create_ai(provider="openai", model="gpt-4o")
    
    print("用户: 请写一首关于春天的诗")
    print("AI: ", end="", flush=True)
    
    async for chunk in ai.stream("请写一首关于春天的诗"):
        print(chunk, end="", flush=True)
    
    print("\n")


async def agent_example():
    """Agent执行示例"""
    print("=" * 50)
    print("3. Agent执行示例")
    print("=" * 50)
    
    # 创建Agent
    agent = ReActAgent(AgentConfig(
        name="research-agent",
        agent_type=AgentType.REACT,
        system_prompt="You are a helpful research assistant.",
        model="gpt-4o",
        max_iterations=5,
    ))
    
    # 执行任务
    result = await agent.execute("What is the capital of France?")
    
    print(f"任务: What is the capital of France?")
    print(f"成功: {result.success}")
    print(f"结果: {result.output}")
    print()


async def tool_example():
    """工具使用示例"""
    print("=" * 50)
    print("4. 工具使用示例")
    print("=" * 50)
    
    # 创建工具管理器
    tool_manager = ToolManager()
    BuiltinTools.register_builtins(tool_manager)
    
    # 使用计算器
    result = await tool_manager.execute_tool("calculator", expression="2 + 3 * 4")
    print(f"计算 2 + 3 * 4 = {result}")
    
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
    print("5. 多轮对话示例")
    print("=" * 50)
    
    agent = ConversationalAgent(AgentConfig(
        name="assistant",
        agent_type=AgentType.CONVERSATIONAL,
        model="gpt-4o",
    ))
    
    conversation_id = "user-123"
    
    # 第一轮
    result1 = await agent.execute("我叫张三", conversation_id=conversation_id)
    print(f"用户: 我叫张三")
    print(f"AI: {result1.output}")
    
    # 第二轮（Agent会记住名字）
    result2 = await agent.execute("我的名字是什么？", conversation_id=conversation_id)
    print(f"\n用户: 我的名字是什么？")
    print(f"AI: {result2.output}")
    print()


async def provider_switching_example():
    """供应商切换示例"""
    print("=" * 50)
    print("6. 供应商切换示例")
    print("=" * 50)
    
    providers = ["openai", "anthropic", "deepseek"]
    
    for provider_name in providers:
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
    print("7. 上下文管理示例")
    print("=" * 50)
    
    manager = ContextManager()
    conversation_id = "demo-conversation"
    
    # 添加消息
    await manager.add_message(conversation_id, "user", "Hello")
    await manager.add_message(conversation_id, "assistant", "Hi there!")
    await manager.add_message(conversation_id, "user", "How are you?")
    
    # 获取历史
    messages = await manager.get_messages(conversation_id)
    print(f"对话消息数: {len(messages)}")
    
    # 计算Token
    token_count = manager.calculate_tokens(messages)
    print(f"Token数: {token_count}")
    
    # 获取上下文信息
    info = manager.get_context_info(conversation_id)
    print(f"上下文信息: {info}")
    print()


async def main():
    """运行所有示例"""
    print("\n" + "=" * 50)
    print("AI Foundation 使用示例")
    print("=" * 50 + "\n")
    
    # 运行所有示例
    await basic_chat_example()
    await stream_chat_example()
    await agent_example()
    await tool_example()
    await conversation_example()
    await provider_switching_example()
    await context_management_example()
    
    print("=" * 50)
    print("所有示例运行完成!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
