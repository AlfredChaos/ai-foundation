# [Input] 项目根目录下的 `src` 包与运行时配置（如 API Key）。
# [Output] 提供可执行的 OpenRouter LLM 使用示例入口，包含多模态图片识别。
# [Pos] examples 层 OpenRouter 示例脚本，兼容直接运行与模块方式运行。

import asyncio
import sys
import base64
from pathlib import Path
from typing import List, Dict, Any

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
    from src.core.interfaces import AIRequest, Message, MessageRole
except ModuleNotFoundError as exc:
    if exc.name != "src":
        raise
    # 兼容 `python examples/openrouter_usage.py` 直接运行场景
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
    from src.core.interfaces import AIRequest, Message, MessageRole


def _encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def basic_chat_example():
    """基础对话示例"""
    print("=" * 50)
    print("1. 基础对话示例")
    print("=" * 50)

    # 创建AI基座
    ai = create_ai(provider="openrouter", model="anthropic/claude-3.5-sonnet")

    # 简单对话
    response = await ai.chat("你好，请用中文介绍一下 OpenRouter 是什么")
    print(f"用户: 你好，请用中文介绍一下 OpenRouter 是什么")
    print(f"AI: {response}")
    print()


async def stream_chat_example():
    """流式对话示例"""
    print("=" * 50)
    print("2. 流式对话示例")
    print("=" * 50)

    ai = create_ai(provider="openrouter", model="anthropic/claude-3.5-sonnet")

    print("用户: 请写一首关于科技的诗")
    print("AI: ", end="", flush=True)

    async for chunk in ai.stream("请写一首关于科技的诗"):
        print(chunk, end="", flush=True)

    print("\n")


async def image_recognition_example():
    """图片内容识别示例 - 使用多模态模型"""
    print("=" * 50)
    print("3. 图片内容识别示例 (多模态)")
    print("=" * 50)

    provider = get_llm_provider("openrouter")

    # 示例图片 URL (使用公开的测试图片)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"

    # 构建多模态消息内容
    multimodal_content = [
        {
            "type": "text",
            "text": "请详细描述这张图片的内容，包括你看到的形状、颜色和任何文字。"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }
    ]

    request = AIRequest(
        model="anthropic/claude-3.5-sonnet",  # 支持视觉的多模态模型
        messages=[Message(role=MessageRole.USER, content=str(multimodal_content))],
    )

    print(f"用户: 请描述图片内容")
    print(f"图片 URL: {image_url}")
    print("\nAI:")

    try:
        # 直接使用 provider 的 API 调用以支持复杂消息格式
        response = await provider._get_async_client()
        if response:
            api_response = await response.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": multimodal_content}],
                max_tokens=500
            )
            print(api_response.choices[0].message.content)
        else:
            print("无法连接到 OpenRouter 服务")
    except Exception as e:
        print(f"图片识别出错: {e}")
    print()


async def image_recognition_local_example():
    """本地图片识别示例 - 使用 base64 编码"""
    print("=" * 50)
    print("4. 本地图片识别示例 (base64)")
    print("=" * 50)

    # 检查是否有测试图片
    test_image_path = Path(__file__).parent.parent / "tests" / "fixtures" / "test_image.png"

    if not test_image_path.exists():
        print("提示: 请将测试图片放在项目根目录下的 test_image.png")
        print("      然后重新运行此示例")
        # 使用一个简单的 SVG 作为示例
        test_image_path = Path(__file__).parent / "test_sample.svg"
        if not test_image_path.exists():
            # 创建一个简单的测试文件
            test_image_path = Path(__file__).parent / "test_sample.png"
            # 创建一个简单的 PNG 图片 (1x1 像素透明 PNG)
            import struct
            png_data = (
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
                b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            )
            with open(test_image_path, "wb") as f:
                f.write(png_data)

    provider = get_llm_provider("openrouter")

    # 将本地图片编码为 base64
    base64_image = _encode_image_to_base64(str(test_image_path))

    multimodal_content = [
        {
            "type": "text",
            "text": "这是一个测试图片，请告诉我你看到了什么。"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        }
    ]

    print(f"用户: 这是一个测试图片，请告诉我你看到了什么。")
    print(f"图片: {test_image_path.name} (base64 编码)")
    print("\nAI:")

    try:
        client = await provider._get_async_client()
        if client:
            api_response = await client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": multimodal_content}],
                max_tokens=500
            )
            print(api_response.choices[0].message.content)
        else:
            print("无法连接到 OpenRouter 服务")
    except Exception as e:
        print(f"图片识别出错: {e}")
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
        model="anthropic/claude-3.5-sonnet",
        max_iterations=5,
    ))

    # 执行任务
    result = await agent.execute("What is OpenRouter and how does it work?")

    print("任务: What is OpenRouter and how does it work?")
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
    result = await tool_manager.execute_tool("calculator", expression="2 ** 8 + 100")
    print(f"计算 2^8 + 100 = {result}")

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
    print("7. 多轮对话示例")
    print("=" * 50)

    agent = ConversationalAgent(AgentConfig(
        name="openrouter-assistant",
        agent_type=AgentType.CONVERSATIONAL,
        model="anthropic/claude-3.5-sonnet",
    ))

    conversation_id = "user-openrouter-001"

    # 第一轮
    result1 = await agent.execute("我对 AI 和机器学习很感兴趣", conversation_id=conversation_id)
    print("用户: 我对 AI 和机器学习很感兴趣")
    print(f"成功: {result1.success}")
    print(f"AI: {result1.output}")
    if result1.error:
        print(f"错误: {result1.error}")

    # 第二轮（AI 会记住上下文）
    result2 = await agent.execute("请推荐一些学习资源", conversation_id=conversation_id)
    print("\n用户: 请推荐一些学习资源")
    print(f"成功: {result2.success}")
    print(f"AI: {result2.output}")
    if result2.error:
        print(f"错误: {result2.error}")
    print()


async def model_switching_example():
    """模型切换示例 - OpenRouter 支持多种模型"""
    print("=" * 50)
    print("8. 模型切换示例")
    print("=" * 50)

    # OpenRouter 支持的多种模型
    models = [
        ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet - 强大的多模态模型"),
        ("google/gemini-pro-1.5", "Gemini Pro 1.5 - Google 的多模态模型"),
        ("meta-llama/llama-3.1-8b-instruct:free", "Llama 3.1 8B - 免费开源模型"),
    ]

    provider = get_llm_provider("openrouter")

    for model_id, description in models:
        print(f"\n模型: {model_id}")
        print(f"描述: {description}")
        try:
            request = AIRequest(
                model=model_id,
                messages=[Message(role=MessageRole.USER, content="用一句话介绍你自己")],
                max_tokens=100
            )
            response = await provider.generate(request)
            print(f"回复: {response.content[:100]}...")
        except Exception as e:
            print(f"错误: {e}")
    print()


async def provider_info_example():
    """供应商信息查看"""
    print("=" * 50)
    print("9. 供应商信息查看")
    print("=" * 50)

    provider_name = "openrouter"

    try:
        provider = get_llm_provider(provider_name)
        info = provider.get_model_info()
        print(f"供应商: {provider_name}")
        print(f"  可用: {info.get('available', False)}")
        print(f"  模型: {info.get('models', {})}")
        print(f"\n说明: OpenRouter 是一个 LLM 聚合服务，支持访问多个 AI 提供商的模型。")
    except Exception as e:
        print(f"供应商 {provider_name}: {e}")
    print()


async def context_management_example():
    """上下文管理示例"""
    print("=" * 50)
    print("10. 上下文管理示例")
    print("=" * 50)

    manager = ContextManager()
    conversation_id = "openrouter-demo-conversation"

    # 添加消息
    await manager.add_message(conversation_id, "user", "Hello OpenRouter!")
    await manager.add_message(conversation_id, "assistant", "Hi! I'm connected via OpenRouter.")
    await manager.add_message(conversation_id, "user", "What models can I use?")

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
    print("OpenRouter LLM 使用示例")
    print("=" * 50 + "\n")

    # 运行所有示例
    await basic_chat_example()
    await stream_chat_example()
    await image_recognition_example()  # 多模态图片识别
    await image_recognition_local_example()  # 本地图片识别
    await agent_example()
    await tool_example()
    await conversation_example()
    await model_switching_example()
    await provider_info_example()
    await context_management_example()

    print("=" * 50)
    print("所有示例运行完成!")
    print("=" * 50)
    print("\n提示: 某些示例需要有效的 OPENROUTER_API_KEY 环境变量")
    print("获取 API Key: https://openrouter.ai/keys")


if __name__ == "__main__":
    asyncio.run(main())
