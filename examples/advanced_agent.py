# Agent示例
# 展示不同类型Agent的使用方法

import asyncio
from src import (
    ReActAgent,
    SimpleReActAgent,
    ConversationalAgent,
    AgentConfig,
    AgentType,
    AgentResult,
    ToolManager,
    BuiltinTools,
)


async def react_agent_detailed():
    """详细ReAct Agent示例"""
    print("=" * 60)
    print("ReAct Agent - 详细示例")
    print("=" * 60)
    
    # 创建Agent配置
    config = AgentConfig(
        name="math-agent",
        agent_type=AgentType.REACT,
        system_prompt="""You are a mathematical reasoning agent.
You excel at solving math problems step by step.
Always show your work and explain each step.""",
        model="gpt-4o",
        temperature=0.3,  # 较低温度以获得确定性结果
        max_iterations=10,
    )
    
    # 创建Agent
    agent = ReActAgent(config)
    
    # 任务：解决数学问题
    task = "If a train travels 120 km in 2 hours, what is its average speed in m/s?"
    
    print(f"\n任务: {task}")
    print("-" * 40)
    
    # 执行
    result = await agent.execute(task)
    
    print(f"成功: {result.success}")
    print(f"输出: {result.output}")
    print(f"迭代次数: {len(result.intermediate_steps)}")
    print(f"预估Token: {result.tokens_used}")
    
    if result.error:
        print(f"错误: {result.error}")


async def simple_react_agent():
    """简单ReAct Agent示例"""
    print("\n" + "=" * 60)
    print("Simple ReAct Agent - 简单示例")
    print("=" * 60)
    
    # 创建简单Agent
    agent = SimpleReActAgent(name="simple", model="gpt-4o")
    
    task = "Explain quantum computing in simple terms"
    
    print(f"\n任务: {task}")
    print("-" * 40)
    
    result = await agent.execute(task)
    
    print(f"成功: {result.success}")
    print(f"输出: {result.output}")


async def conversational_agent_with_history():
    """带历史管理的对话Agent"""
    print("\n" + "=" * 60)
    print("Conversational Agent - 对话示例")
    print("=" * 60)
    
    # 创建对话Agent
    agent = ConversationalAgent(
        AgentConfig(
            name="chatbot",
            agent_type=AgentType.CONVERSATIONAL,
            model="gpt-4o",
        )
    )
    
    conversation_id = "user-session-001"
    
    print("\n开始多轮对话:")
    print("-" * 40)
    
    # 对话轮次
    conversations = [
        "Hi, I'm looking for restaurant recommendations.",
        "What type of cuisine do you prefer?",
        "How about Italian food?",
        "Can you suggest a specific restaurant?",
    ]
    
    for user_msg in conversations:
        result = await agent.execute(user_msg, conversation_id=conversation_id)
        print(f"\n用户: {user_msg}")
        print(f"AI: {result.output}")
        print(f"Token使用: {result.tokens_used}")
    
    # 获取Agent信息
    info = agent.get_info()
    print(f"\nAgent信息: {info}")


async def agent_with_tools():
    """带工具的Agent"""
    print("\n" + "=" * 60)
    print("Agent with Tools - 工具集成示例")
    print("=" * 60)
    
    # 创建工具管理器并注册内置工具
    tool_manager = ToolManager()
    BuiltinTools.register_builtins(tool_manager)
    
    # 创建带工具的Agent配置
    config = AgentConfig(
        name="tool-agent",
        agent_type=AgentType.REACT,
        model="gpt-4o",
        tools=tool_manager.list_tools(),
        max_iterations=5,
    )
    
    # 创建Agent
    agent = ReActAgent(config)
    
    task = "Calculate 15 * 25 and get the current date"
    
    print(f"\n任务: {task}")
    print("-" * 40)
    
    # 使用自定义工具调用器
    async def custom_tool_caller(tool_name: str, params: dict):
        return await tool_manager.execute_tool(tool_name, **params)
    
    result = await agent.execute(task, tool_caller=custom_tool_caller)
    
    print(f"成功: {result.success}")
    print(f"输出: {result.output}")
    
    # 打印执行轨迹
    trace = agent.get_execution_trace()
    print(f"\n执行步骤数: {len(trace)}")
    for step in trace:
        print(f"  Step {step.step_number}: {step.thought[:50]}...")


async def custom_agent_subclass():
    """自定义Agent子类示例"""
    print("\n" + "=" * 60)
    print("Custom Agent - 自定义Agent示例")
    print("=" * 60)
    
    from src.agents.base import BaseAgent, AgentConfig, AgentResult, AgentType
    
    class CustomAgent(BaseAgent):
        """自定义Agent - 简单的规则匹配Agent"""
        
        def __init__(self, config: AgentConfig):
            super().__init__(config)
            self.rules = {
                "hello": "Hello! How can I help you today?",
                "time": "I can tell you the current time.",
                "help": "I'm here to assist you!",
            }
        
        async def execute(self, task: str, **kwargs) -> AgentResult:
            """执行任务"""
            task_lower = task.lower()
            
            for keyword, response in self.rules.items():
                if keyword in task_lower:
                    return AgentResult(
                        success=True,
                        output=response,
                        tokens_used=len(task) // 4,
                    )
            
            return AgentResult(
                success=True,
                output="I received your message but don't have a specific response.",
                tokens_used=len(task) // 4,
            )
        
        async def plan(self, task: str) -> list:
            """制定计划"""
            return ["Receive task", "Match against rules", "Return response"]
        
        async def evaluate(self, result: Any) -> bool:
            """评估结果"""
            return isinstance(result, AgentResult) and result.success
    
    # 使用自定义Agent
    agent = CustomAgent(AgentConfig(
        name="custom",
        agent_type=AgentType.CONVERSATIONAL,
    ))
    
    tests = ["hello there", "what time is it", "help me please"]
    
    print("\n测试自定义规则:")
    for test in tests:
        result = await agent.execute(test)
        print(f"  '{test}' -> {result.output}")


async def multi_turn_reasoning():
    """多步推理示例"""
    print("\n" + "=" * 60)
    print("Multi-step Reasoning - 多步推理示例")
    print("=" * 60)
    
    agent = ReActAgent(AgentConfig(
        name="reasoner",
        agent_type=AgentType.REACT,
        model="gpt-4o",
        max_iterations=15,
        temperature=0.5,
    ))
    
    complex_task = """
    A company has 500 employees. 
    60% are in sales, 25% are in engineering, 
    and the rest are in administration. 
    If 20% of sales and 40% of engineering employees 
    work remotely, how many employees work on-site?
    """
    
    print(f"\n复杂任务:")
    print(complex_task.strip())
    print("-" * 40)
    
    result = await agent.execute(complex_task)
    
    print(f"成功: {result.success}")
    print(f"\n最终答案: {result.output}")
    print(f"\n推理步骤:")
    for i, step in enumerate(result.intermediate_steps, 1):
        thought = step.get("thought", "")[:100]
        action = step.get("action")
        observation = step.get("observation", "")[:100] if step.get("observation") else ""
        print(f"  {i}. {thought}...")
        if action:
            print(f"     Action: {action}")
        if observation:
            print(f"     Observation: {observation}...")


async def main():
    """运行所有Agent示例"""
    print("\n" + "#" * 60)
    print("# AI Foundation - Agent Examples")
    print("#" * 60 + "\n")
    
    await react_agent_detailed()
    await simple_react_agent()
    await conversational_agent_with_history()
    await agent_with_tools()
    await custom_agent_subclass()
    await multi_turn_reasoning()
    
    print("\n" + "#" * 60)
    print("# All Examples Completed!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())
