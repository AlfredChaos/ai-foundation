# [Input] AgentConfig、任务文本与 LLM/工具调用依赖。
# [Output] 提供可执行的 ReAct（思考-行动-观察）循环结果。
# [Pos] agents 层 ReAct Agent 实现，负责响应解析与迭代控制。

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.agents.base import BaseAgent, AgentConfig, AgentResult, AgentType
from src.core.interfaces import Message, MessageRole
from src.providers.llm.factory import ILLMProvider


@dataclass
class ReactStep:
    """ReAct步骤"""
    step_number: int
    thought: str
    action: Optional[str]
    action_input: Optional[Dict[str, Any]]
    observation: Optional[str]
    is_final: bool = False


class ReActAgent(BaseAgent):
    """ReAct Agent - 实现思考-行动-观察循环"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.agent_type = AgentType.REACT
        self._steps: List[ReactStep] = []
    
    def _get_default_system_prompt(self) -> str:
        """获取ReAct Agent默认系统提示"""
        return f"""You are {self.name}, an AI assistant that uses the ReAct (Reasoning and Acting) pattern.

## Your Approach

You solve problems through a structured reasoning process:

1. **THOUGHT**: Analyze the problem and plan your next action
2. **ACTION**: Decide on a tool to use (if needed)
3. **ACTION INPUT**: Provide the input for the tool
4. **OBSERVATION**: Review the result of your action

Repeat this cycle until you reach a final answer.

## Rules

- Think carefully before each action
- If an action fails, try a different approach
- Use tools only when necessary
- Provide clear, concise observations
- Always conclude with a final answer when done

## Available Tools

{self._get_tools_description()}

## Output Format

When you need to use a tool, respond with:
```
THOUGHT: [Your reasoning]
ACTION: [tool_name]
ACTION INPUT: {{"param1": "value1", "param2": "value2"}}
```

When you have the final answer:
```
THOUGHT: [Your reasoning]
ACTION: none
ACTION INPUT: {{}}
```

Remember to think step by step and use the tools effectively."""
    
    def _get_tools_description(self) -> str:
        """获取工具描述"""
        if not self.config.tools:
            return "No tools available."
        
        descriptions = []
        for tool in self.config.tools:
            desc = f"- **{tool['name']}**: {tool.get('description', 'No description')}"
            if 'parameters' in tool:
                params = tool['parameters'].get('properties', {})
                param_list = [f"`{k}`" for k in params.keys()]
                if param_list:
                    desc += f" (params: {', '.join(param_list)})"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    async def execute(self, task: str, **kwargs) -> AgentResult:
        """执行ReAct循环"""
        self._steps = []
        intermediate_steps = []
        
        try:
            # 初始化：添加任务到历史
            self._update_history(MessageRole.USER, task)
            
            # 准备工具调用函数
            tool_caller = kwargs.get('tool_caller', self._default_tool_caller)
            
            # 执行ReAct循环
            for iteration in range(self.config.max_iterations):
                # 生成下一步
                step = await self._generate_next_step(task, iteration)
                self._steps.append(step)
                
                # 记录中间步骤
                intermediate_steps.append({
                    "step": iteration + 1,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                })
                
                # 如果是最终步骤，完成
                if step.is_final:
                    output = step.observation or step.thought
                    return AgentResult(
                        success=True,
                        output=output,
                        intermediate_steps=intermediate_steps,
                        tokens_used=self._estimate_tokens(intermediate_steps),
                    )
                
                # 执行动作
                if step.action and step.action != "none":
                    if not self.config.tools:
                        # 无工具可用时，避免模型反复请求动作导致无穷迭代。
                        return AgentResult(
                            success=True,
                            output=step.thought,
                            intermediate_steps=intermediate_steps,
                            tokens_used=self._estimate_tokens(intermediate_steps),
                        )
                    observation = await tool_caller(step.action, step.action_input or {})
                    step.observation = observation
                    
                    # 添加观察到历史
                    self._update_history(
                        MessageRole.TOOL,
                        f"Observation: {observation}"
                    )
            
            # 达到最大迭代次数
            return AgentResult(
                success=False,
                output="Maximum iterations reached",
                intermediate_steps=intermediate_steps,
                error="Exceeded maximum iterations",
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                intermediate_steps=intermediate_steps,
                error=str(e),
            )
    
    async def _generate_next_step(self, task: str, iteration: int) -> ReactStep:
        """生成下一步"""
        # 构建上下文
        context = self._build_context(iteration)
        
        # 创建请求
        request = self._create_request(context)
        
        # 调用LLM
        llm = self._get_llm_provider()
        response = await llm.generate(request)
        
        # 解析响应
        return self._parse_response(response.content, iteration)
    
    def _build_context(self, iteration: int) -> str:
        """构建上下文"""
        context_parts = [
            f"**Task**: {self._conversation_history[0].content}",
            f"**Iteration**: {iteration + 1}/{self.config.max_iterations}",
        ]
        
        # 添加之前的步骤
        if self._steps:
            context_parts.append("**Previous Steps**:")
            for step in self._steps:
                context_parts.append(f"- Step {step.step_number}: {step.thought[:100]}...")
                if step.observation:
                    context_parts.append(f"  Observation: {step.observation[:100]}...")
        
        return "\n".join(context_parts)
    
    def _parse_response(self, response: str, step_number: int) -> ReactStep:
        """解析LLM响应"""
        thought = ""
        action = None
        action_input = None
        is_final = False
        has_action_field = False
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            normalized_key = key.strip().lower()
            normalized_value = value.strip()

            if normalized_key == "thought":
                thought = normalized_value
            elif normalized_key == "action":
                has_action_field = True
                action_value = normalized_value.lower()
                if action_value in ["none", "final", "finish", "final_answer"] or action_value == "":
                    is_final = True
                    action = "none"
                else:
                    action = action_value
            elif normalized_key == "action input":
                try:
                    import json
                    action_input = json.loads(normalized_value)
                except Exception:
                    action_input = {}
        
        # 如果没有解析到thought，使用完整响应
        if not thought:
            thought = response.strip()

        # 模型未输出 ACTION 字段时，将其视为最终答案，避免无意义迭代。
        if not has_action_field:
            is_final = True
            action = "none"
        
        return ReactStep(
            step_number=step_number + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=None,
            is_final=is_final,
        )
    
    async def _default_tool_caller(self, tool_name: str, params: Dict[str, Any]) -> str:
        """默认工具调用器"""
        tool_manager = self._get_tool_manager()
        
        try:
            result = await tool_manager.execute_tool(tool_name, **params)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    async def plan(self, task: str) -> List[str]:
        """制定执行计划"""
        # 简化版本：返回任务的主要步骤
        return [
            f"Analyze task: {task}",
            "Determine if tools are needed",
            "Execute reasoning-action-observation loop",
            "Formulate final answer",
        ]
    
    async def evaluate(self, result: Any) -> bool:
        """评估执行结果"""
        if isinstance(result, AgentResult):
            return result.success and len(result.intermediate_steps) > 0
        return False
    
    def get_execution_trace(self) -> List[ReactStep]:
        """获取执行轨迹"""
        return self._steps.copy()
    
    def _estimate_tokens(self, steps: List[Dict]) -> int:
        """估算使用的Token数"""
        # 粗略估算
        return len(steps) * 500
    
    def _get_purpose(self) -> str:
        """获取Agent目的"""
        return "solve complex tasks through reasoning and action"


class SimpleReActAgent(ReActAgent):
    """简化版ReAct Agent - 适用于简单场景"""
    
    def __init__(self, name: str = "SimpleReAct", model: str = "gpt-4o"):
        config = AgentConfig(
            name=name,
            agent_type=AgentType.REACT,
            system_prompt="You are a helpful assistant that thinks step by step.",
            model=model,
            max_iterations=5,
        )
        super().__init__(config)
    
    def _parse_response(self, response: str, step_number: int) -> ReactStep:
        """简化版解析 - 假设响应就是thought"""
        thought = response.strip()
        # 检查是否包含最终答案的标记
        is_final = "final answer" in thought.lower() or "答案是" in thought
        
        return ReactStep(
            step_number=step_number + 1,
            thought=thought,
            action=None,
            action_input=None,
            observation=None,
            is_final=is_final,
        )
