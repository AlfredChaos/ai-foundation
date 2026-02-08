# gRPC服务实现

import asyncio
from concurrent import futures
from typing import Iterator, Dict, Any
import json
import logging

import grpc

from src.config.manager import get_config
from src.providers.llm.factory import get_llm_provider
from src.agents import ReActAgent, ConversationalAgent, AgentConfig, AgentType
from src.context.manager import ContextManager
from src.tools.tool_manager import ToolManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AICoreServicer:
    """AI Core gRPC服务实现"""
    
    def __init__(self):
        self.llm_provider = None
        self.context_manager = ContextManager()
        self.tool_manager = ToolManager()
        self._initialize()
    
    def _initialize(self):
        """初始化"""
        try:
            self.llm_provider = get_llm_provider("openai")
            logger.info("LLM provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM provider: {e}")
    
    async def _convert_messages(self, messages: list) -> list:
        """转换消息格式"""
        from src.core.interfaces import Message, MessageRole
        
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            converted.append(Message(
                role=MessageRole(role),
                content=msg.get("content", ""),
                name=msg.get("name"),
            ))
        return converted
    
    async def SingleCall(self, request, context):
        """单次调用"""
        try:
            messages = await self._convert_messages(request.messages)
            
            from src.core.interfaces import AIRequest, Message
            
            ai_request = AIRequest(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens if request.max_tokens > 0 else None,
                json_mode=request.json_mode,
            )
            
            response = await self.llm_provider.generate(ai_request)
            
            return {
                "content": response.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "model": response.model,
                "finish_reason": response.finish_reason,
                "tool_calls": response.tool_calls or [],
            }
        except Exception as e:
            logger.error(f"SingleCall error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {}
    
    async def Chat(self, request, context):
        """Chat对话"""
        try:
            messages = await self._convert_messages(request.messages)
            
            conversation_id = request.conversation_id or "default"
            
            # 添加用户消息
            user_msg = messages[-1] if messages else None
            if user_msg:
                await self.context_manager.add_message(
                    conversation_id, 
                    user_msg.role.value, 
                    user_msg.content
                )
            
            # 获取历史
            history = await self.context_manager.get_messages(conversation_id)
            
            from src.core.interfaces import AIRequest, Message
            
            ai_request = AIRequest(
                model=request.model,
                messages=history,
                temperature=request.temperature,
                max_tokens=request.max_tokens if request.max_tokens > 0 else None,
            )
            
            response = await self.llm_provider.generate(ai_request)
            
            # 添加助手回复
            await self.context_manager.add_message(
                conversation_id, 
                "assistant", 
                response.content
            )
            
            return {
                "content": response.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "conversation_id": conversation_id,
                "tool_calls": response.tool_calls or [],
            }
        except Exception as e:
            logger.error(f"Chat error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {}
    
    def StreamCall(self, request, context) -> Iterator[Dict]:
        """流式回复"""
        try:
            from src.core.interfaces import AIRequest, Message, MessageRole
            
            messages = asyncio.run(self._convert_messages(request.messages))
            
            ai_request = AIRequest(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens if request.max_tokens > 0 else None,
                stream=True,
            )
            
            loop = asyncio.new_event_loop()
            try:
                async def generate():
                    chunk_count = 0
                    async for chunk in self.llm_provider.stream_generate(ai_request):
                        chunk_count += 1
                        yield {
                            "chunk": chunk,
                            "is_final": False,
                            "usage": {},
                        }
                    
                    # 最终响应
                    yield {
                        "chunk": "",
                        "is_final": True,
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": chunk_count * 10,  # 估算
                            "total_tokens": chunk_count * 10,
                        },
                    }
                
                return generate()
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"StreamCall error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return iter([])
    
    async def ExecuteAgent(self, request, context):
        """Agent执行"""
        try:
            agent = ReActAgent(AgentConfig(
                name="grpc-agent",
                agent_type=AgentType.REACT,
                model=request.model,
                max_iterations=request.max_iterations or 10,
            ))
            
            result = await agent.execute(request.task)
            
            return {
                "success": result.success,
                "output": result.output,
                "steps": [
                    {
                        "step": i + 1,
                        "thought": step.get("thought", ""),
                        "action": step.get("action"),
                        "action_input": str(step.get("action_input", "")),
                        "observation": step.get("observation"),
                    }
                    for i, step in enumerate(result.intermediate_steps or [])
                ],
                "tokens_used": result.tokens_used,
                "error": result.error or "",
            }
        except Exception as e:
            logger.error(f"ExecuteAgent error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"success": False, "error": str(e)}
    
    async def GenerateImage(self, request, context):
        """图像生成"""
        # TODO: 实现图像生成
        return {
            "image_url": "",
            "image_base64": "",
            "provider": request.provider,
        }
    
    async def StoreMemory(self, request, context):
        """存储记忆"""
        # TODO: 实现记忆存储
        return {
            "success": True,
            "message": "Memory stored",
        }
    
    async def RetrieveMemory(self, request, context):
        """检索记忆"""
        # TODO: 实现记忆检索
        return {"results": []}
    
    async def CallTool(self, request, context):
        """工具调用"""
        try:
            params = json.loads(request.parameters) if request.parameters else {}
            result = await self.tool_manager.execute_tool(request.tool_name, **params)
            
            return {
                "success": True,
                "result": str(result),
                "error": "",
            }
        except Exception as e:
            return {
                "success": False,
                "result": "",
                "error": str(e),
            }


class HealthServicer:
    """健康检查服务"""
    
    def Check(self, request, context):
        """检查"""
        return {
            "status": 1,  # SERVING
            "version": "1.0.0",
        }
    
    def Watch(self, request, context):
        """监控"""
        while True:
            yield {
                "status": 1,
                "version": "1.0.0",
            }
            import time
            time.sleep(5)


async def serve(port: str = "50051", max_workers: int = 10):
    """启动gRPC服务器"""
    import grpc.aio
    
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    # TODO: 添加服务实现
    
    server.add_insecure_port(f'[::]:{port}')
    await server.start()
    
    logger.info(f"gRPC server started on port {port}")
    
    return server


if __name__ == "__main__":
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "50051"
    
    server = asyncio.run(serve(port))
    server.wait_for_termination()
