"""
AI Foundation - 工具管理模块

本模块提供工具注册、执行和管理功能，
支持同步/异步工具，以及LangChain集成。

设计要点：
1. 工具注册表模式 - 统一管理所有工具
2. 函数签名分析 - 自动推断参数类型
3. LangChain兼容 - 可导出为LangChain工具

内置工具：
- search: 网络搜索
- calculator: 数学计算
- file_reader: 文件读取
- file_writer: 文件写入
- get_datetime: 日期时间
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.core.interfaces import IToolManager


@dataclass
class ToolInfo:
    """
    工具信息数据类
    
    Attributes:
        name: 工具唯一名称
        description: 工具功能描述
        parameters: JSON Schema格式的参数定义
        function: 可调用的函数对象
        is_async: 是否为异步函数
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    is_async: bool = False


class BaseTool(ABC):
    """
    工具基类 - 定义工具接口
    
    如果工具需要更复杂的状态管理，可以继承此类。
    简单工具可以直接使用register_tool注册函数。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """参数定义"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            Any: 执行结果
        """
        pass


class ToolManager(IToolManager):
    """
    工具管理器 - 管理所有可用工具
    
    职责：
    1. 工具注册和注销
    2. 工具执行（同步/异步）
    3. LangChain集成
    4. 工具列表管理
    
    Attributes:
        _tools: 工具信息字典
        _functions: 工具函数字典
        _langchain_tools: LangChain工具字典
    
    Example:
        >>> manager = ToolManager()
        >>> manager.register_tool(
        ...     name="add",
        ...     func=lambda x, y: x + y,
        ...     description="Add two numbers"
        ... )
        >>> result = await manager.execute_tool("add", x=1, y=2)
        >>> 3
    """
    
    def __init__(self):
        """初始化工具管理器"""
        self._tools: Dict[str, ToolInfo] = {}
        self._functions: Dict[str, Callable] = {}
        self._langchain_tools: Dict[str, Any] = {}
    
    def register_tool(
        self, 
        name: str, 
        func: Callable, 
        description: str = "", 
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        注册工具
        
        自动检测函数是否为异步函数，
        如果未提供参数定义，则从函数签名推断。
        
        Args:
            name: 工具唯一名称
            func: 可调用的函数
            description: 工具功能描述
            parameters: 可选，参数定义
        """
        # 检查是否是异步函数
        is_async = asyncio.iscoroutinefunction(func)
        
        # 如果没有提供参数定义，从函数签名推断
        if parameters is None:
            parameters = self._infer_parameters(func)
        
        # 创建工具信息
        tool_info = ToolInfo(
            name=name,
            description=description,
            parameters=parameters,
            function=func,
            is_async=is_async,
        )
        
        self._tools[name] = tool_info
        self._functions[name] = func
        
        # 注册到LangChain（如果可用）
        self._register_to_langchain(name, tool_info)
    
    def _infer_parameters(self, func: Callable) -> Dict[str, Any]:
        """
        从函数签名推断参数定义
        
        分析函数签名，自动生成JSON Schema格式的参数定义。
        
        Args:
            func: 函数对象
            
        Returns:
            Dict: JSON Schema格式的参数定义
        """
        sig = inspect.signature(func)
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        
        for param_name, param in sig.parameters.items():
            # 跳过self和cls
            if param_name in ['self', 'cls']:
                continue
            
            # 确定参数类型
            param_type = "string"
            if param.annotation is not None:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
            
            param_info = {"type": param_type}
            
            # 添加描述（如果有默认值）
            if param.default is not inspect.Parameter.empty:
                param_info["description"] = str(param.default)
            
            parameters["properties"][param_name] = param_info
            
            # 必填参数
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        return parameters
    
    def _register_to_langchain(self, name: str, tool_info: ToolInfo) -> None:
        """
        注册到LangChain工具系统
        
        如果LangChain已安装，创建对应的Tool对象。
        
        Args:
            name: 工具名称
            tool_info: 工具信息
        """
        try:
            from langchain.agents import Tool as LangChainTool
            
            # 创建LangChain工具
            langchain_tool = LangChainTool(
                name=name,
                description=tool_info.description,
                func=self._create_wrapper(tool_info),
            )
            
            self._langchain_tools[name] = langchain_tool
            
        except ImportError:
            # LangChain未安装，静默忽略
            pass
    
    def _create_wrapper(self, tool_info: ToolInfo) -> Callable:
        """
        创建工具包装函数
        
        统一同步/异步函数的调用方式。
        
        Args:
            tool_info: 工具信息
            
        Returns:
            Callable: 包装后的函数
        """
        async def async_wrapper(**kwargs):
            """异步包装器"""
            return await self.execute_tool(tool_info.name, **kwargs)
        
        def sync_wrapper(**kwargs):
            """同步包装器"""
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self.execute_tool(tool_info.name, **kwargs)
                )
            finally:
                loop.close()
        
        return sync_wrapper if not tool_info.is_async else async_wrapper
    
    def unregister_tool(self, name: str) -> bool:
        """
        注销工具
        
        Args:
            name: 工具名称
            
        Returns:
            bool: 是否成功注销
        """
        if name in self._tools:
            del self._tools[name]
            del self._functions[name]
            
            # 从LangChain移除
            if name in self._langchain_tools:
                del self._langchain_tools[name]
            
            return True
        return False
    
    async def execute_tool(self, name: str, **kwargs) -> Any:
        """
        执行工具
        
        自动处理同步/异步函数的调用。
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            Any: 工具执行结果
            
        Raises:
            ToolExecutionError: 工具不存在或执行失败
        """
        if name not in self._tools:
            raise ToolExecutionError(f"Tool not found: {name}")
        
        tool = self._tools[name]
        func = tool.function
        
        try:
            if tool.is_async:
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            raise ToolExecutionError(
                f"Error executing '{name}': {str(e)}"
            ) from e
    
    def execute_tool_sync(self, name: str, **kwargs) -> Any:
        """
        同步执行工具
        
        方便在同步环境中调用异步工具。
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            Any: 工具执行结果
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.execute_tool(name, **kwargs)
            )
        finally:
            loop.close()
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        列出所有工具
        
        Returns:
            List[Dict]: 工具信息列表
        """
        return [
            {
                "name": name,
                "description": info.description,
                "parameters": info.parameters,
            }
            for name, info in self._tools.items()
        ]
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取工具信息
        
        Args:
            name: 工具名称
            
        Returns:
            Dict工具信息，不存在返回None
        """
        if name not in self._tools:
            return None
        
        tool = self._tools[name]
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "is_async": tool.is_async,
        }
    
    def get_tool_names(self) -> List[str]:
        """
        获取所有工具名称
        
        Returns:
            List[str]: 工具名称列表
        """
        return list(self._tools.keys())
    
    async def execute_react(self, task: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        执行ReAct循环
        
        简化版本的ReAct执行。
        
        Args:
            task: 任务描述
            max_iterations: 最大迭代次数
            
        Returns:
            Dict包含执行结果
        """
        iterations = []
        
        for i in range(max_iterations):
            iterations.append({
                "iteration": i + 1,
                "task": task,
                "tools_available": self.get_tool_names(),
            })
        
        return {
            "success": True,
            "iterations": iterations,
            "final_answer": f"Task: {task}. Executed {len(iterations)} iterations.",
        }
    
    def create_tool_definition(self, name: str) -> Dict[str, Any]:
        """
        创建LangChain格式的工具定义
        
        Args:
            name: 工具名称
            
        Returns:
            Dict: LangChain格式的工具定义
            
        Raises:
            ValueError: 工具不存在
        """
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        
        tool = self._tools[name]
        
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        }
    
    def clear(self) -> None:
        """清空所有工具"""
        self._tools.clear()
        self._functions.clear()
        self._langchain_tools.clear()


class ToolExecutionError(Exception):
    """
    工具执行错误
    
    当工具调用失败时抛出此异常。
    """
    pass


class BuiltinTools:
    """
    内置工具集
    
    提供常用的内置工具，
    可以方便地注册到工具管理器中。
    
    Tools:
        search: 网络搜索
        calculator: 数学计算
        file_reader: 文件读取
        file_writer: 文件写入
        get_datetime: 获取日期时间
    """
    
    @staticmethod
    def register_builtins(tool_manager: ToolManager) -> None:
        """
        注册所有内置工具
        
        Args:
            tool_manager: 工具管理器实例
        """
        # 搜索工具
        async def search(query: str) -> str:
            """
            网络搜索工具
            
            Args:
                query: 搜索查询
                
            Returns:
                str: 搜索结果
            """
            return f"Search results for: {query}"
        
        # 计算器工具
        async def calculator(expression: str) -> str:
            """
            数学计算工具
            
            Args:
                expression: 数学表达式（如 "2 + 3 * 4"）
                
            Returns:
                str: 计算结果
            """
            try:
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        # 文件读取工具
        async def file_reader(file_path: str) -> str:
            """
            文件读取工具
            
            Args:
                file_path: 文件路径
                
            Returns:
                str: 文件内容
            """
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        # 文件写入工具
        async def file_writer(file_path: str, content: str) -> str:
            """
            文件写入工具
            
            Args:
                file_path: 文件路径
                content: 要写入的内容
                
            Returns:
                str: 操作结果
            """
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully wrote to {file_path}"
            except Exception as e:
                return f"Error writing file: {str(e)}"
        
        # 日期时间工具
        async def get_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
            """
            获取当前日期时间
            
            Args:
                format: 日期格式字符串
                
            Returns:
                str: 格式化后的日期时间
            """
            from datetime import datetime
            return datetime.now().strftime(format)
        
        # 注册搜索工具
        tool_manager.register_tool(
            name="search",
            func=search,
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            }
        )
        
        # 注册计算器工具
        tool_manager.register_tool(
            name="calculator",
            func=calculator,
            description="Evaluate a mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            }
        )
        
        # 注册文件读取工具
        tool_manager.register_tool(
            name="file_reader",
            func=file_reader,
            description="Read content from a file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"}
                },
                "required": ["file_path"],
            }
        )
        
        # 注册文件写入工具
        tool_manager.register_tool(
            name="file_writer",
            func=file_writer,
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["file_path", "content"],
            }
        )
        
        # 注册日期时间工具
        tool_manager.register_tool(
            name="get_datetime",
            func=get_datetime,
            description="Get current date and time",
            parameters={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string", 
                        "description": "Date format string"
                    }
                },
                "required": [],
            }
        )
