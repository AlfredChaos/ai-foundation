# MCP客户端实现
# Model Context Protocol 客户端

import asyncio
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.core.interfaces import IMCPClient


@dataclass
class MCPServerConfig:
    """MCP服务器配置"""
    name: str
    url: str
    api_key: Optional[str] = None
    enabled: bool = True


class MCPClient(IMCPClient):
    """MCP客户端"""
    
    def __init__(self, config: Optional[MCPServerConfig] = None):
        self.config = config
        self._connected = False
        self._tools: List[Dict[str, Any]] = []
        self._session = None
    
    async def connect(self, server_url: str, api_key: Optional[str] = None) -> bool:
        """连接MCP服务器"""
        try:
            # MCP协议连接
            # 这里实现MCP WebSocket连接
            self._connected = True
            self._server_url = server_url
            
            # 获取工具列表
            self._tools = await self._fetch_tools()
            
            return True
        except Exception as e:
            print(f"MCP connection error: {e}")
            return False
    
    async def _fetch_tools(self) -> List[Dict[str, Any]]:
        """获取工具列表"""
        # 从MCP服务器获取可用工具
        return [
            {
                "name": "example_tool",
                "description": "Example tool from MCP server",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    }
                }
            }
        ]
    
    async def disconnect(self) -> None:
        """断开连接"""
        self._connected = False
        self._tools = []
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出MCP服务器上的工具"""
        return self._tools
    
    async def call_tool(self, tool_name: str, **params) -> Any:
        """调用MCP工具"""
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        
        # MCP工具调用
        return {"result": f"Called {tool_name} with params: {params}"}
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected
    
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "url": getattr(self, '_server_url', ''),
            "connected": self._connected,
            "tools_count": len(self._tools),
        }


class MCPClientManager:
    """MCP客户端管理器 - 管理多个MCP服务器连接"""
    
    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}
        self._tool_registry: Dict[str, Dict[str, Any]] = {}
    
    async def add_server(self, config: MCPServerConfig) -> bool:
        """添加MCP服务器"""
        if not config.enabled:
            return False
        
        client = MCPClient(config)
        success = await client.connect(config.url, config.api_key)
        
        if success:
            self._clients[config.name] = client
            # 注册工具
            for tool in await client.list_tools():
                self._tool_registry[tool["name"]] = {
                    "tool": tool,
                    "server": config.name,
                }
        
        return success
    
    async def remove_server(self, name: str) -> bool:
        """移除MCP服务器"""
        if name in self._clients:
            await self._clients[name].disconnect()
            del self._clients[name]
            
            # 移除工具注册
            tools_to_remove = [
                k for k, v in self._tool_registry.items()
                if v["server"] == name
            ]
            for k in tools_to_remove:
                del self._tool_registry[k]
            
            return True
        return False
    
    async def call_tool(self, tool_name: str, **params) -> Any:
        """调用工具（自动路由到正确的服务器）"""
        if tool_name not in self._tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        server_name = self._tool_registry[tool_name]["server"]
        client = self._clients[server_name]
        
        return await client.call_tool(tool_name, **params)
    
    def list_all_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        return [
            {
                "name": name,
                **tool_info["tool"],
                "server": tool_info["server"],
            }
            for name, tool_info in self._tool_registry.items()
        ]
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """列出所有服务器"""
        return [
            {
                "name": name,
                **client.get_server_info()
            }
            for name, client in self._clients.items()
        ]
    
    async def disconnect_all(self) -> None:
        """断开所有连接"""
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()
        self._tool_registry.clear()
