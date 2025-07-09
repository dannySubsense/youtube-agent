"""
Tool interfaces for YouTube Agent System.
Agent-agnostic wrappers for MCP tools and MVP tools integration.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from config.settings import Settings
from .mcp_client import MCPClient
from tools import VideoSearcher, TranscriptManager, VectorDatabaseManager, KnowledgeBaseManager


class ToolInterfaceError(Exception):
    """Custom exception for tool interface related errors."""
    pass


@dataclass
class ToolResult:
    """Standardized tool result format."""
    tool_name: str
    success: bool
    data: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0


class ToolInterface(ABC):
    """Abstract base class for tool interfaces."""
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information and parameters."""
        pass


class MCPToolInterface(ToolInterface):
    """Interface for MCP tools."""
    
    def __init__(self, mcp_client: MCPClient, tool_name: str):
        """
        Initialize MCP tool interface.
        
        Args:
            mcp_client: MCP client instance
            tool_name: Name of the MCP tool
        """
        self.mcp_client = mcp_client
        self.tool_name = tool_name
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute MCP tool.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            import time
            start_time = time.time()
            
            result = self.mcp_client.call_tool(self.tool_name, **kwargs)
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                data=None,
                error_message=str(e)
            )
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get MCP tool information."""
        tool_info = self.mcp_client.get_tool_info(self.tool_name)
        if tool_info:
            return {
                "name": tool_info.name,
                "description": tool_info.description,
                "parameters": tool_info.parameters,
                "type": "mcp"
            }
        return {"name": self.tool_name, "type": "mcp", "available": False}


class MVPToolInterface(ToolInterface):
    """Interface for MVP tools."""
    
    def __init__(self, tool_instance: Any, tool_name: str):
        """
        Initialize MVP tool interface.
        
        Args:
            tool_instance: MVP tool instance
            tool_name: Name of the MVP tool
        """
        self.tool_instance = tool_instance
        self.tool_name = tool_name
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute MVP tool.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            import time
            start_time = time.time()
            
            # Route to appropriate MVP tool method
            if self.tool_name == "search_videos":
                result = self.tool_instance.search_videos(**kwargs)
            elif self.tool_name == "process_transcripts":
                result = self.tool_instance.process_transcripts(**kwargs)
            elif self.tool_name == "create_vectorstore":
                result = self.tool_instance.create_vectorstore(**kwargs)
            elif self.tool_name == "chat_with_kb":
                result = self.tool_instance.chat_with_knowledge_base(**kwargs)
            else:
                raise ToolInterfaceError(f"Unknown MVP tool: {self.tool_name}")
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                data=None,
                error_message=str(e)
            )
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get MVP tool information."""
        return {
            "name": self.tool_name,
            "description": f"MVP tool: {self.tool_name}",
            "type": "mvp"
        }


class AgentToolInterface:
    """
    Unified interface for both MCP and MVP tools.
    
    Provides agent-agnostic access to all available tools:
    - 14 MCP tools for autonomous analysis
    - 4 MVP tools for batch processing
    """
    
    def __init__(self, settings: Settings, mcp_client: MCPClient):
        """
        Initialize agent tool interface.
        
        Args:
            settings: Application settings
            mcp_client: MCP client instance
        """
        self.settings = settings
        self.mcp_client = mcp_client
        self.tools: Dict[str, ToolInterface] = {}
        self._initialize_tools()
    
    def _initialize_tools(self) -> None:
        """Initialize all available tools."""
        try:
            # Initialize MCP tools
            for tool_name in self.mcp_client.get_available_tools():
                self.tools[tool_name] = MCPToolInterface(self.mcp_client, tool_name)
            
            # Initialize MVP tools
            from tools import (
                create_video_searcher,
                create_transcript_manager,
                create_vector_database_manager,
                create_knowledge_base_manager
            )
            
            video_searcher = create_video_searcher(self.settings)
            transcript_manager = create_transcript_manager(self.settings)
            vector_db_manager = create_vector_database_manager(self.settings)
            kb_manager = create_knowledge_base_manager(self.settings)
            
            self.tools["mvp_search_videos"] = MVPToolInterface(video_searcher, "search_videos")
            self.tools["mvp_process_transcripts"] = MVPToolInterface(transcript_manager, "process_transcripts")
            self.tools["mvp_create_vectorstore"] = MVPToolInterface(vector_db_manager, "create_vectorstore")
            self.tools["mvp_chat_with_kb"] = MVPToolInterface(kb_manager, "chat_with_kb")
            
        except Exception as e:
            raise ToolInterfaceError(f"Failed to initialize tools: {e}") from e
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ToolInterfaceError: If tool not found or execution fails
        """
        if tool_name not in self.tools:
            raise ToolInterfaceError(f"Tool '{tool_name}' not found")
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of all available tools.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def get_mcp_tools(self) -> List[str]:
        """
        Get list of MCP tools only.
        
        Returns:
            List of MCP tool names
        """
        return [name for name in self.tools.keys() if not name.startswith("mvp_")]
    
    def get_mvp_tools(self) -> List[str]:
        """
        Get list of MVP tools only.
        
        Returns:
            List of MVP tool names
        """
        return [name for name in self.tools.keys() if name.startswith("mvp_")]
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool information or None if not found
        """
        if tool_name not in self.tools:
            return None
        
        return self.tools[tool_name].get_tool_info()
    
    def get_tools_by_type(self, tool_type: str) -> List[str]:
        """
        Get tools by type.
        
        Args:
            tool_type: Tool type ("mcp" or "mvp")
            
        Returns:
            List of tool names of specified type
        """
        if tool_type == "mcp":
            return self.get_mcp_tools()
        elif tool_type == "mvp":
            return self.get_mvp_tools()
        else:
            return []


def create_agent_tool_interface(settings: Settings, mcp_client: MCPClient) -> AgentToolInterface:
    """
    Factory function to create agent tool interface.
    
    Args:
        settings: Application settings
        mcp_client: MCP client instance
        
    Returns:
        Configured agent tool interface
    """
    return AgentToolInterface(settings, mcp_client) 