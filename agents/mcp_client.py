"""
MCP client for YouTube Agent System.
Handles connection to YouTube MCP Server with robust subprocess management.
"""

import subprocess
import json
import os
import asyncio
import signal
import sys
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from config.settings import Settings


class MCPClientError(Exception):
    """Custom exception for MCP client related errors."""
    pass


@dataclass
class MCPTool:
    """Representation of an MCP tool."""
    name: str
    description: str
    parameters: Dict[str, Any]


class MCPClient:
    """
    Client for connecting to YouTube MCP Server.
    
    Provides access to YouTube MCP tools for autonomous agent use:
    - get_video_details, search_videos, analyze_video_engagement
    - evaluate_video_for_knowledge_base, get_video_transcript
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize MCP client with YouTube MCP Server connection.
        
        Args:
            settings: Application settings
            
        Raises:
            MCPClientError: If initialization fails
        """
        self.settings = self._validate_settings(settings)
        self.server_path = self._find_youtube_mcp_server()
        self.server_process: Optional[subprocess.Popen] = None
        self.available_tools: List[MCPTool] = []
        self._initialize_connection()
    
    def _validate_settings(self, settings: Settings) -> Settings:
        """
        Validate settings for MCP client.
        
        Args:
            settings: Application settings
            
        Returns:
            Validated settings
            
        Raises:
            MCPClientError: If settings are invalid
        """
        if not settings:
            raise MCPClientError("Settings cannot be None")
        
        if not settings.youtube_api_key:
            raise MCPClientError("YouTube API key is required")
        
        if not settings.openai_api_key:
            raise MCPClientError("OpenAI API key is required")
        
        # Validate API key format (basic check) - allow test keys for testing
        if len(settings.youtube_api_key) < 4:
            raise MCPClientError("YouTube API key appears to be invalid (too short)")
        
        return settings
    
    def _find_youtube_mcp_server(self) -> str:
        """
        Find YouTube MCP Server location within the project.
        
        Returns:
            Path to YouTube MCP Server
            
        Raises:
            MCPClientError: If server not found
        """
        # The server is now integrated into the project structure
        server_path = Path(__file__).parent.parent / "mcp_integration" / "youtube_mcp_server.py"
        
        if not server_path.exists():
            raise MCPClientError(
                f"YouTube MCP Server not found at expected location: {server_path}"
            )
        
        # Verify server file is readable
        if not server_path.is_file():
            raise MCPClientError(f"Server path exists but is not a file: {server_path}")
        
        return str(server_path)
    
    def _initialize_connection(self) -> None:
        """Initialize connection to MCP server and discover tools."""
        try:
            # Start the MCP server process
            self._start_server()
            
            # Define the available tools based on the integrated server
            self.available_tools = [
                MCPTool(
                    "get_video_details", 
                    "Get detailed information about a YouTube video", 
                    {"video_input": {"type": "string", "description": "YouTube video URL or video ID"}}
                ),
                MCPTool(
                    "search_videos", 
                    "Search YouTube for videos by keywords", 
                    {
                        "query": {"type": "string", "description": "Search query keywords"},
                        "max_results": {"type": "integer", "description": "Maximum results (1-50)", "default": 10},
                        "order": {"type": "string", "description": "Sort order", "default": "relevance"}
                    }
                ),
                MCPTool(
                    "analyze_video_engagement", 
                    "Analyze video engagement metrics and provide insights", 
                    {"video_input": {"type": "string", "description": "YouTube video URL or video ID"}}
                ),
                MCPTool(
                    "evaluate_video_for_knowledge_base", 
                    "Evaluate video for knowledge base inclusion", 
                    {"video_input": {"type": "string", "description": "YouTube video URL or video ID"}}
                ),
                MCPTool(
                    "get_video_transcript", 
                    "Extract transcript content from YouTube videos", 
                    {
                        "video_input": {"type": "string", "description": "YouTube video URL or video ID"},
                        "language": {"type": "string", "description": "Language code", "default": "en"}
                    }
                ),
            ]
        except Exception as e:
            raise MCPClientError(f"Failed to initialize MCP connection: {e}") from e
    
    def _start_server(self) -> None:
        """Start the MCP server process with improved error handling."""
        try:
            env = self._prepare_server_environment()
            self.server_process = self._launch_server_process(env)
            self._validate_server_startup()
        except Exception as e:
            self._cleanup_failed_server()
            raise MCPClientError(f"Failed to start MCP server: {e}") from e
    
    def _prepare_server_environment(self) -> Dict[str, str]:
        """
        Prepare environment for server startup.
        
        Returns:
            Environment dictionary for subprocess
        """
        python_executable = self._find_python_executable()
        
        # Prepare environment
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent.parent)
        
        return env
    
    def _launch_server_process(self, env: Dict[str, str]) -> subprocess.Popen:
        """
        Launch the MCP server subprocess.
        
        Args:
            env: Environment variables for the process
            
        Returns:
            Started subprocess
        """
        python_executable = self._find_python_executable()
        
        return subprocess.Popen(
            [python_executable, self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered for real-time communication
            env=env,
            cwd=Path(__file__).parent.parent,  # Set working directory
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Process group for clean shutdown
        )
    
    def _validate_server_startup(self) -> None:
        """
        Validate that server started successfully.
        
        Raises:
            MCPClientError: If server failed to start
        """
        if not self.server_process:
            raise MCPClientError("Server process was not created")
        
        # Give server time to initialize
        time.sleep(2)  # Increased startup time
        
        # Check if server started successfully
        if self.server_process.poll() is not None:
            stderr_output = self._get_server_error_output()
            raise MCPClientError(f"MCP server failed to start. Error: {stderr_output}")
    
    def _get_server_error_output(self) -> str:
        """
        Get error output from failed server process.
        
        Returns:
            Error output string
        """
        if not self.server_process or not self.server_process.stderr:
            return "Unable to read error output"
        
        try:
            return self.server_process.stderr.read()
        except Exception:
            return "Unable to read error output"
    
    def _find_python_executable(self) -> str:
        """Find the appropriate Python executable."""
        # Try different Python executable names
        python_candidates = ['python3', 'python', sys.executable]
        
        for candidate in python_candidates:
            if candidate and Path(candidate).exists():
                return candidate
        
        # Fallback to system python
        return 'python3'
    
    def _cleanup_failed_server(self) -> None:
        """Clean up failed server process."""
        if self.server_process:
            try:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
            except:
                pass
            finally:
                self.server_process = None
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute MCP tool and return results with improved error handling.
        
        Args:
            tool_name: Name of the MCP tool to call
            **kwargs: Tool parameters
            
        Returns:
            Tool execution results
            
        Raises:
            MCPClientError: If tool execution fails
        """
        try:
            self._validate_tool_request(tool_name, kwargs)
            request = self._build_mcp_request(tool_name, kwargs)
            response = self._send_mcp_request(request)
            return self._parse_tool_response(response)
            
        except MCPClientError:
            # Re-raise MCP errors as-is
            raise
        except Exception as e:
            raise MCPClientError(f"MCP tool '{tool_name}' execution failed: {e}") from e
    
    def _validate_tool_request(self, tool_name: str, kwargs: Dict[str, Any]) -> None:
        """
        Validate tool request parameters.
        
        Args:
            tool_name: Name of the MCP tool to call
            kwargs: Tool parameters
            
        Raises:
            MCPClientError: If validation fails
        """
        if not tool_name:
            raise MCPClientError("Tool name cannot be empty")
        
        if not self.is_tool_available(tool_name):
            available_tools = ", ".join(self.get_available_tools())
            raise MCPClientError(
                f"Tool '{tool_name}' not available. Available tools: {available_tools}"
            )
        
        if not self.server_process or self.server_process.poll() is not None:
            raise MCPClientError(
                "MCP server is not running. Try reinitializing the client."
            )
        
        # Validate tool parameters
        self._validate_tool_parameters(tool_name, kwargs)
    
    def _build_mcp_request(self, tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build MCP request structure.
        
        Args:
            tool_name: Name of the MCP tool to call
            kwargs: Tool parameters
            
        Returns:
            MCP request dictionary
        """
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": kwargs
            }
        }
    
    def _send_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request to MCP server and get response.
        
        Args:
            request: MCP request dictionary
            
        Returns:
            MCP response dictionary
            
        Raises:
            MCPClientError: If communication fails
        """
        if not self.server_process:
            raise MCPClientError("Server process is not available")
        
        request_json = json.dumps(request) + "\n"
        
        # Send request
        if self.server_process.stdin is None:
            raise MCPClientError("Server stdin is not available")
        
        try:
            self.server_process.stdin.write(request_json)
            self.server_process.stdin.flush()
        except BrokenPipeError:
            raise MCPClientError("Server connection broken (broken pipe)")
        except Exception as e:
            raise MCPClientError(f"Failed to send request to server: {e}")
        
        # Read response
        if self.server_process.stdout is None:
            raise MCPClientError("Server stdout is not available")
        
        try:
            response_line = self.server_process.stdout.readline()
        except Exception as e:
            raise MCPClientError(f"Failed to read response from server: {e}")
        
        if not response_line:
            raise MCPClientError("No response from MCP server (connection may be lost)")
        
        try:
            return json.loads(response_line.strip())
        except json.JSONDecodeError as e:
            raise MCPClientError(f"Invalid JSON response from server: {e}")
    
    def _parse_tool_response(self, response: Dict[str, Any]) -> Any:
        """
        Parse MCP tool response and extract result.
        
        Args:
            response: MCP response dictionary
            
        Returns:
            Tool execution result
            
        Raises:
            MCPClientError: If response contains error
        """
        if "error" in response:
            error_info = response["error"]
            error_message = error_info.get('message', 'Unknown error')
            error_code = error_info.get('code', 'UNKNOWN')
            raise MCPClientError(f"MCP tool error [{error_code}]: {error_message}")
        
        # Extract result
        result = response.get("result", {})
        if isinstance(result, dict) and "content" in result:
            return result["content"]
        elif isinstance(result, dict) and "text" in result:
            return result["text"]
        else:
            return result
    
    def _validate_tool_parameters(self, tool_name: str, params: Dict[str, Any]) -> None:
        """
        Validate tool parameters.
        
        Args:
            tool_name: Name of the tool
            params: Parameters to validate
            
        Raises:
            MCPClientError: If parameters are invalid
        """
        tool_info = self.get_tool_info(tool_name)
        if not tool_info:
            return  # Tool not found, will be caught elsewhere
        
        # Basic parameter validation
        for param_name, param_value in params.items():
            if param_name in tool_info.parameters:
                param_spec = tool_info.parameters[param_name]
                param_type = param_spec.get('type')
                
                # Type validation
                if param_type == 'string' and not isinstance(param_value, str):
                    raise MCPClientError(f"Parameter '{param_name}' must be a string")
                elif param_type == 'integer' and not isinstance(param_value, int):
                    raise MCPClientError(f"Parameter '{param_name}' must be an integer")
                
                # String length validation
                if param_type == 'string' and isinstance(param_value, str):
                    if len(param_value.strip()) == 0:
                        raise MCPClientError(f"Parameter '{param_name}' cannot be empty")
    
    def is_tool_available(self, tool_name: str) -> bool:
        """
        Check if MCP tool is available.
        
        Args:
            tool_name: Tool name to check
            
        Returns:
            True if tool is available
        """
        return any(tool.name == tool_name for tool in self.available_tools)
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available MCP tool names.
        
        Returns:
            List of tool names
        """
        return [tool.name for tool in self.available_tools]
    
    def get_tool_info(self, tool_name: str) -> Optional[MCPTool]:
        """
        Get information about a specific MCP tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool information or None if not found
        """
        for tool in self.available_tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def health_check(self) -> bool:
        """
        Perform health check on MCP server.
        
        Returns:
            True if server is healthy
        """
        try:
            if not self.server_process or self.server_process.poll() is not None:
                return False
            
            # Could add a ping-like request here if the server supports it
            return True
        except:
            return False
    
    def __del__(self):
        """Cleanup method to ensure server process is terminated."""
        try:
            self.shutdown()
        except AttributeError:
            # Object was not fully initialized, nothing to clean up
            pass
    
    def shutdown(self):
        """Gracefully shutdown the MCP server."""
        if hasattr(self, 'server_process') and self.server_process:
            # If process is still running, attempt shutdown
            if self.server_process.poll() is None:
                try:
                    self._attempt_graceful_shutdown()
                except Exception:
                    # Ensure process is cleaned up even if there's an error
                    self._force_shutdown()
            
            # Always clean up the reference, regardless of process state
            self.server_process = None
    
    def _attempt_graceful_shutdown(self) -> None:
        """
        Attempt graceful shutdown of server process.
        
        Raises:
            subprocess.TimeoutExpired: If graceful shutdown times out
        """
        if not self.server_process:
            return
        
        # Send termination signal
        if hasattr(os, 'killpg'):
            self._terminate_process_group()
        else:
            self.server_process.terminate()
        
        # Wait for graceful shutdown
        try:
            self.server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._force_shutdown()
    
    def _terminate_process_group(self) -> None:
        """
        Terminate entire process group if possible.
        """
        if not self.server_process:
            return
        
        try:
            os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
        except Exception:
            # Fallback to regular termination
            self.server_process.terminate()
    
    def _force_shutdown(self) -> None:
        """
        Force shutdown of server process.
        """
        if not self.server_process:
            return
        
        try:
            if hasattr(os, 'killpg'):
                self._kill_process_group()
            else:
                self.server_process.kill()
            
            # Give it one more chance to exit
            try:
                self.server_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass  # Process is stubborn, but we tried
        except Exception:
            pass  # Best effort cleanup
    
    def _kill_process_group(self) -> None:
        """
        Kill entire process group if possible.
        """
        if not self.server_process:
            return
        
        try:
            os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
        except Exception:
            # Fallback to regular kill
            self.server_process.kill()


def create_mcp_client(settings: Settings) -> MCPClient:
    """
    Factory function to create MCP client instance.
    
    Args:
        settings: Application settings
        
    Returns:
        Configured MCP client instance
        
    Raises:
        MCPClientError: If client creation fails
    """
    try:
        return MCPClient(settings)
    except Exception as e:
        raise MCPClientError(f"Failed to create MCP client: {e}") from e 