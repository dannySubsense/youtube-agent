#!/usr/bin/env python3
"""
Unit tests for MCP Client functionality.
Tests subprocess management, communication, and error handling.
"""

import pytest
import sys
import signal
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import subprocess
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.mcp_client import MCPClient, MCPClientError, create_mcp_client
from config.settings import Settings

# Test Constants - No more magic strings
VALID_YOUTUBE_API_KEY = "test_youtube_api_key_123456"
VALID_OPENAI_API_KEY = "test_openai_api_key_123456"
SHORT_API_KEY = "abc"  # Too short for validation
FAKE_SERVER_PATH = "/fake/mcp/server/path"
MCP_SERVER_STARTUP_DELAY = 2  # seconds
MCP_SHUTDOWN_TIMEOUT = 5  # seconds
TOOL_REQUEST_ID = 1
VALID_VIDEO_INPUT = "test_video_id_12345"

# Test Data
AVAILABLE_TOOLS = [
    "get_video_details", 
    "search_videos", 
    "analyze_video_engagement", 
    "evaluate_video_for_knowledge_base", 
    "get_video_transcript"
]

SUCCESSFUL_TOOL_RESPONSE = {
    "jsonrpc": "2.0",
    "id": TOOL_REQUEST_ID,
    "result": {"content": "Tool executed successfully"}
}

ERROR_TOOL_RESPONSE = {
    "jsonrpc": "2.0",
    "id": TOOL_REQUEST_ID,
    "error": {"message": "Tool execution failed", "code": "EXECUTION_ERROR"}
}

# Test Fixtures
@pytest.fixture
def valid_settings():
    """Fixture providing valid settings for MCP client."""
    return Settings(
        youtube_api_key=VALID_YOUTUBE_API_KEY, 
        openai_api_key=VALID_OPENAI_API_KEY
    )

@pytest.fixture
def invalid_settings():
    """Fixture providing invalid settings for testing validation."""
    return Settings(
        youtube_api_key=SHORT_API_KEY,  # Too short
        openai_api_key=VALID_OPENAI_API_KEY
    )

@pytest.fixture
def mock_server_process():
    """Fixture providing mock server process for testing."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Process running
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stderr = MagicMock()
    mock_process.pid = 12345
    return mock_process

@pytest.fixture
def failed_server_process():
    """Fixture providing failed server process for testing."""
    mock_process = MagicMock()
    mock_process.poll.return_value = 1  # Process failed
    mock_process.stderr.read.return_value = "Server failed to start"
    return mock_process


class TestMCPClientInitialization:
    """Test MCP client initialization and server discovery."""
    
    def test_create_mcp_client_factory(self, valid_settings):
        """Test factory function creates client correctly with valid settings."""
        with patch.object(MCPClient, '_initialize_connection'):
            client = create_mcp_client(valid_settings)
            assert isinstance(client, MCPClient)
            assert client.settings == valid_settings
    
    def test_create_mcp_client_factory_with_invalid_settings(self, invalid_settings):
        """Test factory function handles invalid settings correctly."""
        with pytest.raises(MCPClientError) as exc_info:
            create_mcp_client(invalid_settings)
        
        assert "YouTube API key appears to be invalid" in str(exc_info.value)
    
    @patch('agents.mcp_client.Path')
    def test_find_youtube_mcp_server_success(self, mock_path, valid_settings):
        """Test successful server path discovery."""
        # Mock Path to return existing server file
        mock_server_path = MagicMock()
        mock_server_path.exists.return_value = True
        mock_server_path.is_file.return_value = True
        mock_path.return_value.parent.parent = Mock()
        mock_path.return_value.parent.parent.__truediv__ = Mock(return_value=mock_server_path)
        
        with patch.object(MCPClient, '_initialize_connection'):
            client = MCPClient(valid_settings)
            # Should not raise an error
            assert client.server_path is not None
    
    @patch('agents.mcp_client.Path')
    def test_find_youtube_mcp_server_not_found(self, mock_path, valid_settings):
        """Test server path discovery failure when file doesn't exist."""
        # Mock Path to return non-existing server file
        mock_server_path = MagicMock()
        mock_server_path.exists.return_value = False
        mock_server_path.is_file.return_value = False
        
        # Mock the Path construction chain
        mock_path.return_value.parent.parent.__truediv__.return_value = mock_server_path
        
        with pytest.raises(MCPClientError) as exc_info:
            MCPClient(valid_settings)
        
        assert "YouTube MCP Server not found" in str(exc_info.value)


class TestMCPClientSubprocessManagement:
    """Test subprocess management functionality."""
    
    @patch('agents.mcp_client.subprocess.Popen')
    @patch('agents.mcp_client.time.sleep')
    def test_start_server_success(self, mock_sleep, mock_popen, valid_settings, mock_server_process):
        """Test successful server startup with proper process management."""
        mock_popen.return_value = mock_server_process
        
        with patch.object(MCPClient, '_find_youtube_mcp_server', return_value=FAKE_SERVER_PATH):
            client = MCPClient(valid_settings)
            
            # Verify subprocess was started correctly
            mock_popen.assert_called_once()
            mock_sleep.assert_called_once_with(MCP_SERVER_STARTUP_DELAY)
            assert client.server_process == mock_server_process
    
    @patch('agents.mcp_client.subprocess.Popen')
    @patch('agents.mcp_client.time.sleep')
    def test_start_server_failure(self, mock_sleep, mock_popen, valid_settings, failed_server_process):
        """Test server startup failure handling."""
        mock_popen.return_value = failed_server_process
        
        with patch.object(MCPClient, '_find_youtube_mcp_server', return_value=FAKE_SERVER_PATH):
            with pytest.raises(MCPClientError) as exc_info:
                MCPClient(valid_settings)
            
            assert "MCP server failed to start" in str(exc_info.value)
            mock_sleep.assert_called_once_with(MCP_SERVER_STARTUP_DELAY)
    
    @patch('agents.mcp_client.subprocess.Popen')
    def test_start_server_exception(self, mock_popen, valid_settings):
        """Test server startup with subprocess creation exception."""
        mock_popen.side_effect = Exception("Failed to start subprocess")
        
        with patch.object(MCPClient, '_find_youtube_mcp_server', return_value=FAKE_SERVER_PATH):
            with pytest.raises(MCPClientError) as exc_info:
                MCPClient(valid_settings)
            
            assert "Failed to start MCP server" in str(exc_info.value)


class TestMCPClientToolCalls:
    """Test MCP tool calling functionality."""
    
    def test_call_tool_success(self, valid_settings, mock_server_process):
        """Test successful tool execution with proper response handling."""
        mock_server_process.stdout.readline.return_value = json.dumps(SUCCESSFUL_TOOL_RESPONSE) + "\n"
        
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            client.server_process = mock_server_process
            
            result = client.call_tool("get_video_details", video_input=VALID_VIDEO_INPUT)
            
            assert result == "Tool executed successfully"
            
            # Verify request was sent correctly
            mock_server_process.stdin.write.assert_called_once()
            mock_server_process.stdin.flush.assert_called_once()
    
    def test_call_tool_unavailable(self, valid_settings, mock_server_process):
        """Test calling unavailable tool returns proper error."""
        nonexistent_tool = "nonexistent_tool"
        
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            client.server_process = mock_server_process
            
            with pytest.raises(MCPClientError) as exc_info:
                client.call_tool(nonexistent_tool)
            
            error_message = str(exc_info.value)
            assert f"Tool '{nonexistent_tool}' not available" in error_message
            assert "Available tools:" in error_message
    
    def test_call_tool_server_not_running(self, valid_settings, mock_server_process):
        """Test calling tool when server process is not running."""
        mock_server_process.poll.return_value = 1  # Process not running
        
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            client.server_process = mock_server_process
            
            with pytest.raises(MCPClientError) as exc_info:
                client.call_tool("get_video_details", video_input=VALID_VIDEO_INPUT)
            
            assert "MCP server is not running" in str(exc_info.value)
    
    def test_call_tool_no_response(self, valid_settings, mock_server_process):
        """Test tool call handling when server provides no response."""
        mock_server_process.stdout.readline.return_value = ""  # No response
        
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            client.server_process = mock_server_process
            
            with pytest.raises(MCPClientError) as exc_info:
                client.call_tool("get_video_details", video_input=VALID_VIDEO_INPUT)
            
            assert "No response from MCP server" in str(exc_info.value)
    
    def test_call_tool_invalid_json(self, valid_settings, mock_server_process):
        """Test tool call handling with malformed JSON response."""
        invalid_json_response = "invalid json response\n"
        mock_server_process.stdout.readline.return_value = invalid_json_response
        
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            client.server_process = mock_server_process
            
            with pytest.raises(MCPClientError) as exc_info:
                client.call_tool("get_video_details", video_input=VALID_VIDEO_INPUT)
            
            assert "Invalid JSON response" in str(exc_info.value)
    
    def test_call_tool_error_response(self, valid_settings, mock_server_process):
        """Test tool call handling when server returns error response."""
        mock_server_process.stdout.readline.return_value = json.dumps(ERROR_TOOL_RESPONSE) + "\n"
        
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            client.server_process = mock_server_process
            
            with pytest.raises(MCPClientError) as exc_info:
                client.call_tool("get_video_details", video_input=VALID_VIDEO_INPUT)
            
            error_message = str(exc_info.value)
            assert "Tool execution failed" in error_message
            assert "EXECUTION_ERROR" in error_message
    
    @pytest.mark.parametrize("empty_input,expected_error", [
        ("", "Tool name cannot be empty"),
        (None, "Tool name cannot be empty"),
    ])
    def test_call_tool_validation_errors(self, valid_settings, mock_server_process, empty_input, expected_error):
        """Test tool call validation with various invalid inputs."""
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            client.server_process = mock_server_process
            
            with pytest.raises(MCPClientError) as exc_info:
                client.call_tool(empty_input)
            
            assert expected_error in str(exc_info.value)


class TestMCPClientShutdown:
    """Test MCP client shutdown and cleanup functionality."""
    
    def test_shutdown_with_no_process(self, valid_settings):
        """Test shutdown when no server process exists."""
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            client.server_process = None
            
            # Should not raise exception
            client.shutdown()
            assert client.server_process is None
    
    def test_shutdown_with_dead_process(self, valid_settings):
        """Test shutdown when server process is already dead."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process not running
        
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            client.server_process = mock_process
            
            client.shutdown()
            
            # Should not try to terminate dead process
            mock_process.terminate.assert_not_called()
            assert client.server_process is None
    
    def test_shutdown_state_cleanup(self, valid_settings):
        """Test that shutdown properly cleans up client state."""
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            
            # Mock a running process
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Process running
            client.server_process = mock_process
            
            # Mock all the subprocess operations to avoid real calls
            with patch.object(client, '_attempt_graceful_shutdown') as mock_graceful:
                client.shutdown()
                
                mock_graceful.assert_called_once()
                assert client.server_process is None
    
    def test_del_method_calls_shutdown(self, valid_settings):
        """Test that __del__ method calls shutdown safely."""
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            client.server_process = MagicMock()
            
            # Mock shutdown to avoid real subprocess calls
            with patch.object(client, 'shutdown') as mock_shutdown:
                client.__del__()
                mock_shutdown.assert_called_once()
    
    def test_del_method_handles_missing_attributes(self, valid_settings):
        """Test that __del__ handles incomplete initialization gracefully."""
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            # Remove server_process attribute to simulate incomplete initialization
            if hasattr(client, 'server_process'):
                delattr(client, 'server_process')
            
            # Should not raise exception
            client.__del__()  # Should handle AttributeError gracefully


class TestMCPClientUtils:
    """Test MCP client utility methods."""
    
    def test_is_tool_available(self, valid_settings):
        """Test tool availability checking for valid and invalid tools."""
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            
            # Test available tools
            for tool_name in AVAILABLE_TOOLS:
                assert client.is_tool_available(tool_name) is True
            
            # Test unavailable tool
            assert client.is_tool_available("nonexistent_tool") is False
    
    def test_get_available_tools(self, valid_settings):
        """Test getting list of available tools."""
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            
            tools = client.get_available_tools()
            assert isinstance(tools, list)
            assert len(tools) == len(AVAILABLE_TOOLS)
            
            # Verify all expected tools are present
            for expected_tool in AVAILABLE_TOOLS:
                assert expected_tool in tools
    
    def test_get_tool_info(self, valid_settings):
        """Test getting detailed information about specific tools."""
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            
            # Test valid tool
            tool_info = client.get_tool_info("get_video_details")
            assert tool_info is not None
            assert tool_info.name == "get_video_details"
            assert "video_input" in tool_info.parameters
            
            # Test nonexistent tool
            tool_info = client.get_tool_info("nonexistent_tool")
            assert tool_info is None
    
    def test_health_check(self, valid_settings, mock_server_process):
        """Test health check functionality."""
        with patch.object(MCPClient, '_start_server'):
            client = MCPClient(valid_settings)
            
            # Test healthy server
            client.server_process = mock_server_process
            assert client.health_check() is True
            
            # Test unhealthy server
            mock_server_process.poll.return_value = 1  # Process dead
            assert client.health_check() is False
            
            # Test no server process
            client.server_process = None
            assert client.health_check() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 