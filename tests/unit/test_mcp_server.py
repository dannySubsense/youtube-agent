#!/usr/bin/env python3
"""
Unit tests for YouTube MCP Server functions.
Tests core functionality, edge cases, and error handling.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_integration.youtube_mcp_server import (
    get_video_id_from_url,
    get_playlist_id_from_url, 
    get_channel_id_from_url,
    load_api_key
)

# Test Constants - No more magic strings
VALID_VIDEO_ID = "dQw4w9WgXcQ"
VALID_PLAYLIST_ID = "PLExamplePlaylist123"
VALID_CHANNEL_ID = "UCExampleChannel123456"
VALID_USERNAME = "ExampleUser"
TEST_YOUTUBE_API_KEY = "AIza_test_api_key_123456789"
INVALID_API_KEY = "test_api_key_123"
EMPTY_STRING = ""
INVALID_URL = "https://example.com/not-youtube"

# Test Data Fixtures
@pytest.fixture
def valid_video_urls():
    """Fixture providing valid YouTube video URLs and expected video ID."""
    return [
        (f"https://www.youtube.com/watch?v={VALID_VIDEO_ID}", VALID_VIDEO_ID),
        (f"https://youtu.be/{VALID_VIDEO_ID}", VALID_VIDEO_ID),
        (f"https://www.youtube.com/watch?v={VALID_VIDEO_ID}&t=30s&list=PLExample", VALID_VIDEO_ID),
        (f"https://youtu.be/{VALID_VIDEO_ID}?t=30", VALID_VIDEO_ID),
        (f"https://m.youtube.com/watch?v={VALID_VIDEO_ID}", VALID_VIDEO_ID),
        (VALID_VIDEO_ID, VALID_VIDEO_ID),  # Direct video ID
    ]

@pytest.fixture
def invalid_video_inputs():
    """Fixture providing invalid video inputs and expected None result."""
    return [
        (INVALID_URL, None),
        (EMPTY_STRING, None),
        (None, None),
        ("abc123", None),  # Too short
        ("dQw4w9WgXcQ123456", None),  # Too long
        ("dQw4w9WgX@Q", None),  # Invalid characters
    ]

@pytest.fixture
def valid_playlist_urls():
    """Fixture providing valid playlist URLs and expected playlist ID."""
    return [
        (f"https://www.youtube.com/playlist?list={VALID_PLAYLIST_ID}", VALID_PLAYLIST_ID),
        (f"https://www.youtube.com/playlist?list={VALID_PLAYLIST_ID}&index=1", VALID_PLAYLIST_ID),
        (VALID_PLAYLIST_ID, VALID_PLAYLIST_ID),  # Direct playlist ID
    ]

@pytest.fixture
def valid_channel_urls():
    """Fixture providing valid channel URLs and expected channel ID."""
    return [
        (f"https://www.youtube.com/channel/{VALID_CHANNEL_ID}", VALID_CHANNEL_ID),
        (f"https://www.youtube.com/c/{VALID_USERNAME}", VALID_USERNAME),
        (f"https://www.youtube.com/@{VALID_USERNAME}", VALID_USERNAME),
        (f"@{VALID_USERNAME}", VALID_USERNAME),
        (f"https://www.youtube.com/user/{VALID_USERNAME}", VALID_USERNAME),
        (f"{VALID_CHANNEL_ID}789012", f"{VALID_CHANNEL_ID}789012"),  # Direct channel ID
    ]

@pytest.fixture
def mock_credentials_data():
    """Fixture providing mock credentials file data."""
    return {
        'valid_credentials': {'youtube': TEST_YOUTUBE_API_KEY},
        'missing_youtube_key': {'openai': 'some_other_key'},
        'invalid_key_type': {'youtube': 12345},  # Non-string key
        'empty_credentials': {},
    }


class TestVideoIdExtraction:
    """Test video ID extraction from various URL formats."""
    
    @pytest.mark.parametrize("url,expected_id", [
        (f"https://www.youtube.com/watch?v={VALID_VIDEO_ID}", VALID_VIDEO_ID),
        (f"https://youtu.be/{VALID_VIDEO_ID}", VALID_VIDEO_ID),
        (f"https://www.youtube.com/watch?v={VALID_VIDEO_ID}&t=30s&list=PLExample", VALID_VIDEO_ID),
        (f"https://youtu.be/{VALID_VIDEO_ID}?t=30", VALID_VIDEO_ID),
        (f"https://m.youtube.com/watch?v={VALID_VIDEO_ID}", VALID_VIDEO_ID),
        (VALID_VIDEO_ID, VALID_VIDEO_ID),  # Direct video ID
    ])
    def test_valid_video_urls(self, url, expected_id):
        """Test extraction from valid YouTube video URLs."""
        result = get_video_id_from_url(url)
        assert result == expected_id
    
    @pytest.mark.parametrize("invalid_input,expected", [
        (INVALID_URL, None),
        (EMPTY_STRING, None),
        (None, None),
        ("abc123", None),  # Too short
        ("dQw4w9WgXcQ123456", None),  # Too long
        ("dQw4w9WgX@Q", None),  # Invalid characters
    ])
    def test_invalid_video_inputs(self, invalid_input, expected):
        """Test handling of invalid video inputs."""
        result = get_video_id_from_url(invalid_input)
        assert result == expected


class TestPlaylistIdExtraction:
    """Test playlist ID extraction from various URL formats."""
    
    @pytest.mark.parametrize("url,expected_id", [
        (f"https://www.youtube.com/playlist?list={VALID_PLAYLIST_ID}", VALID_PLAYLIST_ID),
        (f"https://www.youtube.com/playlist?list={VALID_PLAYLIST_ID}&index=1", VALID_PLAYLIST_ID),
        (VALID_PLAYLIST_ID, VALID_PLAYLIST_ID),  # Direct playlist ID
    ])
    def test_valid_playlist_urls(self, url, expected_id):
        """Test extraction from valid playlist URLs."""
        result = get_playlist_id_from_url(url)
        assert result == expected_id
    
    @pytest.mark.parametrize("invalid_input,expected", [
        (INVALID_URL, None),
        (EMPTY_STRING, None),
        (None, None),
    ])
    def test_invalid_playlist_inputs(self, invalid_input, expected):
        """Test handling of invalid playlist inputs."""
        result = get_playlist_id_from_url(invalid_input)
        assert result == expected


class TestChannelIdExtraction:
    """Test channel ID extraction from various URL formats."""
    
    @pytest.mark.parametrize("url,expected_id", [
        (f"https://www.youtube.com/channel/{VALID_CHANNEL_ID}", VALID_CHANNEL_ID),
        (f"https://www.youtube.com/c/{VALID_USERNAME}", VALID_USERNAME),
        (f"https://www.youtube.com/@{VALID_USERNAME}", VALID_USERNAME),
        (f"@{VALID_USERNAME}", VALID_USERNAME),
        (f"https://www.youtube.com/user/{VALID_USERNAME}", VALID_USERNAME),
        (f"{VALID_CHANNEL_ID}789012", f"{VALID_CHANNEL_ID}789012"),  # Direct channel ID
    ])
    def test_valid_channel_urls(self, url, expected_id):
        """Test extraction from valid channel URLs."""
        result = get_channel_id_from_url(url)
        assert result == expected_id
    
    @pytest.mark.parametrize("invalid_input,expected", [
        (INVALID_URL, None),
        (EMPTY_STRING, None),
        (None, None),
    ])
    def test_invalid_channel_inputs(self, invalid_input, expected):
        """Test handling of invalid channel inputs."""
        result = get_channel_id_from_url(invalid_input)
        assert result == expected


class TestApiKeyLoading:
    """Test API key loading functionality."""
    
    @patch('mcp_integration.youtube_mcp_server.open')
    @patch('mcp_integration.youtube_mcp_server.yaml.safe_load')
    def test_load_api_key_success(self, mock_yaml_load, mock_open):
        """Test successful API key loading with valid credentials."""
        # Use realistic test data
        mock_yaml_load.return_value = {'youtube': TEST_YOUTUBE_API_KEY}
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        result = load_api_key()
        assert result == TEST_YOUTUBE_API_KEY
        
        # Verify exact file operations
        mock_open.assert_called_once()
        mock_yaml_load.assert_called_once_with(mock_file)
    
    @patch('mcp_integration.youtube_mcp_server.open')
    def test_load_api_key_file_not_found(self, mock_open):
        """Test API key loading when credentials file doesn't exist."""
        mock_open.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(ValueError) as exc_info:
            load_api_key()
        
        assert "credentials.yml file not found" in str(exc_info.value)
        mock_open.assert_called_once()
    
    @patch('mcp_integration.youtube_mcp_server.open')
    @patch('mcp_integration.youtube_mcp_server.yaml.safe_load')
    def test_load_api_key_missing_youtube_key(self, mock_yaml_load, mock_open):
        """Test API key loading when youtube key is missing from credentials."""
        # Mock credentials without youtube key
        mock_yaml_load.return_value = {'openai': 'some_other_key'}
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with pytest.raises(ValueError) as exc_info:
            load_api_key()
        
        assert "youtube key not found" in str(exc_info.value)
        mock_yaml_load.assert_called_once_with(mock_file)
    
    @patch('mcp_integration.youtube_mcp_server.open')
    @patch('mcp_integration.youtube_mcp_server.yaml.safe_load')
    def test_load_api_key_yaml_error(self, mock_yaml_load, mock_open):
        """Test API key loading with invalid YAML format."""
        mock_yaml_load.side_effect = Exception("Invalid YAML syntax")
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with pytest.raises(ValueError) as exc_info:
            load_api_key()
        
        assert "Error reading credentials.yml" in str(exc_info.value)
        mock_yaml_load.assert_called_once_with(mock_file)
    
    @pytest.mark.parametrize("invalid_data,expected_error", [
        ({'youtube': 12345}, "YouTube API key must be a string"),
        ({'youtube': "short"}, "YouTube API key appears to be invalid (too short)"),
        ({'youtube': "invalid_prefix_key"}, "YouTube API key format appears invalid"),
        ({'openai': 'some_key'}, "youtube key not found in credentials.yml"),
        ({}, "credentials.yml file is empty or invalid"),
        (None, "credentials.yml file is empty or invalid"),
    ])
    @patch('mcp_integration.youtube_mcp_server.open')
    @patch('mcp_integration.youtube_mcp_server.yaml.safe_load')
    def test_load_api_key_validation_errors(self, mock_yaml_load, mock_open, invalid_data, expected_error):
        """Test API key validation with various invalid data scenarios."""
        mock_yaml_load.return_value = invalid_data
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with pytest.raises(ValueError) as exc_info:
            load_api_key()
        
        assert expected_error in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 