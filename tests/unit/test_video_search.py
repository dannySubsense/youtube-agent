"""
Unit tests for video search functionality.
Following TDD approach for YouTube Agent System modularization.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from tools.video_search import VideoSearcher, VideoSearchError, create_video_searcher
from config.settings import Settings


class TestVideoSearcher:
    """Test suite for VideoSearcher class."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Create test settings."""
        return Settings(
            youtube_api_key="test_youtube_key",
            openai_api_key="test_openai_key"
        )

    @pytest.fixture
    def mock_youtube_api(self):
        """Mock YouTube API client."""
        mock_youtube = Mock()
        mock_search = Mock()
        mock_videos = Mock()
        
        mock_youtube.search.return_value = mock_search
        mock_youtube.videos.return_value = mock_videos
        
        return mock_youtube

    @pytest.fixture
    def video_searcher(self, settings, mock_youtube_api):
        """Create VideoSearcher instance with mocked API."""
        with patch('tools.video_search.build') as mock_build:
            mock_build.return_value = mock_youtube_api
            searcher = VideoSearcher(settings)
            return searcher

    def test_video_searcher_initialization(self, settings):
        """Test VideoSearcher initializes correctly with settings."""
        with patch('tools.video_search.build') as mock_build:
            mock_build.return_value = Mock()
            searcher = VideoSearcher(settings)
            
            assert searcher.settings == settings
            mock_build.assert_called_once_with('youtube', 'v3', developerKey='test_youtube_key')

    def test_search_videos_success(self, video_searcher, mock_youtube_api):
        """Test successful video search returns formatted results."""
        # Mock search response
        mock_search_response = {
            'items': [
                {'id': {'videoId': 'test_video_1'}},
                {'id': {'videoId': 'test_video_2'}}
            ]
        }
        
        # Mock details response
        mock_details_response = {
            'items': [
                {
                    'id': 'test_video_1',
                    'snippet': {
                        'title': 'Test Video 1',
                        'channelTitle': 'Test Channel',
                        'description': 'Test description',
                        'publishedAt': '2023-01-01T00:00:00Z',
                        'thumbnails': {'medium': {'url': 'http://test.com/thumb1.jpg'}},
                        'categoryId': '28'
                    },
                    'contentDetails': {'duration': 'PT4M13S'},
                    'statistics': {'viewCount': '1000', 'likeCount': '50', 'commentCount': '10'}
                }
            ]
        }
        
        # Setup mock chain
        mock_youtube_api.search().list().execute.return_value = mock_search_response
        mock_youtube_api.videos().list().execute.return_value = mock_details_response
        
        # Execute search
        results = video_searcher.search_videos("test query")
        
        # Verify results
        assert len(results) == 1
        video = results[0]
        assert video['video_id'] == 'test_video_1'
        assert video['title'] == 'Test Video 1'
        assert video['channel'] == 'Test Channel'
        assert video['duration'] == '4:13'
        assert video['view_count'] == 1000
        assert video['like_count'] == 50
        assert video['comment_count'] == 10
        assert video['url'] == 'https://youtube.com/watch?v=test_video_1'

    def test_search_videos_no_results(self, video_searcher, mock_youtube_api):
        """Test search with no results raises VideoSearchError."""
        mock_search_response = {'items': []}
        mock_youtube_api.search().list().execute.return_value = mock_search_response
        
        with pytest.raises(VideoSearchError, match="No videos found for query"):
            video_searcher.search_videos("nonexistent query")

    def test_parse_duration_formats(self, video_searcher):
        """Test duration parsing handles various ISO 8601 formats."""
        test_cases = [
            ('PT4M13S', '4:13'),
            ('PT1H30M45S', '1:30:45'),
            ('PT2H5M', '2:05:00'),
            ('PT45S', '0:45'),
            ('PT1M', '1:00'),
            ('PT2H', '2:00:00'),
            ('INVALID', 'Unknown')
        ]
        
        for duration_iso, expected in test_cases:
            result = video_searcher._parse_duration(duration_iso)
            assert result == expected, f"Failed for {duration_iso}: expected {expected}, got {result}"

    def test_extract_video_metadata_complete(self, video_searcher):
        """Test metadata extraction handles complete video data."""
        api_item = {
            'id': 'test_video_123',
            'snippet': {
                'title': 'Complete Test Video',
                'channelTitle': 'Test Channel',
                'description': 'A' * 250,  # Long description to test truncation
                'publishedAt': '2023-01-01T00:00:00Z',
                'thumbnails': {'medium': {'url': 'http://test.com/thumb.jpg'}},
                'categoryId': '28'
            },
            'contentDetails': {'duration': 'PT10M30S'},
            'statistics': {
                'viewCount': '5000',
                'likeCount': '100',
                'commentCount': '25'
            }
        }
        
        result = video_searcher._extract_video_metadata(api_item)
        
        assert result['video_id'] == 'test_video_123'
        assert result['title'] == 'Complete Test Video'
        assert result['channel'] == 'Test Channel'
        assert len(result['description']) <= 203  # 200 + "..."
        assert result['description'].endswith('...')
        assert result['duration'] == '10:30'
        assert result['view_count'] == 5000
        assert result['like_count'] == 100
        assert result['comment_count'] == 25
        assert result['category_id'] == '28'
        assert result['url'] == 'https://youtube.com/watch?v=test_video_123'

    def test_get_video_details_success(self, video_searcher, mock_youtube_api):
        """Test getting details for specific video."""
        mock_details_response = {
            'items': [
                {
                    'id': 'specific_video',
                    'snippet': {
                        'title': 'Specific Video',
                        'channelTitle': 'Test Channel',
                        'description': 'Test description',
                        'publishedAt': '2023-01-01T00:00:00Z',
                        'thumbnails': {'medium': {'url': 'http://test.com/thumb.jpg'}},
                        'categoryId': '28'
                    },
                    'contentDetails': {'duration': 'PT5M'},
                    'statistics': {'viewCount': '2000', 'likeCount': '75', 'commentCount': '15'}
                }
            ]
        }
        
        mock_youtube_api.videos().list().execute.return_value = mock_details_response
        
        result = video_searcher.get_video_details('specific_video')
        
        assert result is not None
        assert result['video_id'] == 'specific_video'
        assert result['title'] == 'Specific Video'
        assert result['duration'] == '5:00'

    def test_get_video_details_not_found(self, video_searcher, mock_youtube_api):
        """Test getting details for non-existent video."""
        mock_details_response = {'items': []}
        mock_youtube_api.videos().list().execute.return_value = mock_details_response
        
        result = video_searcher.get_video_details('nonexistent_video')
        
        assert result is None

    def test_create_video_searcher_factory(self, settings):
        """Test factory function creates VideoSearcher instance."""
        with patch('tools.video_search.build') as mock_build:
            mock_build.return_value = Mock()
            searcher = create_video_searcher(settings)
            
            assert isinstance(searcher, VideoSearcher)
            assert searcher.settings == settings 