"""
Video search functionality for YouTube Agent System.
Extracted from MVP's YouTubeCrawler class with enhanced type safety.
"""

import re
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.settings import Settings


class VideoSearchError(Exception):
    """Custom exception for video search related errors."""
    pass


class VideoSearcher:
    """
    Handles YouTube video search and metadata extraction.
    
    Extracted from MVP's YouTubeCrawler.search_videos method with:
    - Enhanced type safety with proper hints
    - Better error handling and custom exceptions
    - Configurable search parameters
    - Separation of concerns from transcript processing
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize VideoSearcher with YouTube API client.
        
        Args:
            settings: Application settings containing API keys
        """
        self.settings = settings
        self.youtube = build('youtube', 'v3', developerKey=settings.youtube_api_key)
    
    def search_videos(
        self, 
        query: str, 
        max_results: int = 5,
        duration_filter: str = "medium",
        order: str = "relevance"
    ) -> List[Dict[str, Any]]:
        """
        Search YouTube for videos by topic with comprehensive metadata.
        
        Args:
            query: Search query string
            max_results: Maximum number of videos to return
            duration_filter: Video duration filter ("short", "medium", "long")
            order: Sort order ("relevance", "date", "viewCount", "rating")
            
        Returns:
            List of video dictionaries with metadata
            
        Raises:
            VideoSearchError: If search fails or returns no results
        """
        try:
            # First, search for videos
            search_request = self.youtube.search().list(
                part='snippet',
                q=query,
                type='video',
                maxResults=max_results,
                order=order,
                videoDuration=duration_filter
            )
            search_response = search_request.execute()
            
            if not search_response.get('items'):
                raise VideoSearchError(f"No videos found for query: {query}")
            
            # Get video IDs for detailed info
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            # Get detailed video information including duration and statistics
            details_request = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=','.join(video_ids)
            )
            details_response = details_request.execute()
            
            videos = []
            for item in details_response['items']:
                video_data = self._extract_video_metadata(item)
                videos.append(video_data)
            
            return videos
            
        except HttpError as e:
            raise VideoSearchError(f"YouTube API Error: {e}") from e
        except Exception as e:
            raise VideoSearchError(f"Unexpected error during video search: {e}") from e
    
    def _extract_video_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format video metadata from YouTube API response.
        
        Args:
            item: Video item from YouTube API response
            
        Returns:
            Formatted video metadata dictionary
        """
        # Parse duration from ISO 8601 format
        duration_iso = item['contentDetails']['duration']
        duration_readable = self._parse_duration(duration_iso)
        
        return {
            'video_id': item['id'],
            'title': item['snippet']['title'],
            'channel': item['snippet']['channelTitle'],
            'description': item['snippet']['description'][:200] + "..." if len(item['snippet']['description']) > 200 else item['snippet']['description'],
            'published_at': item['snippet']['publishedAt'],
            'duration': duration_readable,
            'duration_iso': duration_iso,
            'view_count': int(item['statistics'].get('viewCount', 0)),
            'like_count': int(item['statistics'].get('likeCount', 0)),
            'comment_count': int(item['statistics'].get('commentCount', 0)),
            'thumbnail_url': item['snippet']['thumbnails']['medium']['url'],
            'url': f"https://youtube.com/watch?v={item['id']}",
            'category_id': item['snippet'].get('categoryId', 'Unknown')
        }
    
    def _parse_duration(self, duration_iso: str) -> str:
        """
        Convert ISO 8601 duration to readable format.
        
        Args:
            duration_iso: ISO 8601 duration string (e.g., "PT4M13S")
            
        Returns:
            Human-readable duration string (e.g., "4:13")
        """
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_iso)
        if not match:
            return "Unknown"
        
        hours, minutes, seconds = match.groups()
        hours = int(hours) if hours else 0
        minutes = int(minutes) if minutes else 0
        seconds = int(seconds) if seconds else 0
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def get_video_details(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video metadata dictionary or None if not found
        """
        try:
            request = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            )
            response = request.execute()
            
            if not response.get('items'):
                return None
                
            return self._extract_video_metadata(response['items'][0])
            
        except HttpError as e:
            raise VideoSearchError(f"Error fetching video details: {e}") from e


def create_video_searcher(settings: Settings) -> VideoSearcher:
    """
    Factory function to create VideoSearcher instance.
    
    Args:
        settings: Application settings
        
    Returns:
        Configured VideoSearcher instance
    """
    return VideoSearcher(settings) 