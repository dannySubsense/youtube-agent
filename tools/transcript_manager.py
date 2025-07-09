"""
Transcript processing functionality for YouTube Agent System.
Extracted from MVP's YouTubeCrawler class with enhanced error handling and rate limiting.
"""

import time
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeRequestFailed, TranscriptsDisabled, NoTranscriptFound

from config.settings import Settings


@dataclass
class TranscriptResult:
    """Result of transcript extraction for a single video."""
    video_id: str
    title: str
    success: bool
    transcript: str
    error_message: Optional[str] = None
    duration: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ProcessingStats:
    """Statistics for batch transcript processing."""
    total_videos: int
    successful_transcripts: int
    failed_transcripts: int
    success_rate: float


class TranscriptManagerError(Exception):
    """Custom exception for transcript processing related errors."""
    pass


class TranscriptManager:
    """
    Handles YouTube transcript extraction with robust error handling and rate limiting.
    
    Extracted from MVP's YouTubeCrawler.get_transcript and process_videos methods with:
    - Enhanced rate limiting with exponential backoff
    - Comprehensive error handling for different failure modes
    - Detailed processing statistics and progress tracking
    - Separation of concerns from video search functionality
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize TranscriptManager with configuration.
        
        Args:
            settings: Application settings containing rate limiting configuration
        """
        self.settings = settings
        self.rate_limit_delay = settings.rate_limit_delay
        self.max_retries = settings.max_retries
    
    def get_transcript(self, video_id: str) -> Tuple[str, Optional[str]]:
        """
        Get transcript for a single video with rate limiting and retry logic.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Tuple of (transcript_text, error_message)
            If successful: (transcript_text, None)
            If failed: ("", error_message)
        """
        for attempt in range(self.max_retries):
            try:
                # Add delay between requests to avoid rate limiting
                if attempt > 0:
                    delay = self._calculate_backoff_delay(attempt)
                    time.sleep(delay)
                
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([entry['text'] for entry in transcript_list])
                return transcript_text, None
                
            except YouTubeRequestFailed as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    if attempt < self.max_retries - 1:
                        continue  # Retry with backoff
                    else:
                        return "", f"Rate limited after {self.max_retries} attempts. Please wait 15-30 minutes and try again."
                else:
                    return "", f"YouTube API error: {str(e)}"
            
            except TranscriptsDisabled:
                return "", "Transcripts are disabled for this video"
            
            except NoTranscriptFound:
                return "", "No transcript available for this video"
            
            except Exception as e:
                return "", f"Unexpected error: {str(e)}"
        
        return "", f"Failed after {self.max_retries} attempts"
    
    def process_videos(
        self, 
        videos: List[Dict[str, Any]], 
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[TranscriptResult], ProcessingStats]:
        """
        Process multiple videos to extract transcripts with progress tracking.
        
        Args:
            videos: List of video dictionaries with metadata
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Tuple of (transcript_results, processing_stats)
        """
        results = []
        successful_count = 0
        
        total_videos = len(videos)
        
        for i, video in enumerate(videos):
            # Add delay between videos (except first one)
            if i > 0:
                time.sleep(self.rate_limit_delay)
            
            # Progress callback
            if progress_callback:
                progress_callback(i, total_videos, video['title'])
            
            # Extract transcript
            transcript, error = self.get_transcript(video['video_id'])
            
            # Create result
            success = error is None
            if success:
                successful_count += 1
            
            result = TranscriptResult(
                video_id=video['video_id'],
                title=video['title'],
                success=success,
                transcript=transcript,
                error_message=error,
                duration=video.get('duration'),
                url=video.get('url')
            )
            results.append(result)
        
        # Calculate statistics
        stats = ProcessingStats(
            total_videos=total_videos,
            successful_transcripts=successful_count,
            failed_transcripts=total_videos - successful_count,
            success_rate=successful_count / total_videos if total_videos > 0 else 0.0
        )
        
        return results, stats
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        base_delay = 2.0
        # Exponential backoff: 2, 4, 8, 16 seconds
        delay = base_delay * (2 ** attempt)
        # Add random jitter to avoid thundering herd
        jitter = random.uniform(0, 1)
        return delay + jitter
    
    def format_processing_results(
        self, 
        results: List[TranscriptResult], 
        stats: ProcessingStats
    ) -> Dict[str, Any]:
        """
        Format processing results for compatibility with MVP workflow.
        
        Args:
            results: List of transcript results
            stats: Processing statistics
            
        Returns:
            Dictionary compatible with MVP's processed_data format
        """
        processed_data = []
        failed_videos = []
        
        for result in results:
            if result.success:
                processed_data.append({
                    'video_id': result.video_id,
                    'title': result.title,
                    'transcript': result.transcript,
                    'duration': result.duration,
                    'url': result.url,
                })
            else:
                failed_videos.append({
                    'title': result.title,
                    'error': result.error_message
                })
        
        return {
            'processed_data': processed_data,
            'failed_videos': failed_videos,
            'stats': stats,
            'successful_videos': [r.title for r in results if r.success]
        }
    
    def chunk_transcript(
        self, 
        transcript: str, 
        chunk_size: Optional[int] = None
    ) -> List[str]:
        """
        Split transcript into chunks for vector database processing.
        
        Args:
            transcript: Full transcript text
            chunk_size: Size of each chunk (uses settings default if None)
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.settings.transcript_chunk_size
        
        chunks = []
        for i in range(0, len(transcript), chunk_size):
            chunk = transcript[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks


def create_transcript_manager(settings: Settings) -> TranscriptManager:
    """
    Factory function to create TranscriptManager instance.
    
    Args:
        settings: Application settings
        
    Returns:
        Configured TranscriptManager instance
    """
    return TranscriptManager(settings) 