#!/usr/bin/env python3
"""
YouTube MCP Server

A Model Context Protocol server that provides access to YouTube data via the YouTube Data API v3.
Provides tools for getting video details, playlist information, and playlist items.
Adapted for YouTube Agent System project structure.
"""

import os
import re
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import httpx
from mcp.server.fastmcp import FastMCP

# YouTube transcript API imports (optional dependency)
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    TRANSCRIPT_API_AVAILABLE = False
    YouTubeTranscriptApi = None
    TranscriptsDisabled = None
    NoTranscriptFound = None

# Initialize the MCP server
mcp = FastMCP("YouTube Data Server")

# YouTube API configuration
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# Load API key from credentials file
def load_api_key() -> str:
    """Load YouTube API key from project credentials.yml file."""
    try:
        credentials = _load_credentials_file()
        youtube_key = _extract_youtube_key(credentials)
        return _validate_api_key_format(youtube_key)
    except Exception as e:
        raise ValueError(f"Error reading credentials.yml: {str(e)}")


def _load_credentials_file() -> Dict[str, Any]:
    """
    Load and parse credentials file.
    
    Returns:
        Parsed credentials dictionary
        
    Raises:
        ValueError: If file loading fails
    """
    # Navigate to project root credentials.yml (from mcp_integration/ up to project root)
    script_dir = Path(__file__).parent.parent  # Go up from mcp_integration/ to project root
    credentials_file = script_dir / "credentials.yml"
    
    if not credentials_file.exists():
        raise ValueError(f"credentials.yml file not found at {credentials_file}. Please ensure it exists with 'youtube' key.")
    
    if not credentials_file.is_file():
        raise ValueError(f"credentials.yml exists but is not a file: {credentials_file}")
    
    try:
        with open(credentials_file, 'r') as f:
            credentials = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"credentials.yml file not found at {credentials_file}. Please ensure it exists with 'youtube' key.")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in credentials file: {e}")
    
    if not credentials:
        raise ValueError("credentials.yml file is empty or invalid")
    
    return credentials


def _extract_youtube_key(credentials: Dict[str, Any]) -> str:
    """
    Extract YouTube API key from credentials.
    
    Args:
        credentials: Credentials dictionary
        
    Returns:
        YouTube API key string
        
    Raises:
        ValueError: If key extraction fails
    """
    # Extract YouTube API key from project format: youtube: 'api_key'
    youtube_key = credentials.get('youtube')
    if not youtube_key:
        raise ValueError("youtube key not found in credentials.yml")
    
    if not isinstance(youtube_key, str):
        raise ValueError("YouTube API key must be a string")
    
    return youtube_key.strip()


def _validate_api_key_format(youtube_key: str) -> str:
    """
    Validate YouTube API key format.
    
    Args:
        youtube_key: API key to validate
        
    Returns:
        Validated API key
        
    Raises:
        ValueError: If validation fails
    """
    if len(youtube_key) < 10:
        raise ValueError("YouTube API key appears to be invalid (too short)")
    
    # Basic format validation for Google API keys (allow test keys)
    if not (youtube_key.startswith('AIza') or youtube_key.startswith('test') or youtube_key.startswith('fake')):
        raise ValueError("YouTube API key format appears invalid (should start with 'AIza', 'test', or 'fake' for testing)")
    
    return youtube_key

# Validate API key on startup
try:
    API_KEY = load_api_key()
except ValueError as e:
    # Don't fail immediately - let individual tool calls handle the error
    API_KEY = None
    print(f"Warning: API key validation failed: {e}", file=sys.stderr)

def get_video_id_from_url(url: str) -> Optional[str]:
    """
    Extract video ID from various YouTube URL formats.
    
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    """
    if not url:
        return None
        
    # Handle youtu.be format
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    
    # Handle youtube.com format
    parsed = urlparse(url)
    if parsed.hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        query_params = parse_qs(parsed.query)
        return query_params.get("v", [None])[0]
    
    # If it's already just an ID (11 characters, alphanumeric + - and _)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
        
    return None

def get_playlist_id_from_url(url: str) -> Optional[str]:
    """
    Extract playlist ID from YouTube URL formats.
    
    Supports:
    - https://www.youtube.com/playlist?list=PLAYLIST_ID
    - https://youtube.com/playlist?list=PLAYLIST_ID
    """
    if not url:
        return None
        
    parsed = urlparse(url)
    if parsed.hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        query_params = parse_qs(parsed.query)
        return query_params.get("list", [None])[0]
    
    # If it's already just an ID
    if re.match(r'^[a-zA-Z0-9_-]+$', url):
        return url
        
    return None

def get_channel_id_from_url(url: str) -> Optional[str]:
    """
    Extract channel ID from YouTube channel URL formats.
    
    Supports:
    - https://www.youtube.com/channel/CHANNEL_ID
    - https://www.youtube.com/c/channelname
    - https://www.youtube.com/@username
    - https://youtube.com/user/username
    - @username (direct format)
    """
    if not url:
        return None
    
    # Handle direct @username format
    if url.startswith('@'):
        return url[1:]  # Remove the @ symbol
        
    parsed = urlparse(url)
    if parsed.hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        path = parsed.path
        
        # Handle /channel/CHANNEL_ID format
        if "/channel/" in path:
            return path.split("/channel/")[1].split("/")[0]
        
        # Handle /c/channelname format (custom URL)
        elif "/c/" in path:
            return path.split("/c/")[1].split("/")[0]
        
        # Handle /@username format
        elif "/@" in path:
            return path.split("/@")[1].split("/")[0]
        
        # Handle /user/username format (legacy)
        elif "/user/" in path:
            return path.split("/user/")[1].split("/")[0]
    
    # If it's already a channel ID (starts with UC and 22 chars after UC)
    if re.match(r'^UC[a-zA-Z0-9_-]{22}$', url):
        return url
    
    # If it's a username or custom name
    if re.match(r'^[a-zA-Z0-9_-]+$', url):
        return url
        
    return None

async def make_youtube_api_request(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to the YouTube Data API v3."""
    if not API_KEY:
        raise ValueError("YouTube API key is required")
    
    params["key"] = API_KEY
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{YOUTUBE_API_BASE}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                error_data = e.response.json() if e.response.headers.get("content-type", "").startswith("application/json") else {}
                error_message = error_data.get("error", {}).get("message", "API quota exceeded or invalid key")
                raise ValueError(f"YouTube API error (403): {error_message}")
            elif e.response.status_code == 404:
                raise ValueError("YouTube resource not found (404)")
            else:
                raise ValueError(f"YouTube API error ({e.response.status_code}): {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Network error connecting to YouTube API: {str(e)}")

@mcp.tool()
async def get_video_details(video_input: str) -> str:
    """
    Get detailed information about a YouTube video.
    
    Args:
        video_input: YouTube video URL or video ID
        
    Returns:
        Formatted string with video details including title, description, statistics, etc.
    """
    # Extract video ID from URL or use as-is if it's already an ID
    video_id = get_video_id_from_url(video_input)
    if not video_id:
        return f"Error: Could not extract video ID from '{video_input}'. Please provide a valid YouTube URL or 11-character video ID."
    
    try:
        # Get video details
        data = await make_youtube_api_request("videos", {
            "part": "snippet,statistics,contentDetails,status",
            "id": video_id
        })
        
        if not data.get("items"):
            return f"Error: Video with ID '{video_id}' not found or is not accessible."
        
        video = data["items"][0]
        snippet = video.get("snippet", {})
        statistics = video.get("statistics", {})
        content_details = video.get("contentDetails", {})
        status = video.get("status", {})
        
        # Format duration (convert from ISO 8601 format)
        duration = content_details.get("duration", "Unknown")
        if duration.startswith("PT"):
            # Simple parsing for common formats like PT4M13S
            duration = duration.replace("PT", "").replace("H", "h ").replace("M", "m ").replace("S", "s")
        
        # Build formatted response
        result = f"""YouTube Video Details:

Title: {snippet.get('title', 'Unknown')}
Channel: {snippet.get('channelTitle', 'Unknown')}
Published: {snippet.get('publishedAt', 'Unknown')[:10]}
Duration: {duration}

Statistics:
- Views: {int(statistics.get('viewCount', 0)):,}
- Likes: {int(statistics.get('likeCount', 0)):,}
- Comments: {int(statistics.get('commentCount', 0)):,}

Status: {status.get('privacyStatus', 'Unknown').title()}
License: {status.get('license', 'Unknown')}

Description:
{snippet.get('description', 'No description available')[:500]}{'...' if len(snippet.get('description', '')) > 500 else ''}

Video ID: {video_id}
Video URL: https://www.youtube.com/watch?v={video_id}
"""
        
        return result
        
    except Exception as e:
        return f"Error fetching video details: {str(e)}"

@mcp.tool()
async def search_videos(query: str, max_results: int = 10, order: str = "relevance") -> str:
    """
    Search YouTube for videos by keywords.
    
    Args:
        query: Search query keywords
        max_results: Maximum number of results to return (default: 10, max: 50)
        order: Sort order - relevance, date, rating, viewCount, title (default: relevance)
        
    Returns:
        Formatted string with search results
    """
    # Validate max_results
    max_results = max(1, min(50, max_results))
    
    # Validate order parameter
    valid_orders = ["relevance", "date", "rating", "viewCount", "title"]
    if order not in valid_orders:
        order = "relevance"
    
    try:
        # Search for videos
        search_data = await make_youtube_api_request("search", {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "order": order
        })
        
        if not search_data.get("items"):
            return f"No videos found for query '{query}'."
        
        videos = search_data["items"]
        total_results = search_data.get("pageInfo", {}).get("totalResults", len(videos))
        
        result = f"""YouTube Video Search Results:

Query: "{query}"
Sort Order: {order.title()}
Showing: {len(videos)} of {total_results:,} results

Videos:
"""
        
        for i, video in enumerate(videos, 1):
            snippet = video.get("snippet", {})
            video_id = video.get("id", {}).get("videoId", "Unknown")
            
            title = snippet.get("title", "Unknown Title")
            channel = snippet.get("channelTitle", "Unknown Channel")
            published = snippet.get("publishedAt", "Unknown")
            description = snippet.get("description", "No description")
            
            # Format publish date
            if published != "Unknown":
                published = published[:10]  # Just the date part
            
            # Truncate description
            if len(description) > 150:
                description = description[:150] + "..."
            
            result += f"""
{i}. {title}
   Channel: {channel}
   Published: {published}
   Video ID: {video_id}
   URL: https://www.youtube.com/watch?v={video_id}
   Description: {description}

"""
        
        if total_results > len(videos):
            result += f"... and {total_results - len(videos):,} more videos available"
        
        return result
        
    except Exception as e:
        return f"Error searching videos: {str(e)}"

@mcp.tool()
async def analyze_video_engagement(video_input: str) -> str:
    """
    Analyze video engagement metrics and provide insights.
    
    Args:
        video_input: YouTube video URL or video ID
        
    Returns:
        Formatted string with engagement analysis and insights
    """
    # Extract video ID from URL or use as-is if it's already an ID
    video_id = get_video_id_from_url(video_input)
    if not video_id:
        return f"Error: Could not extract video ID from '{video_input}'. Please provide a valid YouTube URL or 11-character video ID."
    
    try:
        # Get comprehensive video data
        video_data = await make_youtube_api_request("videos", {
            "part": "snippet,statistics,contentDetails",
            "id": video_id
        })
        
        if not video_data.get("items"):
            return f"Error: Video with ID '{video_id}' not found or is not accessible."
        
        video = video_data["items"][0]
        snippet = video.get("snippet", {})
        statistics = video.get("statistics", {})
        content_details = video.get("contentDetails", {})
        
        # Extract metrics
        title = snippet.get("title", "Unknown Title")
        channel = snippet.get("channelTitle", "Unknown Channel")
        published = snippet.get("publishedAt", "Unknown")
        
        view_count = int(statistics.get("viewCount", 0))
        like_count = int(statistics.get("likeCount", 0))
        comment_count = int(statistics.get("commentCount", 0))
        
        # Calculate engagement metrics
        if view_count > 0:
            like_rate = (like_count / view_count) * 100
            comment_rate = (comment_count / view_count) * 100
            engagement_rate = like_rate + comment_rate
        else:
            like_rate = comment_rate = engagement_rate = 0
        
        # Calculate video age in days
        video_age_days = "Unknown"
        if published != "Unknown":
            from datetime import datetime
            try:
                pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                current_date = datetime.now(pub_date.tzinfo)
                video_age_days = (current_date - pub_date).days
            except:
                video_age_days = "Unknown"
        
        # Calculate average views per day
        if isinstance(video_age_days, int) and video_age_days > 0:
            avg_views_per_day = view_count / video_age_days
        else:
            avg_views_per_day = "Unknown"
        
        # Format duration
        duration = content_details.get("duration", "Unknown")
        if duration.startswith("PT"):
            duration = duration.replace("PT", "").replace("H", "h ").replace("M", "m ").replace("S", "s")
        
        # Engagement benchmarks (rough industry averages)
        def get_engagement_assessment(rate):
            if rate >= 8.0:
                return "🔥 Exceptional (8%+)"
            elif rate >= 4.0:
                return "⭐ Excellent (4-8%)"
            elif rate >= 2.0:
                return "✅ Good (2-4%)"
            elif rate >= 1.0:
                return "📊 Average (1-2%)"
            else:
                return "📉 Below Average (<1%)"
        
        # Format numbers for display
        def format_number(num):
            if isinstance(num, int):
                if num >= 1000000000:
                    return f"{num/1000000000:.1f}B"
                elif num >= 1000000:
                    return f"{num/1000000:.1f}M"
                elif num >= 1000:
                    return f"{num/1000:.1f}K"
                else:
                    return f"{num:,}"
            return str(num)
        
        result = f"""YouTube Video Engagement Analysis:

Video: {title}
Channel: {channel}
Published: {published[:10] if published != "Unknown" else "Unknown"}
Duration: {duration}

📊 Core Metrics:
- Views: {format_number(view_count)}
- Likes: {format_number(like_count)}
- Comments: {format_number(comment_count)}

🎯 Engagement Rates:
- Like Rate: {like_rate:.2f}% ({like_count:,} likes per 100 views)
- Comment Rate: {comment_rate:.2f}% ({comment_count:,} comments per 100 views)
- Total Engagement Rate: {engagement_rate:.2f}%

📈 Performance Assessment:
- Overall Engagement: {get_engagement_assessment(engagement_rate)}
"""
        
        # Add time-based analysis if available
        if isinstance(video_age_days, int):
            result += f"""
⏰ Time Analysis:
- Video Age: {video_age_days} days
- Average Views/Day: {format_number(int(avg_views_per_day)) if isinstance(avg_views_per_day, (int, float)) else avg_views_per_day}
"""
        
        # Add engagement insights
        result += f"""

🔍 Insights:
- Video is {video_age_days} days old
- Engagement rate suggests {'high' if engagement_rate >= 4 else 'moderate' if engagement_rate >= 2 else 'low'} audience interaction
- {'Strong' if like_rate >= 2 else 'Moderate' if like_rate >= 1 else 'Low'} like engagement
- {'Active' if comment_rate >= 0.5 else 'Limited'} comment discussion

Video URL: https://www.youtube.com/watch?v={video_id}
"""
        
        return result
        
    except Exception as e:
        return f"Error analyzing video engagement: {str(e)}"

@mcp.tool()
async def evaluate_video_for_knowledge_base(video_input: str) -> str:
    """
    Analyze video metadata to help decide if worth adding to knowledge base.
    
    Args:
        video_input: YouTube video URL or video ID
        
    Returns:
        Formatted string with evaluation and recommendation
    """
    # Extract video ID from URL or use as-is if it's already an ID
    video_id = get_video_id_from_url(video_input)
    if not video_id:
        return f"Error: Could not extract video ID from '{video_input}'. Please provide a valid YouTube URL or 11-character video ID."
    
    try:
        # Get comprehensive video data for evaluation
        video_data = await make_youtube_api_request("videos", {
            "part": "snippet,statistics,contentDetails,topicDetails",
            "id": video_id
        })
        
        if not video_data.get("items"):
            return f"Error: Video with ID '{video_id}' not found or is not accessible."
        
        video = video_data["items"][0]
        snippet = video.get("snippet", {})
        statistics = video.get("statistics", {})
        content_details = video.get("contentDetails", {})
        topic_details = video.get("topicDetails", {})
        
        # Extract key metrics for evaluation
        title = snippet.get("title", "Unknown Title")
        channel = snippet.get("channelTitle", "Unknown Channel")
        description = snippet.get("description", "")
        published = snippet.get("publishedAt", "Unknown")
        
        view_count = int(statistics.get("viewCount", 0))
        like_count = int(statistics.get("likeCount", 0))
        comment_count = int(statistics.get("commentCount", 0))
        
        # Parse duration
        duration_str = content_details.get("duration", "PT0S")
        duration_seconds = 0
        if duration_str.startswith("PT"):
            import re
            duration_match = re.findall(r'(\d+)([HMS])', duration_str)
            for value, unit in duration_match:
                if unit == 'H':
                    duration_seconds += int(value) * 3600
                elif unit == 'M':
                    duration_seconds += int(value) * 60
                elif unit == 'S':
                    duration_seconds += int(value)
        
        duration_minutes = duration_seconds / 60
        
        # Calculate quality indicators
        if view_count > 0:
            like_rate = (like_count / view_count) * 100
            engagement_rate = ((like_count + comment_count) / view_count) * 100
        else:
            like_rate = engagement_rate = 0
        
        # Scoring algorithm for knowledge base worthiness
        score = 0
        reasons = []
        
        # Duration scoring (optimal: 10-60 minutes for educational content)
        if 10 <= duration_minutes <= 60:
            score += 20
            reasons.append(f"✅ Good duration ({duration_minutes:.1f} min) for educational content")
        elif 5 <= duration_minutes < 10:
            score += 15
            reasons.append(f"⚠️ Short duration ({duration_minutes:.1f} min) but acceptable")
        elif duration_minutes > 60:
            score += 10
            reasons.append(f"⚠️ Long duration ({duration_minutes:.1f} min) may need chunking")
        else:
            score += 5
            reasons.append(f"❌ Very short duration ({duration_minutes:.1f} min)")
        
        # Engagement scoring
        if engagement_rate >= 3:
            score += 25
            reasons.append(f"✅ High engagement rate ({engagement_rate:.2f}%)")
        elif engagement_rate >= 1:
            score += 15
            reasons.append(f"✅ Moderate engagement rate ({engagement_rate:.2f}%)")
        else:
            score += 5
            reasons.append(f"⚠️ Low engagement rate ({engagement_rate:.2f}%)")
        
        # View count scoring (indicates content quality/relevance)
        if view_count >= 100000:
            score += 20
            reasons.append(f"✅ High view count ({view_count:,})")
        elif view_count >= 10000:
            score += 15
            reasons.append(f"✅ Moderate view count ({view_count:,})")
        elif view_count >= 1000:
            score += 10
            reasons.append(f"⚠️ Lower view count ({view_count:,})")
        else:
            score += 5
            reasons.append(f"❌ Very low view count ({view_count:,})")
        
        # Description quality (longer descriptions often indicate more educational content)
        desc_length = len(description)
        if desc_length >= 500:
            score += 15
            reasons.append(f"✅ Detailed description ({desc_length} chars)")
        elif desc_length >= 200:
            score += 10
            reasons.append(f"✅ Moderate description ({desc_length} chars)")
        else:
            score += 5
            reasons.append(f"⚠️ Brief description ({desc_length} chars)")
        
        # Title analysis for educational keywords
        educational_keywords = ['tutorial', 'guide', 'how to', 'explained', 'learn', 'course', 'lesson', 'training', 'workshop', 'demo', 'walkthrough']
        title_lower = title.lower()
        if any(keyword in title_lower for keyword in educational_keywords):
            score += 10
            reasons.append("✅ Educational keywords in title")
        
        # Final recommendation
        if score >= 70:
            recommendation = "🔥 HIGHLY RECOMMENDED"
            decision = "KEEP"
        elif score >= 50:
            recommendation = "✅ RECOMMENDED"
            decision = "KEEP"
        elif score >= 30:
            recommendation = "⚠️ CONSIDER"
            decision = "DEFER"
        else:
            recommendation = "❌ NOT RECOMMENDED"
            decision = "DROP"
        
        result = f"""Knowledge Base Evaluation:

Video: {title}
Channel: {channel}
Duration: {duration_minutes:.1f} minutes
Views: {view_count:,} | Likes: {like_count:,} | Comments: {comment_count:,}

📊 Quality Score: {score}/100

🔍 Analysis:
{chr(10).join(reasons)}

📋 Final Assessment:
- Recommendation: {recommendation}
- Suggested Action: {decision}
- Knowledge Base Worthiness: {score}%

💡 Notes:
- Scores 70+ indicate high-value educational content
- Scores 50-69 suggest good potential with manual review
- Scores 30-49 may be useful for specific topics
- Scores <30 generally not suitable for knowledge base

Video URL: https://www.youtube.com/watch?v={video_id}
"""
        
        return result
        
    except Exception as e:
        return f"Error evaluating video: {str(e)}"

@mcp.tool()
async def get_video_transcript(video_input: str, language: str = "en") -> str:
    """
    Extract transcript from a YouTube video.
    
    Args:
        video_input: YouTube video URL or video ID
        language: Language code for transcript (default: "en")
        
    Returns:
        Formatted string with video transcript or error message
    """
    video_id = get_video_id_from_url(video_input)
    if not video_id:
        return f"Error: Could not extract video ID from '{video_input}'. Please provide a valid YouTube URL or 11-character video ID."
    
    # Check library availability and provide installation guidance
    if not TRANSCRIPT_API_AVAILABLE:
        return f"""YouTube Video Transcript - Installation Required:

Video ID: {video_id}
Status: ❌ Missing Dependency

The 'youtube-transcript-api' library is required for transcript extraction.

🔧 INSTALLATION COMMAND:
pip install youtube-transcript-api

After installation:
1. Restart the MCP server
2. Test this function again

Alternative: Use get_video_details() for basic video information.

Video URL: https://www.youtube.com/watch?v={video_id}

Note: Once installed, this function will extract full transcript content with timestamps."""
    
    # Library is available - proceed with transcript extraction
    try:
        # Get video title for context
        try:
            video_data = await make_youtube_api_request("videos", {
                "part": "snippet",
                "id": video_id
            })
            video_title = video_data["items"][0]["snippet"]["title"] if video_data.get("items") else "Unknown Video"
        except:
            video_title = "Unknown Video"
        
        # Try to get transcript in requested language
        transcript = None
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        except:
            # Fallback to English
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_transcript(['en']).fetch()
                language = 'en'
            except:
                # Try any available transcript
                try:
                    available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                    first_transcript = next(iter(available_transcripts))
                    transcript = first_transcript.fetch()
                    language = first_transcript.language_code
                except:
                    return f"""YouTube Video Transcript - No Transcripts Available:

Video: {video_title}
Video ID: {video_id}

❌ No transcripts found for this video.

Possible reasons:
• Video owner has disabled captions
• Video is too new (captions not yet generated)
• Video is restricted in your region
• Video is private or deleted

Try: Use get_video_details() for basic video information.

Video URL: https://www.youtube.com/watch?v={video_id}"""
        
        if not transcript:
            return f"No transcript content extracted for video '{video_id}'."
        
        # Format transcript content
        formatted_segments = []
        for entry in transcript:
            timestamp = f"[{int(entry['start']//60):02d}:{int(entry['start']%60):02d}]"
            formatted_segments.append(f"{timestamp} {entry['text']}")
        
        full_text = " ".join([entry['text'] for entry in transcript])
        
        # Calculate statistics
        word_count = len(full_text.split())
        duration_minutes = int(transcript[-1]['start']//60) if transcript else 0
        
        # Build comprehensive response
        result = f"""YouTube Video Transcript:

Video: {video_title}
Video ID: {video_id}
Language: {language.upper()}
Duration: ~{duration_minutes} minutes
Segments: {len(transcript)}
Word Count: ~{word_count} words

📝 Full Transcript:
{full_text}

⏰ Timestamped Segments (First 10):
{chr(10).join(formatted_segments[:10])}
{'... and ' + str(len(formatted_segments) - 10) + ' more segments' if len(formatted_segments) > 10 else ''}

Video URL: https://www.youtube.com/watch?v={video_id}

✅ Transcript successfully extracted using youtube-transcript-api.
Note: Quality depends on YouTube's automatic or manual captions."""
        
        return result
        
    except Exception as e:
        # Comprehensive error handling
        error_message = str(e).lower()
        
        if "transcriptsdisabled" in error_message or "disabled" in error_message:
            return f"""YouTube Video Transcript - Transcripts Disabled:

Video: {video_title}
Video ID: {video_id}

❌ Transcripts are disabled for this video.

The video owner has disabled captions/transcripts.

Alternatives:
• Try get_video_details() for basic video information
• Look for similar videos with transcripts enabled

Video URL: https://www.youtube.com/watch?v={video_id}"""
        elif "quota" in error_message:
            return f"❌ YouTube API quota exceeded. Please try again later."
        elif "forbidden" in error_message:
            return f"❌ Access to video '{video_id}' is restricted or private."
        else:
            return f"Error extracting transcript for video '{video_id}': {str(e)}"

# Add resource for server information
@mcp.resource("youtube://server/info")
def get_server_info() -> str:
    """Get information about this YouTube MCP server."""
    return """YouTube MCP Server - Integrated with YouTube Agent System

This server provides access to YouTube data via the YouTube Data API v3.
Adapted for the YouTube Agent System project structure.

Available Tools:
1. get_video_details(video_input) - Get detailed information about a YouTube video
2. search_videos(query, max_results, order) - Search YouTube for videos by keywords
3. analyze_video_engagement(video_input) - Analyze video engagement metrics and insights
4. evaluate_video_for_knowledge_base(video_input) - Evaluate video for knowledge base inclusion
5. get_video_transcript(video_input, language) - Extract transcript content from videos

Core Tools for Agent System:
- Video discovery and analysis
- Quality assessment for knowledge base curation  
- Engagement metrics for autonomous decision making
- Transcript extraction for content processing

Supported URL formats:
- Videos: https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID
- Direct video IDs: 11-character alphanumeric strings

Configuration:
- Uses project credentials.yml for YouTube API key
- Integrated with existing project settings and structure

Environment Requirements:
- YouTube API key in credentials.yml (youtube: 'your_key')
- httpx for HTTP requests
- youtube-transcript-api for transcript extraction (optional)

Note: This is a focused subset of MCP tools optimized for the agent system's core functionality.
Additional tools available in full MCP server if needed.
"""

if __name__ == "__main__":
    # For MCP protocol, we can't print to stdout - it must only contain JSON
    # The API key check will happen when tools are called
    mcp.run() 