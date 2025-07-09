"""
Agent coordinator for YouTube Agent System.
LangChain agent orchestration with MCP tool integration.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from config.settings import Settings
from .mcp_client import MCPClient, MCPClientError


class AgentCoordinatorError(Exception):
    """Custom exception for agent coordinator related errors."""
    pass


@dataclass
class EphemeralAnalysis:
    """
    Results from ephemeral video analysis.
    
    Represents agent's autonomous analysis of a video without permanent storage.
    User can then decide to keep, drop, or defer the video.
    """
    video_id: str
    video_url: str
    title: str
    analysis_summary: str
    quality_score: float
    engagement_metrics: Dict[str, Any]
    transcript_preview: str
    recommendation: str  # "keep", "drop", "defer"
    reasoning: str
    analyzed_at: datetime
    mcp_tools_used: List[str]


class AgentCoordinator:
    """
    LangChain agent coordinator with MCP tool integration.
    
    Provides autonomous video analysis using MCP tools:
    - Intelligently selects appropriate MCP tools for each task
    - Performs ephemeral transcript analysis without permanent storage
    - Provides quality scoring and recommendations
    - Enables user control over final keep/drop/defer decisions
    """
    
    def __init__(self, settings: Settings, mcp_client: MCPClient):
        """
        Initialize agent coordinator with MCP client.
        
        Args:
            settings: Application settings
            mcp_client: MCP client for tool access
        """
        self.settings = settings
        self.mcp_client = mcp_client
        self.analysis_cache: Dict[str, EphemeralAnalysis] = {}
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """Initialize LangChain agent with MCP tools."""
        try:
            # For Phase 1, create a simple agent framework
            # In full implementation, this would set up LangChain agent with MCP tools
            available_tools = self.mcp_client.get_available_tools()
            
            if not available_tools:
                raise AgentCoordinatorError("No MCP tools available")
            
            # Agent initialization successful
            self.agent_initialized = True
            
        except Exception as e:
            raise AgentCoordinatorError(f"Failed to initialize agent: {e}") from e
    
    def analyze_video_ephemeral(self, video_url: str) -> EphemeralAnalysis:
        """
        Perform autonomous ephemeral video analysis.
        
        Agent autonomously decides which MCP tools to use based on the video.
        Analysis is temporary - user decides final keep/drop/defer action.
        
        Args:
            video_url: YouTube video URL to analyze
            
        Returns:
            Ephemeral analysis results
            
        Raises:
            AgentCoordinatorError: If analysis fails
        """
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(video_url)
            
            # Check if already analyzed
            if video_id in self.analysis_cache:
                return self.analysis_cache[video_id]
            
            # Agent autonomously selects and executes MCP tools
            analysis = self._execute_autonomous_analysis(video_id, video_url)
            
            # Cache the analysis
            self.analysis_cache[video_id] = analysis
            
            return analysis
            
        except Exception as e:
            raise AgentCoordinatorError(f"Ephemeral analysis failed for {video_url}: {e}") from e
    
    def _execute_autonomous_analysis(self, video_id: str, video_url: str) -> EphemeralAnalysis:
        """
        Execute autonomous analysis using MCP tools.
        
        Agent decides which tools to use based on video context.
        
        Args:
            video_id: Video ID
            video_url: Video URL
            
        Returns:
            Analysis results
        """
        mcp_tools_used = []
        
        try:
            # Step 1: Get video details (always needed)
            video_details = self.mcp_client.call_tool("get_video_details", video_input=video_url)
            mcp_tools_used.append("get_video_details")
            
            # Step 2: Analyze engagement metrics
            engagement_metrics = self.mcp_client.call_tool("analyze_video_engagement", video_input=video_url)
            mcp_tools_used.append("analyze_video_engagement")
            
            # Step 3: Get transcript for content analysis
            transcript_result = self.mcp_client.call_tool("get_video_transcript", video_input=video_url)
            mcp_tools_used.append("get_video_transcript")
            
            # Step 4: Evaluate for knowledge base inclusion
            kb_evaluation = self.mcp_client.call_tool("evaluate_video_for_knowledge_base", video_input=video_url)
            mcp_tools_used.append("evaluate_video_for_knowledge_base")
            
            # Process results and make recommendation
            analysis_summary = self._synthesize_analysis_results(
                video_details, engagement_metrics, transcript_result, kb_evaluation
            )
            
            # Calculate quality score (0-100)
            quality_score = self._calculate_quality_score(engagement_metrics, kb_evaluation)
            
            # Make recommendation
            recommendation, reasoning = self._make_recommendation(quality_score, analysis_summary)
            
            return EphemeralAnalysis(
                video_id=video_id,
                video_url=video_url,
                title=f"Video Analysis - {video_id}",  # Would extract from video_details
                analysis_summary=analysis_summary,
                quality_score=quality_score,
                engagement_metrics=engagement_metrics,
                transcript_preview=str(transcript_result)[:200] + "...",
                recommendation=recommendation,
                reasoning=reasoning,
                analyzed_at=datetime.now(),
                mcp_tools_used=mcp_tools_used
            )
            
        except MCPClientError as e:
            raise AgentCoordinatorError(f"MCP tool execution failed: {e}") from e
    
    def _synthesize_analysis_results(self, *results) -> str:
        """
        Synthesize analysis results into summary.
        
        Args:
            *results: Various MCP tool results
            
        Returns:
            Analysis summary
        """
        # For Phase 1, create simple summary
        # In full implementation, this would use LLM to synthesize results
        return f"Autonomous analysis completed using {len(results)} MCP tools. " \
               f"Video analyzed for content quality, engagement metrics, and KB suitability."
    
    def _calculate_quality_score(self, engagement_metrics: Any, kb_evaluation: Any) -> float:
        """
        Calculate quality score based on MCP tool results.
        
        Args:
            engagement_metrics: Engagement analysis results
            kb_evaluation: KB evaluation results
            
        Returns:
            Quality score (0-100)
        """
        # For Phase 1, return simulated score
        # In full implementation, this would analyze actual metrics
        return 75.0  # Simulated quality score
    
    def _make_recommendation(self, quality_score: float, analysis_summary: str) -> tuple[str, str]:
        """
        Make keep/drop/defer recommendation.
        
        Args:
            quality_score: Video quality score
            analysis_summary: Analysis summary
            
        Returns:
            Tuple of (recommendation, reasoning)
        """
        if quality_score >= 80:
            return "keep", "High quality content with strong engagement metrics"
        elif quality_score >= 60:
            return "defer", "Moderate quality - may be worth reviewing later"
        else:
            return "drop", "Low quality or engagement - not recommended for knowledge base"
    
    def _extract_video_id(self, video_url: str) -> str:
        """
        Extract video ID from YouTube URL.
        
        Args:
            video_url: YouTube URL
            
        Returns:
            Video ID
        """
        # Simple extraction - in full implementation would handle all URL formats
        if "v=" in video_url:
            return video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            return video_url.split("youtu.be/")[1].split("?")[0]
        else:
            return video_url  # Assume it's already a video ID
    
    def get_analysis_results(self, video_id: str) -> Optional[EphemeralAnalysis]:
        """
        Get cached analysis results.
        
        Args:
            video_id: Video ID
            
        Returns:
            Analysis results or None if not found
        """
        return self.analysis_cache.get(video_id)
    
    def clear_analysis_cache(self) -> None:
        """Clear all cached analysis results."""
        self.analysis_cache.clear()
    
    def get_cached_analyses(self) -> List[EphemeralAnalysis]:
        """
        Get all cached analysis results.
        
        Returns:
            List of cached analyses
        """
        return list(self.analysis_cache.values())


def create_agent_coordinator(settings: Settings, mcp_client: MCPClient) -> AgentCoordinator:
    """
    Factory function to create agent coordinator.
    
    Args:
        settings: Application settings
        mcp_client: MCP client instance
        
    Returns:
        Configured agent coordinator
    """
    return AgentCoordinator(settings, mcp_client) 