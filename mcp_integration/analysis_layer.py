"""
Analysis layer for YouTube Agent System.
Coordinates video analysis between MCP tools and MVP workflow.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from config.settings import Settings
from agents.coordinator import AgentCoordinator, EphemeralAnalysis
from agents.mcp_client import MCPClient
from .ephemeral_manager import EphemeralTranscriptManager, EphemeralTranscript, DecisionStatus


class AnalysisStrategy(Enum):
    """Video analysis strategy options."""
    AUTONOMOUS_ONLY = "autonomous_only"  # Agent decides everything
    HYBRID_REVIEW = "hybrid_review"      # Agent analyzes, user reviews
    MVP_FALLBACK = "mvp_fallback"        # Use MVP workflow only


@dataclass
class AnalysisRequest:
    """Request for video analysis."""
    video_urls: List[str]
    strategy: AnalysisStrategy
    topic: str
    max_videos: int = 5
    quality_threshold: float = 70.0
    user_notes: Optional[str] = None


@dataclass
class AnalysisResult:
    """Result of video analysis coordination."""
    request: AnalysisRequest
    ephemeral_analyses: List[EphemeralTranscript]
    autonomous_decisions: List[str]  # Video IDs agent decided to keep
    user_review_required: List[str]  # Video IDs requiring user review
    mvp_fallback_videos: List[str]   # Video IDs using MVP workflow
    total_processed: int
    success_rate: float
    analysis_time: float
    recommendations: List[str]


class AnalysisLayerError(Exception):
    """Custom exception for analysis layer related errors."""
    pass


class AnalysisLayer:
    """
    Coordinates video analysis between MCP tools and MVP workflow.
    
    Key responsibilities:
    - Orchestrate autonomous agent analysis
    - Manage ephemeral transcript workflow
    - Coordinate user decision points
    - Provide fallback to MVP workflow
    - Integrate analysis results
    """
    
    def __init__(
        self,
        settings: Settings,
        agent_coordinator: AgentCoordinator,
        ephemeral_manager: EphemeralTranscriptManager,
        mcp_client: MCPClient
    ):
        """
        Initialize analysis layer.
        
        Args:
            settings: Application settings
            agent_coordinator: Agent coordinator instance
            ephemeral_manager: Ephemeral transcript manager
            mcp_client: MCP client for tool access
        """
        self.settings = settings
        self.agent_coordinator = agent_coordinator
        self.ephemeral_manager = ephemeral_manager
        self.mcp_client = mcp_client
        
        # Analysis configuration
        self.autonomous_threshold = 80.0  # Auto-keep videos above this score
        self.drop_threshold = 50.0        # Auto-drop videos below this score
    
    def analyze_videos(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Coordinate video analysis based on strategy.
        
        Args:
            request: Analysis request
            
        Returns:
            Analysis results
            
        Raises:
            AnalysisLayerError: If analysis fails
        """
        try:
            start_time = datetime.now()
            
            # Route to appropriate analysis strategy
            if request.strategy == AnalysisStrategy.AUTONOMOUS_ONLY:
                result = self._analyze_autonomous_only(request)
            elif request.strategy == AnalysisStrategy.HYBRID_REVIEW:
                result = self._analyze_hybrid_review(request)
            elif request.strategy == AnalysisStrategy.MVP_FALLBACK:
                result = self._analyze_mvp_fallback(request)
            else:
                raise AnalysisLayerError(f"Unknown analysis strategy: {request.strategy}")
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            result.analysis_time = analysis_time
            
            return result
            
        except Exception as e:
            raise AnalysisLayerError(f"Video analysis failed: {e}") from e
    
    def _analyze_autonomous_only(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Autonomous analysis - agent decides everything.
        
        Args:
            request: Analysis request
            
        Returns:
            Analysis results
        """
        ephemeral_analyses = []
        autonomous_decisions = []
        processing_errors = 0
        
        for video_url in request.video_urls:
            try:
                # Agent performs ephemeral analysis
                ephemeral_transcript = self.ephemeral_manager.process_video_ephemeral(
                    video_url, request.user_notes
                )
                ephemeral_analyses.append(ephemeral_transcript)
                
                # Agent makes autonomous decision based on quality score
                quality_score = ephemeral_transcript.analysis.quality_score
                
                if quality_score >= self.autonomous_threshold:
                    decision = "keep"
                    autonomous_decisions.append(ephemeral_transcript.video_id)
                elif quality_score <= self.drop_threshold:
                    decision = "drop"
                else:
                    decision = "defer"  # Borderline cases get deferred
                
                # Execute decision
                self.ephemeral_manager.make_decision(
                    ephemeral_transcript.video_id, 
                    decision, 
                    f"Autonomous decision: quality score {quality_score:.1f}"
                )
                
            except Exception as e:
                processing_errors += 1
                print(f"Error processing {video_url}: {e}")
        
        success_rate = (len(ephemeral_analyses) / len(request.video_urls)) * 100
        
        return AnalysisResult(
            request=request,
            ephemeral_analyses=ephemeral_analyses,
            autonomous_decisions=autonomous_decisions,
            user_review_required=[],
            mvp_fallback_videos=[],
            total_processed=len(ephemeral_analyses),
            success_rate=success_rate,
            analysis_time=0.0,  # Will be set by caller
            recommendations=self._generate_recommendations(ephemeral_analyses)
        )
    
    def _analyze_hybrid_review(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Hybrid analysis - agent analyzes, user reviews decisions.
        
        Args:
            request: Analysis request
            
        Returns:
            Analysis results
        """
        ephemeral_analyses = []
        autonomous_decisions = []
        user_review_required = []
        processing_errors = 0
        
        for video_url in request.video_urls:
            try:
                # Agent performs ephemeral analysis
                ephemeral_transcript = self.ephemeral_manager.process_video_ephemeral(
                    video_url, request.user_notes
                )
                ephemeral_analyses.append(ephemeral_transcript)
                
                # Agent makes recommendation, but user reviews
                quality_score = ephemeral_transcript.analysis.quality_score
                
                if quality_score >= self.autonomous_threshold:
                    # High quality - agent auto-keeps
                    decision = "keep"
                    autonomous_decisions.append(ephemeral_transcript.video_id)
                    self.ephemeral_manager.make_decision(
                        ephemeral_transcript.video_id, 
                        decision, 
                        f"Auto-keep: quality score {quality_score:.1f}"
                    )
                elif quality_score <= self.drop_threshold:
                    # Low quality - agent auto-drops
                    decision = "drop"
                    self.ephemeral_manager.make_decision(
                        ephemeral_transcript.video_id, 
                        decision, 
                        f"Auto-drop: quality score {quality_score:.1f}"
                    )
                else:
                    # Borderline - requires user review
                    user_review_required.append(ephemeral_transcript.video_id)
                
            except Exception as e:
                processing_errors += 1
                print(f"Error processing {video_url}: {e}")
        
        success_rate = (len(ephemeral_analyses) / len(request.video_urls)) * 100
        
        return AnalysisResult(
            request=request,
            ephemeral_analyses=ephemeral_analyses,
            autonomous_decisions=autonomous_decisions,
            user_review_required=user_review_required,
            mvp_fallback_videos=[],
            total_processed=len(ephemeral_analyses),
            success_rate=success_rate,
            analysis_time=0.0,  # Will be set by caller
            recommendations=self._generate_recommendations(ephemeral_analyses)
        )
    
    def _analyze_mvp_fallback(self, request: AnalysisRequest) -> AnalysisResult:
        """
        MVP fallback analysis - use existing MVP workflow.
        
        Args:
            request: Analysis request
            
        Returns:
            Analysis results
        """
        # For Phase 1, simulate MVP fallback
        # In full implementation, this would integrate with MVP tools
        mvp_fallback_videos = [self._extract_video_id(url) for url in request.video_urls]
        
        return AnalysisResult(
            request=request,
            ephemeral_analyses=[],
            autonomous_decisions=[],
            user_review_required=[],
            mvp_fallback_videos=mvp_fallback_videos,
            total_processed=len(request.video_urls),
            success_rate=100.0,  # MVP workflow is always successful
            analysis_time=0.0,  # Will be set by caller
            recommendations=["Using MVP workflow - all videos processed through existing pipeline"]
        )
    
    def _generate_recommendations(self, analyses: List[EphemeralTranscript]) -> List[str]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            analyses: List of ephemeral analyses
            
        Returns:
            List of recommendations
        """
        if not analyses:
            return ["No analyses completed"]
        
        recommendations = []
        
        # Quality score analysis
        quality_scores = [a.analysis.quality_score for a in analyses]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        if avg_quality >= 80:
            recommendations.append("Excellent video quality detected - consider expanding search")
        elif avg_quality >= 60:
            recommendations.append("Good video quality - recommend manual review of borderline cases")
        else:
            recommendations.append("Low average quality - consider refining search criteria")
        
        # Decision distribution
        keep_count = len([a for a in analyses if a.decision_status == DecisionStatus.KEEP])
        drop_count = len([a for a in analyses if a.decision_status == DecisionStatus.DROP])
        
        if keep_count > drop_count * 2:
            recommendations.append("High keep rate - search criteria are well-tuned")
        elif drop_count > keep_count * 2:
            recommendations.append("High drop rate - consider adjusting search strategy")
        
        return recommendations
    
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
    
    def get_analysis_summary(self, result: AnalysisResult) -> Dict[str, Any]:
        """
        Get summary of analysis results.
        
        Args:
            result: Analysis result
            
        Returns:
            Summary dictionary
        """
        return {
            "strategy": result.request.strategy.value,
            "total_videos": len(result.request.video_urls),
            "processed": result.total_processed,
            "success_rate": result.success_rate,
            "analysis_time": result.analysis_time,
            "autonomous_decisions": len(result.autonomous_decisions),
            "user_review_required": len(result.user_review_required),
            "mvp_fallback": len(result.mvp_fallback_videos),
            "recommendations": result.recommendations
        }
    
    def process_user_decisions(self, decisions: Dict[str, str]) -> Dict[str, Any]:
        """
        Process batch user decisions for pending analyses.
        
        Args:
            decisions: Dictionary of video_id -> decision
            
        Returns:
            Processing results
        """
        results = {
            "processed": 0,
            "errors": 0,
            "decisions": {"keep": 0, "drop": 0, "defer": 0}
        }
        
        for video_id, decision in decisions.items():
            try:
                self.ephemeral_manager.make_decision(video_id, decision, "User decision")
                results["processed"] += 1
                results["decisions"][decision] += 1
            except Exception as e:
                results["errors"] += 1
                print(f"Error processing decision for {video_id}: {e}")
        
        return results


def create_analysis_layer(
    settings: Settings,
    agent_coordinator: AgentCoordinator,
    ephemeral_manager: EphemeralTranscriptManager,
    mcp_client: MCPClient
) -> AnalysisLayer:
    """
    Factory function to create analysis layer.
    
    Args:
        settings: Application settings
        agent_coordinator: Agent coordinator instance
        ephemeral_manager: Ephemeral transcript manager
        mcp_client: MCP client instance
        
    Returns:
        Configured analysis layer
    """
    return AnalysisLayer(settings, agent_coordinator, ephemeral_manager, mcp_client) 