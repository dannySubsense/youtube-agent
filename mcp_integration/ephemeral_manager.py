"""
Ephemeral transcript manager for YouTube Agent System.
Handles temporary video processing for agent analysis without permanent storage.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from config.settings import Settings
from agents.coordinator import EphemeralAnalysis, AgentCoordinator


class DecisionStatus(Enum):
    """Status of user decision on ephemeral analysis."""
    PENDING = "pending"
    KEEP = "keep"
    DROP = "drop"
    DEFER = "defer"


@dataclass
class EphemeralTranscript:
    """
    Temporary transcript data for agent analysis.
    
    This is not stored permanently - exists only for user decision making.
    """
    video_id: str
    video_url: str
    transcript_text: str
    analysis: EphemeralAnalysis
    decision_status: DecisionStatus
    created_at: datetime
    expires_at: datetime
    user_notes: Optional[str] = None


class EphemeralTranscriptManagerError(Exception):
    """Custom exception for ephemeral transcript manager related errors."""
    pass


class EphemeralTranscriptManager:
    """
    Manages temporary transcript processing for agent analysis.
    
    Key features:
    - Temporary storage of video transcripts for agent analysis
    - User decision tracking (keep/drop/defer)
    - Automatic cleanup of expired entries
    - No permanent storage - preserves MVP workflow integrity
    """
    
    def __init__(self, settings: Settings, agent_coordinator: AgentCoordinator):
        """
        Initialize ephemeral transcript manager.
        
        Args:
            settings: Application settings
            agent_coordinator: Agent coordinator for analysis
        """
        self.settings = settings
        self.agent_coordinator = agent_coordinator
        self.ephemeral_transcripts: Dict[str, EphemeralTranscript] = {}
        self.expiry_hours = 24  # Ephemeral transcripts expire after 24 hours
    
    def process_video_ephemeral(self, video_url: str, user_notes: Optional[str] = None) -> EphemeralTranscript:
        """
        Process video for ephemeral analysis.
        
        Agent autonomously analyzes video, user decides final action.
        
        Args:
            video_url: YouTube video URL
            user_notes: Optional user notes
            
        Returns:
            Ephemeral transcript with analysis
            
        Raises:
            EphemeralTranscriptManagerError: If processing fails
        """
        try:
            # Extract video ID
            video_id = self._extract_video_id(video_url)
            
            # Check if already processed
            if video_id in self.ephemeral_transcripts:
                return self.ephemeral_transcripts[video_id]
            
            # Agent performs autonomous analysis
            analysis = self.agent_coordinator.analyze_video_ephemeral(video_url)
            
            # Create ephemeral transcript (simulated transcript text)
            transcript_text = f"[Ephemeral transcript for {video_id}]\n{analysis.transcript_preview}"
            
            # Create ephemeral transcript entry
            now = datetime.now()
            ephemeral_transcript = EphemeralTranscript(
                video_id=video_id,
                video_url=video_url,
                transcript_text=transcript_text,
                analysis=analysis,
                decision_status=DecisionStatus.PENDING,
                created_at=now,
                expires_at=now + timedelta(hours=self.expiry_hours),
                user_notes=user_notes
            )
            
            # Store temporarily
            self.ephemeral_transcripts[video_id] = ephemeral_transcript
            
            return ephemeral_transcript
            
        except Exception as e:
            raise EphemeralTranscriptManagerError(f"Ephemeral processing failed for {video_url}: {e}") from e
    
    def make_decision(self, video_id: str, decision: str, user_notes: Optional[str] = None) -> None:
        """
        User/agent decides: 'keep', 'drop', 'defer'
        
        This is the key decision point from the SDD where user controls final action.
        
        Args:
            video_id: Video ID
            decision: Decision string ('keep', 'drop', 'defer')
            user_notes: Optional user notes
            
        Raises:
            EphemeralTranscriptManagerError: If decision processing fails
        """
        try:
            if video_id not in self.ephemeral_transcripts:
                raise EphemeralTranscriptManagerError(f"No ephemeral transcript found for video {video_id}")
            
            # Validate decision
            valid_decisions = ["keep", "drop", "defer"]
            if decision not in valid_decisions:
                raise EphemeralTranscriptManagerError(f"Invalid decision '{decision}'. Must be one of: {valid_decisions}")
            
            # Update decision status
            ephemeral_transcript = self.ephemeral_transcripts[video_id]
            ephemeral_transcript.decision_status = DecisionStatus(decision)
            
            if user_notes:
                ephemeral_transcript.user_notes = user_notes
            
            # Process decision
            if decision == "keep":
                self._process_keep_decision(ephemeral_transcript)
            elif decision == "drop":
                self._process_drop_decision(ephemeral_transcript)
            elif decision == "defer":
                self._process_defer_decision(ephemeral_transcript)
            
        except Exception as e:
            raise EphemeralTranscriptManagerError(f"Decision processing failed: {e}") from e
    
    def _process_keep_decision(self, ephemeral_transcript: EphemeralTranscript) -> None:
        """
        Process 'keep' decision - add to permanent knowledge base.
        
        Args:
            ephemeral_transcript: Ephemeral transcript to process
        """
        # For Phase 1, log the decision
        # In full implementation, this would integrate with MVP tools to add to KB
        print(f"KEEP decision for {ephemeral_transcript.video_id}: {ephemeral_transcript.analysis.reasoning}")
    
    def _process_drop_decision(self, ephemeral_transcript: EphemeralTranscript) -> None:
        """
        Process 'drop' decision - discard analysis.
        
        Args:
            ephemeral_transcript: Ephemeral transcript to process
        """
        # For Phase 1, log the decision
        print(f"DROP decision for {ephemeral_transcript.video_id}: {ephemeral_transcript.analysis.reasoning}")
    
    def _process_defer_decision(self, ephemeral_transcript: EphemeralTranscript) -> None:
        """
        Process 'defer' decision - extend expiry time.
        
        Args:
            ephemeral_transcript: Ephemeral transcript to process
        """
        # Extend expiry by another 24 hours
        ephemeral_transcript.expires_at = datetime.now() + timedelta(hours=self.expiry_hours)
        print(f"DEFER decision for {ephemeral_transcript.video_id}: Extended expiry to {ephemeral_transcript.expires_at}")
    
    def get_pending_analyses(self) -> List[EphemeralTranscript]:
        """
        Get all pending ephemeral analyses.
        
        Returns:
            List of pending analyses
        """
        self._cleanup_expired()
        return [
            transcript for transcript in self.ephemeral_transcripts.values()
            if transcript.decision_status == DecisionStatus.PENDING
        ]
    
    def get_analyses_by_status(self, status: DecisionStatus) -> List[EphemeralTranscript]:
        """
        Get analyses by decision status.
        
        Args:
            status: Decision status to filter by
            
        Returns:
            List of analyses with specified status
        """
        self._cleanup_expired()
        return [
            transcript for transcript in self.ephemeral_transcripts.values()
            if transcript.decision_status == status
        ]
    
    def get_analysis(self, video_id: str) -> Optional[EphemeralTranscript]:
        """
        Get ephemeral analysis by video ID.
        
        Args:
            video_id: Video ID
            
        Returns:
            Ephemeral transcript or None if not found
        """
        self._cleanup_expired()
        return self.ephemeral_transcripts.get(video_id)
    
    def _cleanup_expired(self) -> None:
        """Remove expired ephemeral transcripts."""
        now = datetime.now()
        expired_ids = [
            video_id for video_id, transcript in self.ephemeral_transcripts.items()
            if transcript.expires_at < now
        ]
        
        for video_id in expired_ids:
            del self.ephemeral_transcripts[video_id]
    
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get ephemeral transcript statistics.
        
        Returns:
            Statistics dictionary
        """
        self._cleanup_expired()
        
        total = len(self.ephemeral_transcripts)
        pending = len([t for t in self.ephemeral_transcripts.values() if t.decision_status == DecisionStatus.PENDING])
        kept = len([t for t in self.ephemeral_transcripts.values() if t.decision_status == DecisionStatus.KEEP])
        dropped = len([t for t in self.ephemeral_transcripts.values() if t.decision_status == DecisionStatus.DROP])
        deferred = len([t for t in self.ephemeral_transcripts.values() if t.decision_status == DecisionStatus.DEFER])
        
        return {
            "total": total,
            "pending": pending,
            "kept": kept,
            "dropped": dropped,
            "deferred": deferred
        }
    
    def clear_all(self) -> None:
        """Clear all ephemeral transcripts."""
        self.ephemeral_transcripts.clear()


def create_ephemeral_transcript_manager(settings: Settings, agent_coordinator: AgentCoordinator) -> EphemeralTranscriptManager:
    """
    Factory function to create ephemeral transcript manager.
    
    Args:
        settings: Application settings
        agent_coordinator: Agent coordinator instance
        
    Returns:
        Configured ephemeral transcript manager
    """
    return EphemeralTranscriptManager(settings, agent_coordinator) 