"""
MCP integration layer for YouTube Agent System.
Enhanced with intelligent video analysis and ephemeral transcript management.
"""

# MCP integration modules
from .tool_registry import MCPToolRegistry, create_mcp_tool_registry
from .ephemeral_manager import (
    EphemeralTranscriptManager,
    EphemeralTranscriptManagerError,
    EphemeralTranscript,
    DecisionStatus,
    create_ephemeral_transcript_manager
)
from .analysis_layer import (
    AnalysisLayer,
    AnalysisLayerError,
    AnalysisStrategy,
    AnalysisRequest,
    AnalysisResult,
    create_analysis_layer
)

__all__ = [
    # Tool Registry
    'MCPToolRegistry',
    'create_mcp_tool_registry',
    
    # Ephemeral Manager
    'EphemeralTranscriptManager',
    'EphemeralTranscriptManagerError',
    'EphemeralTranscript',
    'DecisionStatus',
    'create_ephemeral_transcript_manager',
    
    # Analysis Layer
    'AnalysisLayer',
    'AnalysisLayerError',
    'AnalysisStrategy',
    'AnalysisRequest',
    'AnalysisResult',
    'create_analysis_layer',
] 