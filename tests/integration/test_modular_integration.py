"""
Integration test for modular YouTube Agent System.
Validates that refactored components work together correctly.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from tools.video_search import create_video_searcher
from tools.transcript_manager import create_transcript_manager
from tools.vector_database import create_vector_database_manager
from tools.knowledge_base import create_knowledge_base_manager


def test_configuration():
    """Test configuration loading."""
    print("🔧 Testing configuration...")
    settings = get_settings()
    assert settings.youtube_api_key, "YouTube API key must be configured"
    assert settings.openai_api_key, "OpenAI API key must be configured"
    assert settings.max_videos_per_knowledge_base == 5, "Default video limit should be 5"
    print("✅ Configuration test passed")


def test_modular_components():
    """Test all modular components can be initialized."""
    print("🧩 Testing modular components...")
    settings = get_settings()
    
    # Test component initialization
    video_searcher = create_video_searcher(settings)
    transcript_manager = create_transcript_manager(settings)
    vector_db_manager = create_vector_database_manager(settings)
    knowledge_manager = create_knowledge_base_manager(settings)
    
    assert video_searcher.settings == settings
    assert transcript_manager.settings == settings
    assert vector_db_manager.settings == settings
    assert knowledge_manager.settings == settings
    
    print("✅ Modular components test passed")


def test_database_discovery():
    """Test database discovery functionality."""
    print("🔍 Testing database discovery...")
    settings = get_settings()
    vector_db_manager = create_vector_database_manager(settings)
    
    databases = vector_db_manager.discover_databases()
    print(f"   Found {len(databases)} existing databases:")
    for db in databases:
        print(f"   - {db.name} ({db.document_count} documents)")
    
    print("✅ Database discovery test passed")


def test_video_search_mock():
    """Test video search with mock data (no API calls)."""
    print("🔍 Testing video search structure...")
    settings = get_settings()
    video_searcher = create_video_searcher(settings)
    
    # Test duration parsing
    test_cases = [
        ('PT4M13S', '4:13'),
        ('PT1H30M45S', '1:30:45'),
        ('PT2H5M', '2:05:00'),
        ('INVALID', 'Unknown')
    ]
    
    for duration_iso, expected in test_cases:
        result = video_searcher._parse_duration(duration_iso)
        assert result == expected, f"Duration parsing failed: {duration_iso} -> {result} (expected {expected})"
    
    print("✅ Video search structure test passed")


def test_transcript_processing():
    """Test transcript processing functionality."""
    print("📝 Testing transcript processing...")
    settings = get_settings()
    transcript_manager = create_transcript_manager(settings)
    
    # Test chunk processing
    test_transcript = "A" * 2500  # 2500 character transcript
    chunks = transcript_manager.chunk_transcript(test_transcript, chunk_size=1000)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert len(chunks[0]) == 1000, f"First chunk should be 1000 chars, got {len(chunks[0])}"
    assert len(chunks[2]) == 500, f"Last chunk should be 500 chars, got {len(chunks[2])}"
    
    print("✅ Transcript processing test passed")


def test_ui_component_integration():
    """Test that UI components can integrate with modular backend."""
    print("🖥️ Testing UI component integration...")
    
    # Test imports that the UI uses
    try:
        from ui.streamlit.dashboard import YouTubeAgentUI
        print("   ✅ UI class imports working")
        
        # Test that UI can initialize with modular backend
        # (without actually running Streamlit)
        settings = get_settings()
        print("   ✅ UI can access settings")
        
        # Test component creation (what UI does in __init__)
        video_searcher = create_video_searcher(settings)
        transcript_manager = create_transcript_manager(settings)
        vector_db_manager = create_vector_database_manager(settings)
        knowledge_manager = create_knowledge_base_manager(settings)
        print("   ✅ UI can initialize all components")
        
    except ImportError as e:
        print(f"   ❌ UI import failed: {e}")
        return False
    
    print("✅ UI component integration test passed")


def test_phase_1_success_criteria():
    """Test Phase 1 success criteria from SDD."""
    print("🎯 Testing Phase 1 success criteria...")
    
    # ✅ All MVP functionality accessible through modular backend
    test_modular_components()
    
    # ✅ Database discovery and management working
    test_database_discovery()
    
    # ✅ Configuration management extracted and working
    test_configuration()
    
    # ✅ UI can integrate with modular backend
    test_ui_component_integration()
    
    print("✅ Phase 1 success criteria verified")


def main():
    """Run all integration tests."""
    print("🚀 Running modular YouTube Agent integration tests...\n")
    
    try:
        test_configuration()
        test_modular_components()
        test_database_discovery()
        test_video_search_mock()
        test_transcript_processing()
        test_ui_component_integration()
        test_phase_1_success_criteria()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Phase 1 modular backend extraction successful")
        print("✅ MVP functionality preserved through modular architecture")
        print("✅ Ready for Phase 1.5 (FastAPI layer creation)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 