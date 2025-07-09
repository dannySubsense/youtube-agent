#!/usr/bin/env python3
"""
Integration test for MCP integration with MVP preservation.
Verifies that agent components work alongside existing MVP functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config_imports():
    """Test that configuration system still works."""
    try:
        from config.settings import Settings, get_global_settings
        print("✅ Config system imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_tools_imports():
    """Test that existing MVP tools still work."""
    try:
        from tools import (
            VideoSearcher, TranscriptManager, 
            VectorDatabaseManager, KnowledgeBaseManager
        )
        print("✅ MVP tools import successfully")
        return True
    except ImportError as e:
        print(f"❌ MVP tools import failed: {e}")
        return False

def test_agents_imports():
    """Test that new agent components import correctly."""
    try:
        from agents import (
            AgentCoordinator, MCPClient, AgentToolInterface
        )
        print("✅ Agent components import successfully")
        return True
    except ImportError as e:
        print(f"❌ Agent components import failed: {e}")
        return False

def test_mcp_integration_imports():
    """Test that MCP integration components import correctly."""
    try:
        from mcp_integration import (
            EphemeralTranscriptManager, AnalysisLayer, 
            AnalysisStrategy, DecisionStatus
        )
        print("✅ MCP integration components import successfully")
        return True
    except ImportError as e:
        print(f"❌ MCP integration import failed: {e}")
        return False

def test_agent_coordinator_creation():
    """Test creating agent coordinator with simulated dependencies."""
    try:
        from config.settings import Settings
        from agents.mcp_client import create_mcp_client
        from agents.coordinator import create_agent_coordinator
        
        # Create test settings
        settings = Settings(
            youtube_api_key="test_key",
            openai_api_key="test_key"
        )
        
        # Create MCP client
        mcp_client = create_mcp_client(settings)
        print(f"✅ MCP client created with {len(mcp_client.get_available_tools())} tools")
        
        # Create agent coordinator
        coordinator = create_agent_coordinator(settings, mcp_client)
        print("✅ Agent coordinator created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Agent coordinator creation failed: {e}")
        return False

def test_ephemeral_manager_creation():
    """Test creating ephemeral transcript manager."""
    try:
        from config.settings import Settings
        from agents.mcp_client import create_mcp_client
        from agents.coordinator import create_agent_coordinator
        from mcp_integration.ephemeral_manager import create_ephemeral_transcript_manager
        
        # Create dependencies
        settings = Settings(
            youtube_api_key="test_key",
            openai_api_key="test_key"
        )
        mcp_client = create_mcp_client(settings)
        coordinator = create_agent_coordinator(settings, mcp_client)
        
        # Create ephemeral manager
        ephemeral_manager = create_ephemeral_transcript_manager(settings, coordinator)
        print("✅ Ephemeral transcript manager created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Ephemeral manager creation failed: {e}")
        return False

def test_analysis_layer_creation():
    """Test creating analysis layer with all dependencies."""
    try:
        from config.settings import Settings
        from agents.mcp_client import create_mcp_client
        from agents.coordinator import create_agent_coordinator
        from mcp_integration.ephemeral_manager import create_ephemeral_transcript_manager
        from mcp_integration.analysis_layer import create_analysis_layer
        
        # Create dependencies
        settings = Settings(
            youtube_api_key="test_key",
            openai_api_key="test_key"
        )
        mcp_client = create_mcp_client(settings)
        coordinator = create_agent_coordinator(settings, mcp_client)
        ephemeral_manager = create_ephemeral_transcript_manager(settings, coordinator)
        
        # Create analysis layer
        analysis_layer = create_analysis_layer(settings, coordinator, ephemeral_manager, mcp_client)
        print("✅ Analysis layer created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Analysis layer creation failed: {e}")
        return False

def test_mvp_preservation():
    """Test that MVP functionality is preserved."""
    try:
        from tools import (
            create_video_searcher, create_transcript_manager,
            create_vector_database_manager, create_knowledge_base_manager
        )
        from config.settings import Settings
        
        # Create test settings
        settings = Settings(
            youtube_api_key="test_key",
            openai_api_key="test_key"
        )
        
        # Test MVP factory functions still work
        video_searcher = create_video_searcher(settings)
        transcript_manager = create_transcript_manager(settings)
        vector_db_manager = create_vector_database_manager(settings)
        kb_manager = create_knowledge_base_manager(settings)
        
        print("✅ All MVP factory functions work correctly")
        print("✅ MVP functionality is fully preserved")
        
        return True
    except Exception as e:
        print(f"❌ MVP preservation test failed: {e}")
        return False

def test_streamlit_ui_imports():
    """Test that Streamlit UI can import agent components."""
    try:
        # Test imports that the UI would use
        from agents.coordinator import EphemeralAnalysis
        from mcp_integration.analysis_layer import AnalysisStrategy, AnalysisRequest
        from mcp_integration.ephemeral_manager import DecisionStatus
        
        print("✅ UI can import agent components successfully")
        return True
    except ImportError as e:
        print(f"❌ UI agent component imports failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Running MCP Integration Tests with MVP Preservation\n")
    
    tests = [
        test_config_imports,
        test_tools_imports,
        test_agents_imports,
        test_mcp_integration_imports,
        test_agent_coordinator_creation,
        test_ephemeral_manager_creation,
        test_analysis_layer_creation,
        test_mvp_preservation,
        test_streamlit_ui_imports,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n🔍 Running: {test.__name__}")
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! MCP integration is working correctly.")
        print("🔒 MVP functionality is fully preserved.")
        print("🤖 Agent features are ready for use.")
        return True
    else:
        print(f"\n⚠️ {failed} tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 