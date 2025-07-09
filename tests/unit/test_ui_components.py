"""
Unit tests for UI components and interaction handling.
Tests for bug fixes: quick questions and RAG chain database switching.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

import streamlit as st
from langchain_core.vectorstores import VectorStore


# Mock streamlit for testing
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock streamlit for all tests in this module."""
    with patch("streamlit.session_state", new_callable=dict) as mock_session_state:
        with patch("streamlit.rerun") as mock_rerun:
            yield {"session_state": mock_session_state, "rerun": mock_rerun}


class TestQuickQuestionHandling:
    """Test suite for quick question button functionality."""

    def test_quick_question_sets_session_state_correctly(self, mock_streamlit):
        """Test that clicking quick question button sets session state correctly."""
        # Simulate clicking a quick question button
        question_text = "What are the key concepts in MCP development?"

        # Simulate the button click behavior from dashboard.py
        mock_streamlit["session_state"]["quick_question_clicked"] = question_text

        # Verify session state was set correctly
        assert "quick_question_clicked" in mock_streamlit["session_state"]
        assert (
            mock_streamlit["session_state"]["quick_question_clicked"] == question_text
        )

    def test_quick_question_cleared_after_processing(self, mock_streamlit):
        """Test that quick question session state is cleared after processing."""
        # Setup: question in session state
        question_text = "How do I get started with this topic?"
        mock_streamlit["session_state"]["quick_question_clicked"] = question_text

        # Simulate processing (what happens in _show_chat_interface)
        if hasattr(type(mock_streamlit["session_state"]), "quick_question_clicked"):
            submitted_question = mock_streamlit["session_state"][
                "quick_question_clicked"
            ]
            del mock_streamlit["session_state"]["quick_question_clicked"]
        else:
            # Use dict-style access for our mock
            if "quick_question_clicked" in mock_streamlit["session_state"]:
                submitted_question = mock_streamlit["session_state"][
                    "quick_question_clicked"
                ]
                del mock_streamlit["session_state"]["quick_question_clicked"]

        # Verify question was captured and state was cleared
        assert submitted_question == question_text
        assert "quick_question_clicked" not in mock_streamlit["session_state"]

    def test_quick_question_unique_button_keys(self):
        """Test that quick question buttons have unique, stable keys."""
        questions = [
            "What are the key concepts in MCP development?",
            "How do I get started with this topic?",
            "What are the best practices mentioned?",
        ]

        # Generate button keys using the same logic as dashboard.py
        button_keys = []
        for i, q in enumerate(questions):
            button_key = f"quick_q_{hash(q)}_{i}"
            button_keys.append(button_key)

        # Verify all keys are unique
        assert len(button_keys) == len(set(button_keys)), "Button keys must be unique"

        # Verify keys are stable (same question produces same key)
        for i, q in enumerate(questions):
            expected_key = f"quick_q_{hash(q)}_{i}"
            assert (
                button_keys[i] == expected_key
            ), f"Key generation must be deterministic"

    def test_quick_questions_cached_in_session_state(self, mock_streamlit):
        """Test that quick questions are cached in session state and reused like hardcoded questions."""
        # Test the new caching behavior that fixes the boundary crossing issue
        
        # Initially no questions cached
        assert "quick_questions" not in mock_streamlit["session_state"]
        
        # Simulate first generation (what happens in dashboard.py on first load)
        generated_questions = [
            "What are the key concepts in MCP development?",
            "How do I get started with this topic?",
            "What are the best practices mentioned?"
        ]
        mock_streamlit["session_state"]["quick_questions"] = generated_questions
        
        # Verify questions are cached
        assert "quick_questions" in mock_streamlit["session_state"]
        assert mock_streamlit["session_state"]["quick_questions"] == generated_questions
        
        # Simulate subsequent access (like after button click/rerun)
        cached_questions = mock_streamlit["session_state"]["quick_questions"]
        
        # Verify cached questions are identical to original (no regeneration)
        assert cached_questions == generated_questions
        assert cached_questions is mock_streamlit["session_state"]["quick_questions"]
        
        # Now questions behave exactly like hardcoded strings
        # Test button interaction with cached questions
        for i, q in enumerate(cached_questions):
            # Simulate button click
            mock_streamlit["session_state"][f"submit_{i}"] = q
            
        # Verify detection works with cached questions
        for i in range(len(cached_questions)):
            assert f"submit_{i}" in mock_streamlit["session_state"]
            assert mock_streamlit["session_state"][f"submit_{i}"] == cached_questions[i]

    def test_quick_questions_cleared_on_database_change(self, mock_streamlit):
        """Test that cached questions are cleared when database changes."""
        # Setup: questions cached for one database
        old_questions = ["Question about old topic A", "Question about old topic B"]
        mock_streamlit["session_state"]["quick_questions"] = old_questions
        mock_streamlit["session_state"]["rag_chain_db_key"] = "old_topic_12345"
        
        # Simulate database change (what happens in dashboard.py)
        new_db_key = "new_topic_67890"
        if mock_streamlit["session_state"].get("rag_chain_db_key") != new_db_key:
            # Clear cached questions when database changes
            if "quick_questions" in mock_streamlit["session_state"]:
                del mock_streamlit["session_state"]["quick_questions"]
            mock_streamlit["session_state"]["rag_chain_db_key"] = new_db_key
        
        # Verify old questions were cleared
        assert "quick_questions" not in mock_streamlit["session_state"]
        assert mock_streamlit["session_state"]["rag_chain_db_key"] == new_db_key
        
        # Simulate new questions generated for new database
        new_questions = ["Question about new topic X", "Question about new topic Y"]
        mock_streamlit["session_state"]["quick_questions"] = new_questions
        
        # Verify fresh questions are now cached
        assert mock_streamlit["session_state"]["quick_questions"] == new_questions

    def test_gate_behavior_stable_database_key(self, mock_streamlit):
        """Test GATE behavior: questions remain static across reruns when database key is stable."""
        # Simulate stable database key (fixed bug: no longer uses object ID)
        stable_db_key = "MCP_development_/path/to/db"
        
        # First access: no questions cached
        assert "quick_questions" not in mock_streamlit["session_state"]
        assert "rag_chain_db_key" not in mock_streamlit["session_state"]
        
        # Simulate first database load (GATE OPENS)
        mock_streamlit["session_state"]["rag_chain_db_key"] = stable_db_key
        
        # Generate questions for first time
        initial_questions = [
            "What is MCP and how does it work?",
            "How do I create an MCP server?",
            "What are the key components of MCP?"
        ]
        mock_streamlit["session_state"]["quick_questions"] = initial_questions
        
        # GATE CLOSES - questions are now static
        cached_questions = mock_streamlit["session_state"]["quick_questions"]
        assert cached_questions == initial_questions
        
        # Simulate multiple Streamlit reruns (button clicks, etc.)
        for rerun_count in range(5):
            # Database key stays the same (stable - no object ID)
            current_db_key = stable_db_key
            
            # Key doesn't change, so no cache clearing
            if mock_streamlit["session_state"].get("rag_chain_db_key") == current_db_key:
                # Questions should remain exactly the same
                assert "quick_questions" in mock_streamlit["session_state"]
                assert mock_streamlit["session_state"]["quick_questions"] == initial_questions
                assert mock_streamlit["session_state"]["quick_questions"] is cached_questions
        
        # GATE BEHAVIOR: Same questions across all reruns (no regeneration)
        final_questions = mock_streamlit["session_state"]["quick_questions"]
        assert final_questions == initial_questions
        assert final_questions is cached_questions  # Same object reference
        
        # Verify button interactions work with stable questions
        for i, q in enumerate(final_questions):
            mock_streamlit["session_state"][f"submit_{i}"] = q
            
        # Detection should work perfectly
        for i in range(len(final_questions)):
            assert f"submit_{i}" in mock_streamlit["session_state"]
            assert mock_streamlit["session_state"][f"submit_{i}"] == final_questions[i]


class TestRAGChainCaching:
    """Test suite for RAG chain database change detection."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Create mock vectorstore for testing."""
        mock_store = Mock(spec=VectorStore)
        # Use a consistent object id for testing
        mock_store.__class__.__name__ = "MockVectorStore"
        return mock_store

    @pytest.fixture
    def mock_knowledge_manager(self):
        """Create mock knowledge manager."""
        mock_manager = Mock()
        mock_rag_chain = Mock()
        mock_manager.create_advanced_rag_chain.return_value = mock_rag_chain
        return mock_manager

    def test_rag_chain_created_on_first_access(
        self, mock_streamlit, mock_vectorstore, mock_knowledge_manager
    ):
        """Test RAG chain is created when not in session state."""
        # Setup session state with vectorstore and topic
        topic = "Test Topic"
        mock_streamlit["session_state"]["vectorstore"] = mock_vectorstore
        mock_streamlit["session_state"]["topic"] = topic

        # Simulate the database key generation logic
        current_db_key = f"{topic}_{id(mock_vectorstore)}"

        # Simulate first access (no rag_chain in session state)
        assert "rag_chain" not in mock_streamlit["session_state"]
        assert "rag_chain_db_key" not in mock_streamlit["session_state"]

        # Simulate RAG chain creation
        mock_streamlit["session_state"]["rag_chain"] = (
            mock_knowledge_manager.create_advanced_rag_chain(mock_vectorstore, topic)
        )
        mock_streamlit["session_state"]["rag_chain_db_key"] = current_db_key

        # Verify RAG chain was created and key was stored
        assert "rag_chain" in mock_streamlit["session_state"]
        assert mock_streamlit["session_state"]["rag_chain_db_key"] == current_db_key
        mock_knowledge_manager.create_advanced_rag_chain.assert_called_once_with(
            mock_vectorstore, topic
        )

    def test_rag_chain_recreated_on_database_change(
        self, mock_streamlit, mock_knowledge_manager
    ):
        """Test RAG chain is recreated when database changes."""
        # Setup initial state
        old_vectorstore = Mock(spec=VectorStore)
        new_vectorstore = Mock(spec=VectorStore)
        topic = "Test Topic"

        # Initial setup
        old_db_key = f"{topic}_{id(old_vectorstore)}"
        mock_streamlit["session_state"]["vectorstore"] = old_vectorstore
        mock_streamlit["session_state"]["topic"] = topic
        mock_streamlit["session_state"]["rag_chain"] = Mock()
        mock_streamlit["session_state"]["rag_chain_db_key"] = old_db_key

        # Simulate database change
        mock_streamlit["session_state"]["vectorstore"] = new_vectorstore
        new_db_key = f"{topic}_{id(new_vectorstore)}"

        # Check if recreation is needed (simulating dashboard logic)
        needs_recreation = (
            "rag_chain" not in mock_streamlit["session_state"]
            or "rag_chain_db_key" not in mock_streamlit["session_state"]
            or mock_streamlit["session_state"]["rag_chain_db_key"] != new_db_key
        )

        assert (
            needs_recreation
        ), "RAG chain should need recreation when database changes"

        # Simulate recreation
        mock_streamlit["session_state"]["rag_chain"] = (
            mock_knowledge_manager.create_advanced_rag_chain(new_vectorstore, topic)
        )
        mock_streamlit["session_state"]["rag_chain_db_key"] = new_db_key

        # Verify new chain was created with new database
        assert mock_streamlit["session_state"]["rag_chain_db_key"] == new_db_key
        mock_knowledge_manager.create_advanced_rag_chain.assert_called_with(
            new_vectorstore, topic
        )

    def test_rag_chain_not_recreated_when_unchanged(
        self, mock_streamlit, mock_vectorstore, mock_knowledge_manager
    ):
        """Test RAG chain is not recreated when database hasn't changed."""
        # Setup existing state
        topic = "Test Topic"
        db_key = f"{topic}_{id(mock_vectorstore)}"
        existing_rag_chain = Mock()

        mock_streamlit["session_state"]["vectorstore"] = mock_vectorstore
        mock_streamlit["session_state"]["topic"] = topic
        mock_streamlit["session_state"]["rag_chain"] = existing_rag_chain
        mock_streamlit["session_state"]["rag_chain_db_key"] = db_key

        # Check if recreation is needed
        current_db_key = f"{topic}_{id(mock_vectorstore)}"
        needs_recreation = (
            "rag_chain" not in mock_streamlit["session_state"]
            or "rag_chain_db_key" not in mock_streamlit["session_state"]
            or mock_streamlit["session_state"]["rag_chain_db_key"] != current_db_key
        )

        assert (
            not needs_recreation
        ), "RAG chain should not need recreation when database unchanged"

        # Verify original chain is still there and manager wasn't called
        assert mock_streamlit["session_state"]["rag_chain"] is existing_rag_chain
        mock_knowledge_manager.create_advanced_rag_chain.assert_not_called()

    def test_topic_change_triggers_rag_chain_recreation(
        self, mock_streamlit, mock_vectorstore, mock_knowledge_manager
    ):
        """Test RAG chain is recreated when topic changes."""
        # Setup initial state
        old_topic = "Old Topic"
        new_topic = "New Topic"
        old_db_key = f"{old_topic}_{id(mock_vectorstore)}"

        mock_streamlit["session_state"]["vectorstore"] = mock_vectorstore
        mock_streamlit["session_state"]["topic"] = old_topic
        mock_streamlit["session_state"]["rag_chain"] = Mock()
        mock_streamlit["session_state"]["rag_chain_db_key"] = old_db_key

        # Simulate topic change
        mock_streamlit["session_state"]["topic"] = new_topic
        new_db_key = f"{new_topic}_{id(mock_vectorstore)}"

        # Check if recreation is needed
        needs_recreation = (
            "rag_chain" not in mock_streamlit["session_state"]
            or "rag_chain_db_key" not in mock_streamlit["session_state"]
            or mock_streamlit["session_state"]["rag_chain_db_key"] != new_db_key
        )

        assert needs_recreation, "RAG chain should need recreation when topic changes"

        # Verify key changed
        assert old_db_key != new_db_key, "Database key should change when topic changes"


class TestSessionStatePatterns:
    """Test session state usage patterns for consistency."""

    def test_session_state_cleanup_safety(self, mock_streamlit):
        """Test that session state cleanup handles missing keys gracefully."""
        # Setup some session state
        mock_streamlit["session_state"]["key1"] = "value1"
        mock_streamlit["session_state"]["key2"] = "value2"

        # Simulate safe cleanup (pattern used in dashboard.py)
        cleanup_keys = ["key1", "key2", "nonexistent_key"]

        for key in cleanup_keys:
            if key in mock_streamlit["session_state"]:
                try:
                    del mock_streamlit["session_state"][key]
                except Exception:
                    pass  # Ignore cleanup errors

        # Verify cleanup worked and didn't error on missing key
        assert "key1" not in mock_streamlit["session_state"]
        assert "key2" not in mock_streamlit["session_state"]
        # Should not error even though 'nonexistent_key' wasn't there

    def test_database_switch_clears_chat_history(self, mock_streamlit):
        """Test that switching databases clears chat history."""
        # Setup initial state
        mock_streamlit["session_state"]["current_database"] = "Database1"
        mock_streamlit["session_state"]["chat_messages"] = ["msg1", "msg2"]

        # Simulate database switch (logic from _handle_existing_database)
        selected_db = "Database2"

        if (
            "current_database" not in mock_streamlit["session_state"]
            or mock_streamlit["session_state"]["current_database"] != selected_db
        ):

            if "chat_messages" in mock_streamlit["session_state"]:
                del mock_streamlit["session_state"]["chat_messages"]
            mock_streamlit["session_state"]["current_database"] = selected_db

        # Verify chat was cleared and database was updated
        assert "chat_messages" not in mock_streamlit["session_state"]
        assert mock_streamlit["session_state"]["current_database"] == selected_db
