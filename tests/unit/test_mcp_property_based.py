#!/usr/bin/env python3
"""
Property-based tests for MCP Server functions using hypothesis.
Tests invariants, edge cases, and domain properties.
"""

import pytest
import sys
from pathlib import Path
from hypothesis import given, strategies as st, assume
import re

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_integration.youtube_mcp_server import (
    get_video_id_from_url,
    get_playlist_id_from_url,
    get_channel_id_from_url,
    validate_video_input,
    validate_query_input,
    validate_max_results,
    validate_language_input
)

# Property-based testing strategies
video_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd"], min_codepoint=ord('A'), max_codepoint=ord('z')),
    min_size=11,
    max_size=11
).filter(lambda x: x.isalnum() and len(x) == 11)

youtube_url_strategy = st.one_of(
    st.builds(
        "https://www.youtube.com/watch?v={}".format,
        video_id_strategy
    ),
    st.builds(
        "https://youtu.be/{}".format,
        video_id_strategy
    ),
    st.builds(
        "https://m.youtube.com/watch?v={}".format,
        video_id_strategy
    )
)

playlist_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd"], min_codepoint=ord('A'), max_codepoint=ord('z')),
    min_size=10,
    max_size=50
).filter(lambda x: x.startswith('PL') and len(x) > 10)

search_query_strategy = st.text(
    alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd", "Pc", "Pd"], min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=200
).filter(lambda x: x.strip() and not any(char in x for char in ['<', '>', '&', '"', "'", '\\', '/', '|']))

language_code_strategy = st.one_of(
    st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=2, max_size=2),
    st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=3, max_size=3)
)


class TestVideoIdExtractionProperties:
    """Property-based tests for video ID extraction invariants."""
    
    @given(video_id_strategy)
    def test_video_id_extraction_idempotence(self, video_id):
        """Test that extracting video ID from a video ID returns the same ID."""
        # Property: get_video_id_from_url(video_id) == video_id
        result = get_video_id_from_url(video_id)
        assert result == video_id, f"Expected {video_id}, got {result}"
    
    @given(youtube_url_strategy)
    def test_youtube_url_extraction_consistency(self, youtube_url):
        """Test that extraction from YouTube URLs is consistent."""
        # Property: extracted ID should be exactly 11 characters
        result = get_video_id_from_url(youtube_url)
        assume(result is not None)  # Skip if extraction fails
        assert len(result) == 11, f"Video ID should be 11 characters, got {len(result)}"
        assert result.isalnum(), f"Video ID should be alphanumeric, got {result}"
    
    @given(st.text(min_size=1, max_size=100))
    def test_video_id_extraction_defensive(self, arbitrary_text):
        """Test that video ID extraction doesn't crash on arbitrary input."""
        # Property: function should never crash, always return str or None
        result = get_video_id_from_url(arbitrary_text)
        assert result is None or isinstance(result, str), f"Expected str or None, got {type(result)}"
    
    @given(video_id_strategy)
    def test_video_id_round_trip_property(self, video_id):
        """Test round-trip property: URL -> extract -> should equal original ID."""
        # Build a URL and extract from it
        test_url = f"https://www.youtube.com/watch?v={video_id}"
        extracted_id = get_video_id_from_url(test_url)
        
        # Property: round-trip should preserve the original ID
        assert extracted_id == video_id, f"Round-trip failed: {video_id} -> {extracted_id}"


class TestValidationProperties:
    """Property-based tests for input validation invariants."""
    
    @given(st.text(min_size=1, max_size=10))
    def test_video_input_validation_short_strings(self, short_string):
        """Test video input validation with short strings."""
        # Property: strings shorter than 11 characters should fail validation
        assume(len(short_string) < 11)
        
        try:
            validate_video_input(short_string)
            # If it doesn't raise an exception, it should be a valid format
            assert False, f"Expected validation to fail for short string: {short_string}"
        except ValueError:
            # Expected behavior for short strings
            pass
    
    @given(video_id_strategy)
    def test_video_input_validation_valid_ids(self, video_id):
        """Test that valid video IDs pass validation."""
        # Property: valid video IDs should not raise exceptions
        try:
            validate_video_input(video_id)
            # Should not raise exception
        except ValueError as e:
            assert False, f"Valid video ID failed validation: {video_id}, error: {e}"
    
    @given(search_query_strategy)
    def test_query_validation_reasonable_inputs(self, query):
        """Test query validation with reasonable search queries."""
        # Property: reasonable queries should pass validation
        assume(len(query.strip()) >= 1 and len(query) <= 200)
        
        try:
            validate_query_input(query)
            # Should not raise exception for reasonable queries
        except ValueError as e:
            # If it fails, it should be for a specific reason
            assert any(char in query for char in ['<', '>', '&', '"', "'", '\\', '/', '|']), \
                f"Query validation failed unexpectedly: {query}, error: {e}"
    
    @given(st.integers(min_value=1, max_value=50))
    def test_max_results_validation_valid_range(self, max_results):
        """Test max results validation within valid range."""
        # Property: integers 1-50 should pass validation
        try:
            validate_max_results(max_results)
            # Should not raise exception
        except ValueError as e:
            assert False, f"Valid max_results failed validation: {max_results}, error: {e}"
    
    @given(st.integers().filter(lambda x: x < 1 or x > 50))
    def test_max_results_validation_invalid_range(self, invalid_max_results):
        """Test max results validation outside valid range."""
        # Property: integers outside 1-50 should fail validation
        try:
            validate_max_results(invalid_max_results)
            assert False, f"Expected validation to fail for invalid max_results: {invalid_max_results}"
        except ValueError:
            # Expected behavior for invalid range
            pass
    
    @given(language_code_strategy)
    def test_language_validation_reasonable_codes(self, language_code):
        """Test language validation with reasonable language codes."""
        # Property: 2-3 character lowercase codes should pass validation
        assume(language_code.islower() and 2 <= len(language_code) <= 3)
        
        try:
            validate_language_input(language_code)
            # Should not raise exception for reasonable codes
        except ValueError as e:
            assert False, f"Reasonable language code failed validation: {language_code}, error: {e}"


class TestPlaylistChannelExtractionProperties:
    """Property-based tests for playlist and channel extraction invariants."""
    
    @given(playlist_id_strategy)
    def test_playlist_id_extraction_idempotence(self, playlist_id):
        """Test that extracting playlist ID from a playlist ID returns the same ID."""
        # Property: get_playlist_id_from_url(playlist_id) == playlist_id
        result = get_playlist_id_from_url(playlist_id)
        assert result == playlist_id, f"Expected {playlist_id}, got {result}"
    
    @given(st.text(min_size=1, max_size=100))
    def test_playlist_extraction_defensive(self, arbitrary_text):
        """Test that playlist ID extraction doesn't crash on arbitrary input."""
        # Property: function should never crash, always return str or None
        result = get_playlist_id_from_url(arbitrary_text)
        assert result is None or isinstance(result, str), f"Expected str or None, got {type(result)}"
    
    @given(st.text(min_size=1, max_size=100))
    def test_channel_extraction_defensive(self, arbitrary_text):
        """Test that channel ID extraction doesn't crash on arbitrary input."""
        # Property: function should never crash, always return str or None
        result = get_channel_id_from_url(arbitrary_text)
        assert result is None or isinstance(result, str), f"Expected str or None, got {type(result)}"
    
    @given(st.text(alphabet=st.characters(min_codepoint=ord('A'), max_codepoint=ord('z')), min_size=15, max_size=30))
    def test_channel_id_extraction_uc_prefix(self, channel_suffix):
        """Test channel ID extraction with UC prefix."""
        # Property: UC-prefixed IDs should be extracted correctly
        assume(channel_suffix.isalnum())
        channel_id = f"UC{channel_suffix}"
        
        result = get_channel_id_from_url(channel_id)
        assert result == channel_id, f"Expected {channel_id}, got {result}"


class TestCombinedExtractionProperties:
    """Property-based tests for combined extraction behaviors."""
    
    @given(st.one_of(
        video_id_strategy,
        youtube_url_strategy,
        playlist_id_strategy,
        st.text(min_size=1, max_size=100)
    ))
    def test_extraction_functions_mutual_exclusivity(self, input_string):
        """Test that extraction functions are mutually exclusive."""
        # Property: at most one extraction function should succeed for any input
        video_result = get_video_id_from_url(input_string)
        playlist_result = get_playlist_id_from_url(input_string)
        channel_result = get_channel_id_from_url(input_string)
        
        # Count non-None results
        non_none_results = sum(1 for result in [video_result, playlist_result, channel_result] if result is not None)
        
        # Property: should have at most one successful extraction
        assert non_none_results <= 1, f"Multiple extractions succeeded for {input_string}: video={video_result}, playlist={playlist_result}, channel={channel_result}"
    
    @given(st.text(min_size=1, max_size=1000))
    def test_all_extraction_functions_robust(self, arbitrary_input):
        """Test that all extraction functions are robust to arbitrary input."""
        # Property: none of the extraction functions should crash
        try:
            video_result = get_video_id_from_url(arbitrary_input)
            playlist_result = get_playlist_id_from_url(arbitrary_input)
            channel_result = get_channel_id_from_url(arbitrary_input)
            
            # All results should be string or None
            assert video_result is None or isinstance(video_result, str)
            assert playlist_result is None or isinstance(playlist_result, str)
            assert channel_result is None or isinstance(channel_result, str)
            
        except Exception as e:
            assert False, f"Extraction function crashed on input {arbitrary_input}: {e}"


class TestValidationCommutativity:
    """Property-based tests for validation function commutativity."""
    
    @given(st.text(min_size=1, max_size=100))
    def test_video_validation_deterministic(self, video_input):
        """Test that video validation is deterministic."""
        # Property: validation should always give the same result for the same input
        try:
            result1 = validate_video_input(video_input)
            result2 = validate_video_input(video_input)
            # If no exception, both should succeed
            assert result1 == result2  # Should both be None (success)
        except ValueError as e1:
            # If exception, should consistently throw the same exception
            try:
                validate_video_input(video_input)
                assert False, f"Validation was inconsistent for {video_input}"
            except ValueError as e2:
                # Should get the same error message
                assert str(e1) == str(e2), f"Error messages differ for {video_input}: {e1} vs {e2}"
    
    @given(st.text(min_size=1, max_size=500))
    def test_query_validation_deterministic(self, query_input):
        """Test that query validation is deterministic."""
        # Property: validation should always give the same result for the same input
        try:
            result1 = validate_query_input(query_input)
            result2 = validate_query_input(query_input)
            # If no exception, both should succeed
            assert result1 == result2  # Should both be None (success)
        except ValueError as e1:
            # If exception, should consistently throw the same exception
            try:
                validate_query_input(query_input)
                assert False, f"Query validation was inconsistent for {query_input}"
            except ValueError as e2:
                # Should get the same error message
                assert str(e1) == str(e2), f"Error messages differ for {query_input}: {e1} vs {e2}"


if __name__ == "__main__":
    # Run property-based tests with more examples for thorough testing
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"]) 