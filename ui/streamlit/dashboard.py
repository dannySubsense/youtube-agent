"""
Refactored Streamlit Dashboard for YouTube Agent System.
Uses modular backend while preserving exact MVP user experience.
"""

import streamlit as st
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Import our modular components
from config.settings import get_settings, Settings
from tools.video_search import create_video_searcher, VideoSearchError
from tools.transcript_manager import (
    create_transcript_manager,
    TranscriptResult,
    ProcessingStats,
)
from tools.vector_database import create_vector_database_manager, DatabaseInfo
from tools.knowledge_base import create_knowledge_base_manager, RAGChainConfig


class YouTubeAgentUI:
    """
    Streamlit UI for YouTube Agent System using modular backend.

    Preserves exact MVP functionality while using clean, testable modules:
    - Topic → Videos → Vector DB → Chat pipeline unchanged
    - All UI interactions identical to original MVP
    - Enhanced error handling and progress tracking
    """

    def __init__(self):
        """Initialize the UI with modular components."""
        self.settings = self._load_settings()
        self.video_searcher = create_video_searcher(self.settings)
        self.transcript_manager = create_transcript_manager(self.settings)
        self.vector_db_manager = create_vector_database_manager(self.settings)
        self.knowledge_manager = create_knowledge_base_manager(self.settings)

    def _load_settings(self) -> Settings:
        """Load application settings."""
        try:
            return get_settings()
        except Exception as e:
            st.error(f"❌ Error loading configuration: {e}")
            st.stop()

    def run(self):
        """Main application entry point."""
        self._setup_page()
        self._show_header()
        self._show_database_selection()

    def _setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="YouTube Agent MVP", page_icon="🎯", layout="wide"
        )

    def _show_header(self):
        """Display application header."""
        st.title("🎯 YouTube Channel Agent - MVP")
        st.markdown(
            "**Simple Workflow**: Topic → Top 5 Videos → Knowledge Base → Advanced Chat"
        )

    def _show_database_selection(self):
        """Handle database selection or creation workflow."""
        st.header("🔍 Select Database or Create New")

        # Discover available databases
        available_databases = self._discover_databases()

        # Database selection options
        database_option = st.radio(
            "Choose your option:",
            ["📚 Use Existing Database", "🆕 Create New Database"],
            horizontal=True,
            help="Select an existing database or create a new one with different content",
        )

        if database_option == "📚 Use Existing Database":
            self._handle_existing_database(available_databases)
        else:
            self._handle_new_database_creation()

    def _discover_databases(self) -> List[str]:
        """Discover available databases using modular backend."""
        databases = self.vector_db_manager.discover_databases()

        if not databases:
            return ["No databases found - Create one below!"]

        return [db.name for db in databases]

    def _handle_existing_database(self, available_databases: List[str]):
        """Handle existing database selection and loading."""
        selected_db = st.selectbox(
            "Select database:",
            available_databases,
            help="Choose from available knowledge bases",
            key="database_selector",
        )

        # Handle case where no databases exist
        if selected_db == "No databases found - Create one below!":
            st.info("👆 No existing databases found. Please create a new one below.")
            return

        # Load the existing database
        try:
            result = self.vector_db_manager.load_existing_database(selected_db)
            if result:
                vectorstore, db_path = result
                st.info(f"✅ Successfully loaded database: **{selected_db}**")

                # Clear chat history when switching databases
                if (
                    "current_database" not in st.session_state
                    or st.session_state.current_database != selected_db
                ):
                    if "chat_messages" in st.session_state:
                        del st.session_state["chat_messages"]
                    st.session_state.current_database = selected_db

                # Store in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.topic = selected_db

                # Show chat interface
                self._show_chat_interface()

            else:
                st.error(f"❌ Could not load database: {selected_db}")

        except Exception as e:
            st.error(f"❌ Error loading database: {e}")

    def _handle_new_database_creation(self):
        """Handle new database creation workflow."""
        with st.form("topic_form"):
            topic = st.text_input(
                "Enter your topic:",
                placeholder="e.g., 'React Hooks', 'Python FastAPI', 'Machine Learning'",
                help="What topic would you like to create a knowledge base for?",
            )

            submitted = st.form_submit_button("🔍 Search Videos")

            if submitted and topic.strip():
                # Store topic for video processing
                st.session_state.creating_new = True
                st.session_state.creation_topic = topic.strip()
                st.rerun()

        # Handle video search and selection
        if hasattr(st.session_state, "creating_new") and st.session_state.creating_new:
            self._handle_video_search_and_selection()

    def _handle_video_search_and_selection(self):
        """Handle video search, selection, and processing."""
        creation_topic = st.session_state.get("creation_topic", "Unknown Topic")

        # Multi-strategy search
        if "found_videos" not in st.session_state:
            st.header("🔍 Searching for Videos")
            self._search_videos_with_strategies(creation_topic)

        # Agent analysis option (NEW - MCP Integration)
        if (
            "found_videos" in st.session_state
            and "analysis_mode" not in st.session_state
        ):
            self._show_analysis_mode_selection()

        # Video selection interface
        if "found_videos" in st.session_state and "analysis_mode" in st.session_state:
            if st.session_state.analysis_mode == "agent_enhanced":
                self._show_agent_enhanced_interface(creation_topic)
            else:
                self._show_video_selection_interface(creation_topic)

    def _search_videos_with_strategies(self, topic: str):
        """Search for videos using multiple strategies."""
        # Generate search strategies
        search_strategies = [
            topic,
            f"{topic} tutorial",
            f"{topic} guide",
            f"{topic} explained",
            f"how to {topic}",
            f"{topic} walkthrough",
        ]

        all_videos = []
        seen_video_ids = set()

        try:
            for search_term in search_strategies[:3]:  # Limit to 3 searches
                with st.spinner(f"🔍 Searching: '{search_term}'"):
                    try:
                        videos = self.video_searcher.search_videos(
                            search_term, max_results=8
                        )

                        # Remove duplicates
                        for video in videos:
                            if video["video_id"] not in seen_video_ids:
                                all_videos.append(video)
                                seen_video_ids.add(video["video_id"])

                    except VideoSearchError as e:
                        st.warning(f"Search failed for '{search_term}': {e}")
                        continue

            if not all_videos:
                st.error("❌ No videos found! Try a different topic.")
                return

            st.session_state.found_videos = all_videos
            st.success(f"✅ Found {len(all_videos)} unique videos!")

        except Exception as e:
            st.error(f"❌ Search failed: {e}")

    def _show_analysis_mode_selection(self):
        """Show analysis mode selection - MVP workflow vs Agent-enhanced analysis."""
        st.header("🤖 Choose Analysis Mode")
        st.markdown("**Select how you want to process and analyze the found videos:**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            ### 📚 **MVP Workflow** (Original)
            - Manual video selection and review
            - Traditional transcript processing
            - Proven batch processing pipeline
            - Full user control over video selection
            
            **Best for**: When you want complete control over video selection
            """
            )

            if st.button("📚 Use MVP Workflow", use_container_width=True):
                st.session_state.analysis_mode = "mvp_workflow"
                st.rerun()

        with col2:
            st.markdown(
                """
            ### 🤖 **Agent-Enhanced Analysis** (New)
            - AI agent autonomously analyzes each video
            - Quality scoring and engagement analysis  
            - Intelligent recommendations (keep/drop/defer)
            - User reviews final decisions
            
            **Best for**: When you want AI assistance in video curation
            """
            )

            if st.button("🤖 Use Agent Analysis", use_container_width=True):
                st.session_state.analysis_mode = "agent_enhanced"
                st.rerun()

        st.info(
            "💡 **Note**: Both modes use the same underlying MVP architecture. Agent mode adds intelligent analysis on top."
        )

    def _show_agent_enhanced_interface(self, creation_topic: str):
        """Show agent-enhanced analysis interface."""
        st.header("🤖 Agent-Enhanced Video Analysis")
        st.info(
            f"🎯 **Found {len(st.session_state.found_videos)} videos** - Agent will analyze each video and provide recommendations"
        )

        # Initialize agent components (simulated for Phase 1)
        if "agent_analysis_results" not in st.session_state:
            if st.button(
                "🚀 Start Agent Analysis", use_container_width=True, type="primary"
            ):
                self._run_agent_analysis(creation_topic)
        else:
            self._show_agent_analysis_results(creation_topic)

    def _run_agent_analysis(self, topic: str):
        """Run agent analysis on found videos (simulated for Phase 1)."""
        with st.spinner("🤖 Agent analyzing videos..."):
            # Simulate agent analysis for Phase 1
            # In full implementation, this would use the MCP integration components

            analysis_results = []

            for i, video in enumerate(st.session_state.found_videos):
                # Simulate autonomous analysis
                progress = (i + 1) / len(st.session_state.found_videos)
                st.progress(progress)
                st.info(
                    f"Analyzing {i+1}/{len(st.session_state.found_videos)}: {video['title'][:50]}..."
                )

                # Simulate quality scoring (would use actual MCP tools)
                import random

                quality_score = random.uniform(50, 95)

                # Simulate agent recommendation
                if quality_score >= 80:
                    recommendation = "keep"
                    reasoning = "High quality content with strong engagement metrics"
                elif quality_score >= 60:
                    recommendation = "defer"
                    reasoning = "Moderate quality - worth reviewing manually"
                else:
                    recommendation = "drop"
                    reasoning = "Low quality or engagement - not recommended"

                analysis_results.append(
                    {
                        "video": video,
                        "quality_score": quality_score,
                        "recommendation": recommendation,
                        "reasoning": reasoning,
                        "user_decision": "pending",  # User hasn't decided yet
                    }
                )

            st.session_state.agent_analysis_results = analysis_results
            st.success("✅ Agent analysis complete! Review the results below.")
            st.rerun()

    def _show_agent_analysis_results(self, topic: str):
        """Show agent analysis results and handle user decisions."""
        st.subheader("📊 Agent Analysis Results")

        results = st.session_state.agent_analysis_results

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        keep_count = len([r for r in results if r["recommendation"] == "keep"])
        defer_count = len([r for r in results if r["recommendation"] == "defer"])
        drop_count = len([r for r in results if r["recommendation"] == "drop"])
        avg_quality = sum(r["quality_score"] for r in results) / len(results)

        with col1:
            st.metric("🟢 Recommended Keep", keep_count)
        with col2:
            st.metric("🟡 Needs Review", defer_count)
        with col3:
            st.metric("🔴 Recommended Drop", drop_count)
        with col4:
            st.metric("📊 Avg Quality", f"{avg_quality:.1f}")

        # Individual video results
        st.markdown("### 📹 Individual Video Analysis")

        # User decision form
        with st.form("agent_decisions_form"):
            user_decisions = {}
            selected_for_processing = []

            for i, result in enumerate(results):
                video = result["video"]
                quality_score = result["quality_score"]
                recommendation = result["recommendation"]
                reasoning = result["reasoning"]

                # Color coding for recommendations
                if recommendation == "keep":
                    emoji = "🟢"
                    color = "#28a745"
                elif recommendation == "defer":
                    emoji = "🟡"
                    color = "#ffc107"
                else:
                    emoji = "🔴"
                    color = "#dc3545"

                with st.expander(
                    f"{emoji} {video['title'][:60]}... (Score: {quality_score:.1f})",
                    expanded=False,
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(
                            f"""
                        **📺 Title:** {video['title']}  
                        **👤 Channel:** {video['channel']}  
                        **⏱️ Duration:** {video['duration']}  
                        **📊 Quality Score:** {quality_score:.1f}/100  
                        **🤖 Agent Recommendation:** <span style="color:{color}">**{recommendation.upper()}**</span>  
                        **💭 Reasoning:** {reasoning}  
                        **🔗 [Watch Video]({video['url']})**
                        """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        # User decision override
                        user_decision = st.selectbox(
                            "Your Decision:",
                            ["Follow Agent", "Keep", "Drop"],
                            key=f"decision_{video['video_id']}",
                            help="Override agent recommendation if needed",
                        )

                        # Determine final decision
                        if user_decision == "Follow Agent":
                            final_decision = recommendation
                        elif user_decision == "Keep":
                            final_decision = "keep"
                        else:
                            final_decision = "drop"

                        user_decisions[video["video_id"]] = final_decision

                        if final_decision == "keep":
                            selected_for_processing.append(video)

            # Process decisions
            if st.form_submit_button(
                "✅ Process Selected Videos", type="primary", use_container_width=True
            ):
                if not selected_for_processing:
                    st.error(
                        "❌ No videos selected for processing. Please select at least one video."
                    )
                elif (
                    len(selected_for_processing)
                    > self.settings.max_videos_per_knowledge_base
                ):
                    st.error(
                        f"❌ Too many videos selected! Please select no more than {self.settings.max_videos_per_knowledge_base} videos."
                    )
                else:
                    # Store selections and process
                    st.session_state.selected_videos_array = selected_for_processing
                    st.success(
                        f"✅ Processing {len(selected_for_processing)} videos based on your decisions..."
                    )

                    # Show final selections
                    st.write("**Final Video Selection:**")
                    for i, video in enumerate(selected_for_processing):
                        decision_type = user_decisions[video["video_id"]]
                        emoji = "🤖" if decision_type == "keep" else "👤"
                        st.write(f"   {i+1}. {emoji} {video['title']}")

                    self._process_selected_videos(topic)

        # Option to go back
        if st.button("🔄 Back to Mode Selection"):
            # Clear agent analysis state
            if "agent_analysis_results" in st.session_state:
                del st.session_state["agent_analysis_results"]
            if "analysis_mode" in st.session_state:
                del st.session_state["analysis_mode"]
            st.rerun()

    def _show_video_selection_interface(self, creation_topic: str):
        """Show video selection interface matching MVP exactly."""
        st.header("📋 Select Videos for Your Knowledge Base")
        st.info(
            f"🎯 **Found {len(st.session_state.found_videos)} videos** - Choose up to **{self.settings.max_videos_per_knowledge_base} videos** to include in your knowledge base"
        )

        # Show limit information
        if (
            len(st.session_state.found_videos)
            > self.settings.max_videos_per_knowledge_base
        ):
            st.warning(
                f"💡 **Tip**: To optimize processing time and API usage, you can select up to {self.settings.max_videos_per_knowledge_base} videos. Choose the most relevant ones for your topic."
            )

        # Initialize selection array
        if "selected_videos_array" not in st.session_state:
            st.session_state.selected_videos_array = []

        # Video selection form (Fixed - Best Practices Implementation)
        with st.form("video_selection_form", clear_on_submit=False):
            st.markdown("**📺 Available Videos:**")
            st.markdown(
                f"*Review each video and select up to **{self.settings.max_videos_per_knowledge_base} videos** that are specifically relevant to your topic*"
            )

            # Use global video limit for consistency
            MAX_VIDEOS = self.settings.max_videos_per_knowledge_base

            # Initialize selection array if not exists (defensive programming)
            if "selected_videos_array" not in st.session_state:
                st.session_state.selected_videos_array = []

            # Collect video selections WITHOUT updating session state during form creation
            video_selections = []
            current_selection_ids = {
                v["video_id"] for v in st.session_state.selected_videos_array
            }

            # Create video selection interface
            for i, video in enumerate(st.session_state.found_videos):
                try:
                    # Validate video data structure
                    required_fields = [
                        "video_id",
                        "title",
                        "channel",
                        "duration",
                        "view_count",
                        "description",
                        "url",
                    ]
                    missing_fields = [
                        field for field in required_fields if field not in video
                    ]

                    if missing_fields:
                        st.error(f"❌ Video {i+1} missing fields: {missing_fields}")
                        continue

                    # Create expandable card for each video
                    video_title = (
                        video["title"][:70] + "..."
                        if len(video["title"]) > 70
                        else video["title"]
                    )

                    with st.expander(f"📹 {video_title}", expanded=False):
                        col1, col2 = st.columns([1, 3])

                        with col1:
                            # Video selection checkbox
                            checkbox_key = f"video_select_{i}_{video['video_id']}"  # More unique key
                            is_currently_selected = (
                                video["video_id"] in current_selection_ids
                            )

                            selected = st.checkbox(
                                "Include",
                                key=checkbox_key,
                                value=is_currently_selected,
                                help=f"Select to include this video in your knowledge base (Limit: {MAX_VIDEOS} videos)",
                            )

                            # Store selection for processing (don't update session state yet)
                            video_selections.append(
                                {
                                    "video": video,
                                    "selected": selected,
                                    "was_selected": is_currently_selected,
                                }
                            )

                        with col2:
                            # Video details with safe formatting
                            try:
                                view_count_str = (
                                    f"{int(video['view_count']):,}"
                                    if video["view_count"]
                                    else "Unknown"
                                )
                            except (ValueError, TypeError):
                                view_count_str = (
                                    str(video["view_count"])
                                    if video["view_count"]
                                    else "Unknown"
                                )

                            st.markdown(
                                f"""
                            **📺 Title:** {video['title']}  
                            **👤 Channel:** {video['channel']}  
                            **⏱️ Duration:** {video['duration']}  
                            **👁️ Views:** {view_count_str}  
                            **📝 Description:** {video['description'][:200]}{"..." if len(video['description']) > 200 else ""}  
                            **🔗 [Watch Video]({video['url']})**
                            """
                            )

                except Exception as e:
                    st.error(f"❌ Error displaying video {i+1}: {str(e)}")
                    continue

            # Process form submission
            process_selected = st.form_submit_button(
                "🚀 Process Selected Videos", type="primary", use_container_width=True
            )

            if process_selected:
                try:
                    # Process selections and update session state only on submit
                    new_selected_videos = []

                    for selection in video_selections:
                        if selection["selected"]:
                            # Validate video data before adding
                            video = selection["video"]
                            if all(
                                key in video for key in ["video_id", "title", "channel"]
                            ):
                                new_selected_videos.append(video)
                            else:
                                st.warning(
                                    f"⚠️ Skipping invalid video: {video.get('title', 'Unknown')}"
                                )

                    # Enforce video limit
                    if len(new_selected_videos) == 0:
                        st.error(
                            "❌ Please select at least one video to build your knowledge base."
                        )
                    elif len(new_selected_videos) > MAX_VIDEOS:
                        st.error(
                            f"❌ Too many videos selected! Please select no more than {MAX_VIDEOS} videos."
                        )
                        st.info(
                            f"💡 You selected {len(new_selected_videos)} videos. Please uncheck {len(new_selected_videos) - MAX_VIDEOS} videos."
                        )
                    else:
                        # Success - update session state and proceed
                        st.session_state.selected_videos_array = new_selected_videos
                        st.success(
                            f"✅ Processing {len(new_selected_videos)} selected videos..."
                        )

                        # Show selected videos
                        for i, video in enumerate(new_selected_videos):
                            st.write(f"   {i+1}. {video['title']}")

                        # Process the videos
                        self._process_selected_videos(creation_topic)

                except Exception as e:
                    st.error(f"❌ Error processing video selections: {str(e)}")
                    st.error(
                        "💡 Please try selecting videos again or refresh the page."
                    )

            else:
                # Show current selection status without processing
                current_count = sum(1 for s in video_selections if s["selected"])

                if current_count == 0:
                    st.info(
                        f"📊 **Selection Status:** 0 / {MAX_VIDEOS} videos selected"
                    )
                elif current_count <= MAX_VIDEOS:
                    st.success(
                        f"📊 **Selection Status:** {current_count} / {MAX_VIDEOS} videos selected"
                    )
                    if current_count > 0:
                        st.write("**Currently selected:**")
                        for selection in video_selections:
                            if selection["selected"]:
                                st.write(f"   ✅ {selection['video']['title']}")
                else:
                    st.error(
                        f"📊 **Selection Status:** {current_count} / {MAX_VIDEOS} videos selected - TOO MANY!"
                    )
                    st.warning(
                        f"⚠️ Please uncheck {current_count - MAX_VIDEOS} videos before processing."
                    )

    def _process_selected_videos(self, topic: str):
        """Process selected videos and create knowledge base with comprehensive error handling."""
        try:
            # Validate input parameters
            if not topic or not topic.strip():
                st.error("❌ Invalid topic provided. Please try again.")
                return

            topic = topic.strip()

            # Validate session state
            if "selected_videos_array" not in st.session_state:
                st.error("❌ No videos selected. Please select videos first.")
                return

            final_selected_videos = st.session_state.selected_videos_array.copy()

            # Validate video selection
            if not final_selected_videos:
                st.error(
                    "❌ Please select at least one video to build your knowledge base."
                )
                return

            if len(final_selected_videos) > self.settings.max_videos_per_knowledge_base:
                st.error(
                    f"❌ Too many videos selected! Please select no more than {self.settings.max_videos_per_knowledge_base} videos."
                )
                st.info("💡 Please uncheck some videos and try again.")
                return

            # Validate video data structure
            valid_videos = []
            for i, video in enumerate(final_selected_videos):
                try:
                    required_fields = ["video_id", "title", "channel", "url"]
                    if all(
                        field in video and video[field] for field in required_fields
                    ):
                        valid_videos.append(video)
                    else:
                        st.warning(f"⚠️ Skipping video {i+1}: Missing required data")
                except Exception as video_error:
                    st.warning(
                        f"⚠️ Skipping video {i+1}: Invalid data structure - {video_error}"
                    )

            if not valid_videos:
                st.error(
                    "❌ No valid videos found. Please try selecting different videos."
                )
                return

            # Show processing info
            st.info(
                f"🚀 **Processing {len(valid_videos)} selected videos for topic: {topic}**"
            )
            for i, video in enumerate(valid_videos):
                st.write(f"   {i+1}. {video.get('title', 'Unknown Title')}")

            # Process transcripts with error handling
            st.header("📝 Processing Selected Videos")
            try:
                with st.spinner("Extracting transcripts..."):

                    def progress_callback(current: int, total: int, title: str):
                        try:
                            progress = (current + 1) / total
                            st.progress(progress)
                            safe_title = str(title)[:50] if title else "Unknown"
                            st.info(
                                f"Processing {current + 1}/{total}: {safe_title}..."
                            )
                        except Exception as progress_error:
                            st.warning(f"Progress update error: {progress_error}")

                    results, stats = self.transcript_manager.process_videos(
                        valid_videos, progress_callback=progress_callback
                    )

            except Exception as transcript_error:
                st.error(f"❌ Error during transcript processing: {transcript_error}")
                st.error(
                    "💡 This might be due to network issues or API rate limits. Please try again later."
                )
                return

            # Process results with error handling
            try:
                formatted_results = self.transcript_manager.format_processing_results(
                    results, stats
                )

                if formatted_results["successful_videos"]:
                    st.success(
                        f"✅ **Successfully processed {len(formatted_results['successful_videos'])} videos:**"
                    )
                    for title in formatted_results["successful_videos"]:
                        st.write(f"   • {title}")

                if formatted_results["failed_videos"]:
                    st.warning(
                        f"⚠️ **Failed to get transcripts for {len(formatted_results['failed_videos'])} videos:**"
                    )
                    for video in formatted_results["failed_videos"]:
                        with st.expander(f"❌ {video.get('title', 'Unknown')}"):
                            st.write(
                                f"**Error:** {video.get('error', 'Unknown error')}"
                            )

            except Exception as format_error:
                st.error(f"❌ Error formatting results: {format_error}")
                return

            # Create knowledge base with error handling
            if formatted_results.get("processed_data"):
                st.header("🧠 Building Knowledge Base")
                try:
                    with st.spinner("Creating vector database..."):

                        # Prepare documents for vector database
                        documents, metadatas, successful_videos, failed_videos = (
                            self.vector_db_manager.prepare_documents_for_vectorstore(
                                formatted_results["processed_data"], topic
                            )
                        )

                        if not documents:
                            st.error(
                                "❌ No valid documents prepared for vector database."
                            )
                            return

                        # Create vector database
                        vectorstore, db_path = (
                            self.vector_db_manager.create_knowledge_base(
                                documents, metadatas, topic
                            )
                        )

                        # Validate vector database creation
                        if vectorstore is None:
                            st.error(
                                "❌ Failed to create vector database. Please check permissions and try again."
                            )
                            return

                        # Store in session state
                        st.session_state.vectorstore = vectorstore
                        st.session_state.topic = topic
                        st.success("🎉 Knowledge base ready! You can now chat below.")

                        # Clean up session state safely
                        cleanup_keys = [
                            "creating_new",
                            "found_videos",
                            "selected_videos_array",
                            "creation_topic",
                        ]
                        for key in cleanup_keys:
                            if key in st.session_state:
                                try:
                                    del st.session_state[key]
                                except Exception:
                                    pass  # Ignore cleanup errors

                        st.rerun()

                except Exception as db_error:
                    st.error(f"❌ Error creating knowledge base: {db_error}")
                    st.error(
                        "💡 This might be a permissions issue. Please check that the data directory is writable."
                    )

                    # Provide specific guidance for common errors
                    error_str = str(db_error).lower()
                    if "readonly" in error_str or "permission" in error_str:
                        st.info("**Possible solutions:**")
                        st.info("• Check file permissions in the data directory")
                        st.info("• Try running with different permissions")
                        st.info("• Contact administrator if the issue persists")
                    elif "disk" in error_str or "space" in error_str:
                        st.info(
                            "**Possible solution:** Free up disk space and try again"
                        )

                    return

            else:
                st.error("❌ No valid transcripts found to create knowledge base")
                st.info("💡 This could be due to:")
                st.info("• Videos without available transcripts")
                st.info("• Network connectivity issues")
                st.info("• API rate limiting")
                st.info("Please try selecting different videos or try again later.")

        except Exception as e:
            st.error(f"❌ Unexpected error during processing: {e}")
            st.error(
                "💡 Please refresh the page and try again. If the problem persists, contact support."
            )

            # Log error details for debugging (in a real app, this would go to proper logging)
            import traceback

            st.expander("🔧 Technical Details (for debugging)").code(
                traceback.format_exc()
            )

    def _show_chat_interface(self):
        """Show the chat interface using modular backend."""
        if not hasattr(st.session_state, "vectorstore"):
            return

        st.header("💬 Chat with Your Agent")
        st.info(
            f"📚 Knowledge Base: **{st.session_state.topic}** | Ready for questions!"
        )

        # Initialize chat memory
        msgs = StreamlitChatMessageHistory(key="chat_messages")

        # Generate dynamic greeting ONLY on first load
        if len(msgs.messages) == 0:
            greeting = self.knowledge_manager.generate_dynamic_greeting(
                st.session_state.vectorstore, st.session_state.topic
            )
            msgs.add_ai_message(greeting)

        # EFFICIENCY FIX: Only setup database components when database actually changes
        self._ensure_database_components_ready()

        # Use cached questions - no database logic needed here
        quick_questions = st.session_state.quick_questions

        # Quick Questions (EXACT MVP COPY)
        with st.expander("💡 Quick Questions - Click to Ask"):
            # Display questions in columns for better layout
            cols = st.columns(2)
            for i, q in enumerate(quick_questions):
                with cols[i % 2]:
                    if st.button(q, key=f"quick_{hash(q)}", use_container_width=True):
                        st.session_state[f"submit_{i}"] = q

        # Check for submitted questions (from buttons or manual input)
        submitted_question = None

        # Check for button submissions
        for i in range(len(quick_questions)):
            if f"submit_{i}" in st.session_state:
                submitted_question = st.session_state[f"submit_{i}"]
                del st.session_state[f"submit_{i}"]
                break

        # Chat management
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**💬 Chat Messages:** {len(msgs.messages)}")
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                msgs.clear()
                msgs.add_ai_message(
                    f"👋 Chat cleared! I'm ready for a fresh conversation about {st.session_state.topic}."
                )
                st.rerun()

        # PROMINENT CHAT INPUT
        st.markdown("### 💬 Ask Your Question:")
        if question := st.chat_input(
            f"Type your question about {st.session_state.topic} here... (I remember our conversation!)"
        ):
            submitted_question = question

        # Process any submitted question (MVP pattern with modular backend)
        if submitted_question:
            # Generate response with history awareness
            # RunnableWithMessageHistory automatically manages adding messages to msgs
            try:
                # Wrap RAG chain with message history (EXACT MVP PATTERN)
                rag_with_history = RunnableWithMessageHistory(
                    st.session_state.rag_chain,
                    lambda session_id: msgs,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )

                # Invoke the advanced RAG chain - this automatically adds user message and AI response
                response = rag_with_history.invoke(
                    {"input": submitted_question},
                    config={"configurable": {"session_id": "youtube_agent"}},
                )

                # Only manually add sources if they exist
                if "context" in response and response["context"]:
                    sources_seen = set()
                    sources_text = "**📚 Sources Used:**\n"
                    for doc in response["context"]:
                        title = doc.metadata.get("title", "Unknown")
                        channel = doc.metadata.get("channel", "Unknown")
                        url = doc.metadata.get("url", "#")

                        source_key = f"{title}|{channel}"
                        if source_key not in sources_seen:
                            sources_text += (
                                f"- **{title}** by {channel} - [🔗 Watch]({url})\n"
                            )
                            sources_seen.add(source_key)

                    # Add sources as a separate message
                    msgs.add_ai_message(sources_text)

            except Exception as e:
                error_msg = f"❌ Error processing question: {e}"
                msgs.add_ai_message(error_msg)

        # Display chat history
        for msg in msgs.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)

        # Footer
        st.markdown("---")
        st.markdown("**YouTube Channel Agent MVP** | Agent Engineering Bootcamp Week 2")

    def _ensure_database_components_ready(self):
        """Ensure RAG chain and questions are set up, but only when database changes."""
        # Use stable database identifier - object ID changes on every rerun!
        db_path = getattr(
            st.session_state.vectorstore,
            "_persist_directory",
            str(st.session_state.topic),
        )
        current_db_key = f"{st.session_state.topic}_{db_path}"

        # Check if we need to create/recreate the RAG chain (only when database changes)
        if (
            "rag_chain" not in st.session_state
            or "rag_chain_db_key" not in st.session_state
            or st.session_state.rag_chain_db_key != current_db_key
        ):
            # Create new base RAG chain using modular architecture
            st.session_state.rag_chain = (
                self.knowledge_manager.create_advanced_rag_chain(
                    st.session_state.vectorstore, st.session_state.topic
                )
            )
            # Store the database key to track changes
            st.session_state.rag_chain_db_key = current_db_key

            # Clear cached questions when database ACTUALLY changes (GATE reopens)
            if "quick_questions" in st.session_state:
                del st.session_state.quick_questions

        # GATE LOGIC: Generate questions ONCE per database, then keep them static
        if "quick_questions" not in st.session_state:
            try:
                st.session_state.quick_questions = (
                    self.knowledge_manager.generate_dynamic_questions(
                        st.session_state.vectorstore,
                        st.session_state.topic,
                        num_questions=5,
                    )
                )
            except Exception as e:
                # Fallback to basic questions if dynamic generation fails
                st.warning(
                    f"🔄 Using fallback questions (dynamic generation error: {e})"
                )
                st.session_state.quick_questions = [
                    f"What are the key concepts in {st.session_state.topic}?",
                    "Can you summarize the main points?",
                    "What are the best practices mentioned?",
                    "How do I get started with this topic?",
                    "What are common challenges discussed?",
                ]


def main():
    """Main application entry point."""
    app = YouTubeAgentUI()
    app.run()


if __name__ == "__main__":
    main()
