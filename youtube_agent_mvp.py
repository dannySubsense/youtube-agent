# YOUTUBE CHANNEL AGENT - MVP WITH ADVANCED CHAT
# Simple workflow: Topic → Top 5 Videos → Vector DB → Advanced Q&A Chat

# To run the app from the agent-engineering-bootcamp directory:
# cd python-bootcamp-project && source .venv/bin/activate && streamlit run youtube_agent_mvp.py

import streamlit as st
import yaml
import os
import pandas as pd
from pathlib import Path

# YouTube API imports
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Advanced RAG imports
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# PAGE CONFIG
st.set_page_config(
    page_title="YouTube Agent MVP", 
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 YouTube Channel Agent - MVP")
st.markdown("**Simple Workflow**: Topic → Top 5 Videos → Knowledge Base → Advanced Chat")

# CONFIGURATION
MAX_VIDEOS_PER_KNOWLEDGE_BASE = 5  # Limit to optimize API usage and processing time

# LOAD CREDENTIALS
@st.cache_data
def load_credentials():
    try:
        with open('credentials.yml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"❌ Error loading credentials: {e}")
        st.stop()

credentials = load_credentials()
os.environ["OPENAI_API_KEY"] = credentials['openai']

# YOUTUBE CRAWLER CLASS
class YouTubeCrawler:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def search_videos(self, query, max_results=5):
        """Search YouTube for top videos by topic"""
        try:
            # First, search for videos
            search_request = self.youtube.search().list(
                part='snippet',
                q=query,
                type='video',
                maxResults=max_results,
                order='relevance',
                videoDuration='medium'  # Filter out very short/long videos
            )
            search_response = search_request.execute()
            
            # Get video IDs for detailed info
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            # Get detailed video information including duration
            details_request = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=','.join(video_ids)
            )
            details_response = details_request.execute()
            
            videos = []
            for item in details_response['items']:
                # Parse duration from ISO 8601 format (PT4M13S -> 4:13)
                duration_iso = item['contentDetails']['duration']
                duration_readable = self._parse_duration(duration_iso)
                
                video_data = {
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'],
                    'description': item['snippet']['description'][:200] + "...",
                    'published_at': item['snippet']['publishedAt'],
                    'duration': duration_readable,
                    'duration_iso': duration_iso,
                    'view_count': int(item['statistics'].get('viewCount', 0)),
                    'like_count': int(item['statistics'].get('likeCount', 0)),
                    'thumbnail_url': item['snippet']['thumbnails']['medium']['url'],
                    'url': f"https://youtube.com/watch?v={item['id']}"
                }
                videos.append(video_data)
            
            return videos
        except Exception as e:
            st.error(f"YouTube API Error: {e}")
            return []
    
    def _parse_duration(self, duration_iso):
        """Convert ISO 8601 duration to readable format (PT4M13S -> 4:13)"""
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_iso)
        if not match:
            return "Unknown"
        
        hours, minutes, seconds = match.groups()
        hours = int(hours) if hours else 0
        minutes = int(minutes) if minutes else 0
        seconds = int(seconds) if seconds else 0
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def get_transcript(self, video_id):
        """Get transcript for a video with rate limiting handling"""
        import time
        import random
        from youtube_transcript_api._errors import YouTubeRequestFailed
        
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Add small delay between requests to avoid rate limiting
                if attempt > 0:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    st.info(f"⏳ Rate limited. Waiting {delay:.1f}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(delay)
                
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([entry['text'] for entry in transcript_list])
                return transcript_text
                
            except YouTubeRequestFailed as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    if attempt < max_retries - 1:
                        continue  # Retry with backoff
                    else:
                        return f"Rate limited after {max_retries} attempts. Please wait 15-30 minutes and try again."
                else:
                    return f"Transcript not available: {str(e)}"
            except Exception as e:
                return f"Transcript not available: {str(e)}"
    
    def process_videos(self, videos):
        """Process videos and extract transcripts"""
        processed_data = []
        transcript_success_count = 0
        
        st.subheader("📝 Processing Videos...")
        progress_bar = st.progress(0)
        
        for i, video in enumerate(videos):
            with st.expander(f"Processing: {video['title'][:60]}..."):
                st.write(f"**Channel:** {video['channel']}")
                st.write(f"**Duration:** {video['duration']}")
                st.write(f"**Views:** {video['view_count']:,}")
                st.write(f"**URL:** {video['url']}")
                
                # Add delay between requests to avoid rate limiting
                if i > 0:
                    import time
                    delay = 1.5  # 1.5 second delay between videos
                    st.info(f"⏳ Adding {delay}s delay to avoid rate limiting...")
                    time.sleep(delay)
                
                # Get transcript
                transcript = self.get_transcript(video['video_id'])
                
                if "Transcript not available" in transcript:
                    st.warning("⚠️ No transcript available")
                    st.write(f"**Error:** {transcript}")
                else:
                    st.success("✅ Transcript extracted")
                    st.write(f"**Preview:** {transcript[:200]}...")
                    transcript_success_count += 1
                
                processed_data.append({
                    'video_id': video['video_id'],
                    'title': video['title'],
                    'channel': video['channel'],
                    'description': video['description'],
                    'published_at': video['published_at'],
                    'duration': video['duration'],
                    'duration_iso': video['duration_iso'],
                    'view_count': video['view_count'],
                    'like_count': video['like_count'],
                    'thumbnail_url': video['thumbnail_url'],
                    'transcript': transcript,
                    'url': video['url']
                })
            
            progress_bar.progress((i + 1) / len(videos))
        
        # Show summary with rate limiting guidance
        st.info(f"📊 **Summary:** {transcript_success_count} out of {len(videos)} videos have available transcripts")
        
        # Show rate limiting guidance if many failures
        failed_count = len(videos) - transcript_success_count
        if failed_count > len(videos) * 0.5:  # More than 50% failed
            st.warning("⚠️ **High failure rate detected!**")
            st.info("""
            **This might be due to YouTube rate limiting. Try these solutions:**
            
            1. **Wait 15-30 minutes** and try again with fewer videos
            2. **Select only 2-3 videos** instead of many at once
            3. **Choose popular educational channels** that typically have captions
            4. **Search for terms like**: "tutorial", "explained", "course", "guide"
            """)
        
        return processed_data

# RAG SYSTEM CLASS
class AdvancedRAG:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    def load_existing_database(self, topic):
        """Load existing vector database if it exists"""
        db_path = f"data/{topic.replace(' ', '_').upper()}_db"
        if os.path.exists(db_path):
            try:
                vectorstore = Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embedding_function
                )
                return vectorstore, db_path
            except Exception as e:
                st.warning(f"Could not load existing database: {e}")
                return None, None
        return None, None
    
    def create_knowledge_base(self, processed_data, topic):
        """Create vector database from video transcripts"""
        documents = []
        metadatas = []
        successful_videos = []
        failed_videos = []
        
        for item in processed_data:
            if "Transcript not available" not in item['transcript']:
                # Chunk the transcript
                transcript = item['transcript']
                chunk_size = 1000
                
                for i in range(0, len(transcript), chunk_size):
                    chunk = transcript[i:i + chunk_size]
                    documents.append(chunk)
                    metadatas.append({
                        'title': item['title'],
                        'channel': item['channel'],
                        'video_id': item['video_id'],
                        'url': item['url'],
                        'duration': item.get('duration', 'Unknown'),
                        'view_count': item.get('view_count', 0),
                        'like_count': item.get('like_count', 0),
                        'published_at': item.get('published_at', ''),
                        'topic': topic
                    })
                
                successful_videos.append(item['title'])
            else:
                failed_videos.append({
                    'title': item['title'],
                    'error': item['transcript']
                })
        
        # Show detailed results
        if successful_videos:
            st.success(f"✅ **Successfully processed {len(successful_videos)} videos:**")
            for title in successful_videos:
                st.write(f"   • {title}")
        
        if failed_videos:
            st.error(f"❌ **Failed to get transcripts for {len(failed_videos)} videos:**")
            for video in failed_videos:
                with st.expander(f"❌ {video['title']}"):
                    st.write(f"**Error:** {video['error']}")
        
        if not documents:
            st.error("❌ No valid transcripts found to create knowledge base")
            st.info("💡 **Suggestions:**")
            st.write("1. Try searching for educational/tutorial videos (they often have auto-generated captions)")
            st.write("2. Look for videos from popular tech channels that typically enable captions")
            st.write("3. Search for terms like 'tutorial', 'course', 'explained', 'guide'")
            return None
        
        # Create vector store
        db_path = f"data/{topic.replace(' ', '_').upper()}_db"
        vectorstore = Chroma.from_texts(
            texts=documents,
            metadatas=metadatas,
            embedding=self.embedding_function,
            persist_directory=db_path
        )
        
        st.success(f"✅ Knowledge base created with {len(documents)} text chunks from {len(successful_videos)} videos")
        return vectorstore, db_path
    
    def create_advanced_rag_chain(self, vectorstore, topic):
        """Create history-aware RAG chain with memory integration"""
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # STAGE 1: History-aware question contextualization
            contextualize_q_system_prompt = f"""Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.

Focus on {topic} concepts and YouTube video content."""
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # Create history-aware retriever
            history_aware_retriever = create_history_aware_retriever(
                self.llm, retriever, contextualize_q_prompt
            )
            
            # STAGE 2: Answer generation with context and history
            qa_system_prompt = f"""You are an expert YouTube Channel Agent with access to a curated database of {topic} videos.

You have access to full video transcripts and complete metadata for each source. The context includes both the video content and all available metadata about each video source.

When answering questions, you can reference any information available in the context, including video content, metadata, and source details. Use this information to provide comprehensive, accurate responses.

ALWAYS draw from the video content and source metadata in your responses. Use all available information from the context to give specific, detailed answers.

Context from videos:
{{context}}"""
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            # Create question-answer chain
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            
            # Combine into full RAG chain
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            return rag_chain
            
        except Exception as e:
            st.error(f"❌ Error creating advanced RAG chain: {e}")
            return None

# MAIN APPLICATION
def main():
    # DATABASE SELECTION OR CREATION
    st.header("🔍 Select Database or Create New")
    
    # Check for existing databases
    rag = AdvancedRAG()
    
    # DYNAMIC DATABASE DISCOVERY - scan data folder for existing databases
    data_dir = Path("data")
    available_databases = []
    
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.endswith("_db"):
                # Convert from file format back to readable format
                topic_name = item.name.replace("_db", "").replace("_", " ").title()
                available_databases.append(topic_name)
    
    # Sort for consistent display
    available_databases.sort()
    
    # Fallback if no databases found
    if not available_databases:
        available_databases = ["No databases found - Create one below!"]
    
    # Database selection options
    database_option = st.radio(
        "Choose your option:",
        ["📚 Use Existing Database", "🆕 Create New Database"],
        horizontal=True,
        help="Select an existing database or create a new one with different content"
    )
    
    if database_option == "📚 Use Existing Database":
        # Dropdown for existing databases
        selected_db = st.selectbox(
            "Select database:",
            available_databases,
            help="Choose from available knowledge bases",
            key="database_selector"
        )
        
        # Handle case where no databases exist
        if selected_db == "No databases found - Create one below!":
            st.info("👆 No existing databases found. Please create a new one below.")
            topic = ""
        else:
            topic = selected_db
            
            # ACTUALLY LOAD THE EXISTING DATABASE
            rag = AdvancedRAG()
            vectorstore, db_path = rag.load_existing_database(topic)
            
            if vectorstore:
                st.info(f"✅ Successfully loaded database: **{topic}**")
                
                # Clear chat history when switching to a different database
                previous_topic = st.session_state.get('topic', '')
                if topic != previous_topic and previous_topic != '':
                    # Database has changed - clear chat for fresh greeting
                    if 'chat_messages' in st.session_state:
                        del st.session_state['chat_messages']
                    if 'rag_chain' in st.session_state:
                        del st.session_state.rag_chain
                
                # SET SESSION STATE FOR CHAT INTERFACE
                st.session_state.vectorstore = vectorstore
                st.session_state.topic = topic
            else:
                st.error(f"❌ Failed to load database: {topic}")
                return
            
        # Clear all video creation session state when switching to existing database
        video_creation_keys = [
            'creating_new', 
            'found_videos', 
            'selected_videos_array', 
            'current_creation_topic',
            'proceed_with_creation',
            'creation_topic'
        ]
        
        for key in video_creation_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Also clear video selection checkboxes
        keys_to_delete = [key for key in st.session_state.keys() if key.startswith('video_select_')]
        for key in keys_to_delete:
            del st.session_state[key]
            
    else:  # Create New Database
        st.session_state.creating_new = True
        
        # Clear previous chat and database state when starting new database creation
        if 'chat_messages' in st.session_state:
            del st.session_state['chat_messages']
        if 'vectorstore' in st.session_state:
            del st.session_state.vectorstore
        if 'rag_chain' in st.session_state:
            del st.session_state.rag_chain
        if 'topic' in st.session_state:
            del st.session_state.topic
        
        # Use form for better Enter key handling
        with st.form("new_topic_form", clear_on_submit=False):
            topic = st.text_input(
                "Enter topic for new database:", 
                value="",
                placeholder="e.g., Python programming tutorials, React development, Machine Learning basics, etc.",
                help="⚠️ Enter a NEW topic. This will create a completely new knowledge base with different YouTube videos",
                key="new_topic_input"
            )
            
            # Form submit button
            submitted = st.form_submit_button("🚀 Create Knowledge Base", type="primary")
            
            # Show validation info inside form for immediate feedback
            if topic.strip():
                if topic.strip() in available_databases:
                    st.error(f"⚠️ Database '{topic.strip()}' already exists! Please enter a different topic.")
                else:
                    st.info(f"✅ Topic '{topic.strip()}' is available. Click the button or press Enter to create.")
            
            # Process form submission
            if submitted:
                if not topic.strip():
                    st.error("Please enter a topic name")
                elif topic.strip() in available_databases:
                    st.error(f"⚠️ Database '{topic.strip()}' already exists! Please enter a different topic.")
                else:
                    # Valid topic - proceed with creation
                    st.session_state.proceed_with_creation = True
                    st.session_state.creation_topic = topic.strip()
                    st.rerun()  # Immediate refresh to process creation
        
        # For flow control - if no topic was entered or it's invalid, stop here
        # BUT don't stop if we have a pending creation
        if (not topic or not topic.strip() or topic.strip() in available_databases) and not hasattr(st.session_state, 'proceed_with_creation'):
            return
    
    # Check for pending creation OR existing video selection interface
    if (hasattr(st.session_state, 'proceed_with_creation') and st.session_state.proceed_with_creation) or 'found_videos' in st.session_state:
        
        # Handle initial creation
        if hasattr(st.session_state, 'proceed_with_creation') and st.session_state.proceed_with_creation:
            del st.session_state.proceed_with_creation  # Clear the flag
            creation_topic = st.session_state.get('creation_topic', topic)
            if 'creation_topic' in st.session_state:
                del st.session_state.creation_topic
            # Save the topic for future form submissions
            st.session_state.current_creation_topic = creation_topic
        else:
            # We're returning to the video selection interface
            creation_topic = st.session_state.get('current_creation_topic', 'Unknown Topic')
        crawler = YouTubeCrawler(credentials['youtube'])
        
        # Multi-strategy search for better precision
        st.header("🔍 Searching for Videos")
        
        # Generate dynamic search strategies for any topic
        base_searches = [creation_topic]
        
        # Add universal search variations that work for any topic
        base_searches.extend([
            f"{creation_topic} tutorial",
            f"{creation_topic} guide", 
            f"{creation_topic} explained",
            f"how to {creation_topic}",
            f"{creation_topic} walkthrough"
        ])
        
        # Search with multiple strategies and combine results
        all_videos = []
        seen_video_ids = set()
        
        for search_term in base_searches[:3]:  # Limit to 3 searches to avoid API limits
            with st.spinner(f"🔍 Searching: '{search_term}'"):
                videos = crawler.search_videos(search_term, max_results=8)  # More videos per search
                
                # Remove duplicates
                for video in videos:
                    if video['video_id'] not in seen_video_ids:
                        all_videos.append(video)
                        seen_video_ids.add(video['video_id'])
        
        if not all_videos:
            st.error("No videos found for this topic. Try a different search term.")
            return
        
        # VIDEO SELECTION INTERFACE
        st.header("📋 Select Videos for Your Knowledge Base")
        st.info(f"🎯 **Found {len(all_videos)} videos** - Choose up to **{MAX_VIDEOS_PER_KNOWLEDGE_BASE} videos** to include in your knowledge base")
        
        # Show limit information
        if len(all_videos) > MAX_VIDEOS_PER_KNOWLEDGE_BASE:
            st.warning(f"💡 **Tip**: To optimize processing time and API usage, you can select up to {MAX_VIDEOS_PER_KNOWLEDGE_BASE} videos. Choose the most relevant ones for your topic.")
        
        # Store videos in session state for persistence
        if 'found_videos' not in st.session_state:
            st.session_state.found_videos = all_videos
        
        # Initialize selected videos array if it doesn't exist
        if 'selected_videos_array' not in st.session_state:
            st.session_state.selected_videos_array = []
        
        # Video selection form
        with st.form("video_selection_form", clear_on_submit=False):
            st.markdown("**📺 Available Videos:**")
            st.markdown(f"*Review each video and select up to **{MAX_VIDEOS_PER_KNOWLEDGE_BASE} videos** that are specifically relevant to your topic*")
            
            # Use global video limit
            MAX_VIDEOS = MAX_VIDEOS_PER_KNOWLEDGE_BASE
            
            # Create checkboxes and collect selections DURING form creation
            video_selections = []
            
            # Create columns for better layout
            for i, video in enumerate(st.session_state.found_videos):
                # Create expandable card for each video
                with st.expander(f"📹 {video['title'][:70]}..." if len(video['title']) > 70 else f"📹 {video['title']}", expanded=False):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Video selection checkbox - collect selections immediately
                        checkbox_key = f"video_select_{i}"
                        
                        # Check if this video is already in the persistent array
                        video_id = video['video_id']
                        is_already_selected = video_id in [v['video_id'] for v in st.session_state.selected_videos_array]
                        current_selected_count = len(st.session_state.selected_videos_array)
                        
                        # Disable checkbox if limit reached and this video isn't already selected
                        is_disabled = (current_selected_count >= MAX_VIDEOS) and not is_already_selected
                        
                        selected = st.checkbox(
                            "Include", 
                            key=checkbox_key,
                            value=is_already_selected,  # Set checkbox state based on persistent array
                            disabled=is_disabled,
                            help=f"Select to include this video in your knowledge base (Limit: {MAX_VIDEOS} videos)" if not is_disabled else f"Maximum {MAX_VIDEOS} videos already selected"
                        )
                        
                        # Immediately update persistent array based on checkbox state
                        if selected:
                            # Add to array if checked and not already there, and under limit
                            if video_id not in [v['video_id'] for v in st.session_state.selected_videos_array]:
                                if len(st.session_state.selected_videos_array) < MAX_VIDEOS:
                                    st.session_state.selected_videos_array.append(video)
                                else:
                                    # This shouldn't happen due to disabled checkbox, but just in case
                                    st.warning(f"⚠️ Maximum {MAX_VIDEOS} videos already selected!")
                        else:
                            # Remove from array if unchecked
                            st.session_state.selected_videos_array = [
                                v for v in st.session_state.selected_videos_array 
                                if v['video_id'] != video_id
                            ]
                        
                        video_selections.append((i, video, selected))
                    
                    with col2:
                        # Video details
                        st.markdown(f"""
                        **📺 Title:** {video['title']}  
                        **👤 Channel:** {video['channel']}  
                        **⏱️ Duration:** {video['duration']}  
                        **👁️ Views:** {video['view_count']:,}  
                        **📝 Description:** {video['description']}  
                        **🔗 [Watch Video]({video['url']})**
                        """)
            
            # Selection summary - count from PERSISTENT ARRAY
            current_selected = len(st.session_state.selected_videos_array)
            st.markdown("---")
            
            # Show selection count with limit and color coding
            if current_selected == 0:
                st.markdown(f"**📝 Selected: {current_selected}/{MAX_VIDEOS} videos** - *Select at least 1 video*")
            elif current_selected < MAX_VIDEOS:
                st.markdown(f"**✅ Selected: {current_selected}/{MAX_VIDEOS} videos** - *You can select {MAX_VIDEOS - current_selected} more*")
            else:
                st.markdown(f"**🎯 Selected: {current_selected}/{MAX_VIDEOS} videos** - *Maximum reached!*")
            
            # Show selected videos for user clarity
            if st.session_state.selected_videos_array:
                st.write("**Selected videos:**")
                for i, video in enumerate(st.session_state.selected_videos_array):
                    st.write(f"   {i+1}. {video.get('title', 'NO TITLE')[:50]}...")
            
            # Buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                select_top_5 = st.form_submit_button("📥 Select Top 5", help=f"Select first {MAX_VIDEOS} videos automatically")
            with col2:
                clear_all = st.form_submit_button("🗑️ Clear All", help="Deselect all videos") 
            with col3:
                process_selected = st.form_submit_button("🚀 Build Knowledge Base", type="primary", help="Process selected videos")
        
        # Handle form actions OUTSIDE the form context
        if select_top_5:
            # Use global video limit
            MAX_VIDEOS = MAX_VIDEOS_PER_KNOWLEDGE_BASE
            
            # Add first 5 videos to persistent array
            st.session_state.selected_videos_array = st.session_state.found_videos[:MAX_VIDEOS]
            
            # Clear checkbox keys to force refresh with new values
            keys_to_delete = [key for key in st.session_state.keys() if key.startswith('video_select_')]
            for key in keys_to_delete:
                del st.session_state[key]
                
            selected_count = len(st.session_state.selected_videos_array)
            st.success(f"✅ Selected top {selected_count} videos!")
            st.rerun()
        
        if clear_all:
            # Clear persistent array
            st.session_state.selected_videos_array = []
            
            # Clear checkbox keys to force refresh with new values
            keys_to_delete = [key for key in st.session_state.keys() if key.startswith('video_select_')]
            for key in keys_to_delete:
                del st.session_state[key]
                
            st.success("🗑️ Cleared all selections!")
            st.rerun()
        
        if process_selected:
            # Use global video limit
            MAX_VIDEOS = MAX_VIDEOS_PER_KNOWLEDGE_BASE
            
            # Get selected videos from PERSISTENT ARRAY
            final_selected_videos = st.session_state.selected_videos_array.copy()
            
            if not final_selected_videos:
                st.error("❌ Please select at least one video to build your knowledge base.")
            elif len(final_selected_videos) > MAX_VIDEOS:
                st.error(f"❌ Too many videos selected! Please select no more than {MAX_VIDEOS} videos.")
                st.info("💡 Use the 'Clear All' button and select fewer videos.")
            else:
                # Show what we're about to process
                st.info(f"🚀 **Processing {len(final_selected_videos)} selected videos for topic: {creation_topic}**")
                for i, video in enumerate(final_selected_videos):
                    st.write(f"   {i+1}. {video.get('title', 'NO TITLE')}")
                
                # PROCESS IMMEDIATELY - no session state needed
                try:
                    st.header("📝 Processing Selected Videos")
                    crawler = YouTubeCrawler(credentials['youtube'])
                    processed_data = crawler.process_videos(final_selected_videos)
                    
                    # Create knowledge base
                    st.header("🧠 Building Knowledge Base")
                    with st.spinner("Creating vector database..."):
                        result = rag.create_knowledge_base(processed_data, creation_topic)
                        
                        if result:
                            vectorstore, db_path = result
                            
                            # Clear old RAG chain when creating new database
                            if 'rag_chain' in st.session_state:
                                del st.session_state.rag_chain
                            
                            st.session_state.vectorstore = vectorstore
                            st.session_state.topic = creation_topic
                            st.success("🎉 Knowledge base ready! You can now chat below.")
                            
                            # Clean up session state but keep vectorstore and topic
                            cleanup_keys = ['creating_new', 'found_videos', 'selected_videos_array', 'current_creation_topic']
                            for key in cleanup_keys:
                                if key in st.session_state:
                                    del st.session_state[key]
                            
                            # Clear video selection checkboxes
                            keys_to_delete = [key for key in st.session_state.keys() if key.startswith('video_select_')]
                            for key in keys_to_delete:
                                del st.session_state[key]
                            
                            st.rerun()  # Refresh to show chat interface
                            
                        else:
                            st.error("❌ Failed to create knowledge base")
                            
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
                    st.write("Debug info:", str(e))

    # OLD PROCESSING LOGIC REMOVED - now processing happens immediately in the form above
    
    # CHAT WITH YOUR AGENT
    if hasattr(st.session_state, 'vectorstore'):
        st.header("💬 Chat with Your Agent")
        
        # Show topic info
        st.info(f"📚 Knowledge Base: **{st.session_state.topic}** | Ready for questions!")
        
        # Initialize chat memory
        msgs = StreamlitChatMessageHistory(key="chat_messages")
        if len(msgs.messages) == 0:
            # Generate dynamic greeting based on actual database content
            try:
                # Get actual videos from the current database
                collection = st.session_state.vectorstore._collection
                all_results = collection.get(include=['metadatas'])
                
                # Extract unique videos from metadata
                videos_info = {}
                for metadata in all_results['metadatas']:
                    title = metadata.get('title', 'Unknown')
                    channel = metadata.get('channel', 'Unknown')
                    
                    if title not in videos_info and title != 'Unknown':
                        videos_info[title] = channel
                
                # Create dynamic greeting
                video_count = len(videos_info)
                topic = st.session_state.topic
                
                # Generate video list
                video_list = ""
                for i, (title, channel) in enumerate(videos_info.items(), 1):
                    video_list += f"🎥 \"{title}\" by {channel}\n"
                
                # Generate topic-specific capabilities
                if "MCP" in topic.upper():
                    capabilities = """✅ Step-by-step MCP server creation
✅ Code examples and implementation details  
✅ API integration techniques
✅ Best practices from the tutorials"""
                    question_ending = "What would you like to learn about MCP development?"
                elif "FINBERT" in topic.upper() or "FINANCE" in topic.upper():
                    capabilities = """✅ Financial sentiment analysis concepts
✅ FinBERT model implementation
✅ Financial data processing techniques
✅ Best practices for financial NLP"""
                    question_ending = f"What would you like to learn about {topic}?"
                else:
                    capabilities = f"""✅ Key concepts and fundamentals
✅ Practical implementation examples
✅ Best practices and techniques
✅ Step-by-step guidance"""
                    question_ending = f"What would you like to learn about {topic}?"
                
                greeting = f"""👋 Hi! I'm your YouTube Channel Agent with access to **{video_count} {topic} tutorial videos**!

I have detailed knowledge from these videos:
{video_list.rstrip()}

I can help you with:
{capabilities}

{question_ending}"""
                
                msgs.add_ai_message(greeting)
                
            except Exception as e:
                # Fallback to generic greeting if database reading fails
                msgs.add_ai_message(f"""👋 Hi! I'm your YouTube Channel Agent with access to tutorial videos about **{st.session_state.topic}**!

I'm ready to answer your questions and help you learn. What would you like to know about {st.session_state.topic}?""")
        
        # Create RAG chain (fix the error)
        if 'rag_chain' not in st.session_state:
            with st.spinner("🧠 Initializing chat system..."):
                try:
                    st.session_state.rag_chain = rag.create_advanced_rag_chain(st.session_state.vectorstore, st.session_state.topic)
                    if st.session_state.rag_chain is None:
                        st.error("❌ Failed to initialize chat system")
                        return
                except Exception as e:
                    st.error(f"❌ Error initializing chat: {e}")
                    return
        
        # Quick Questions (better positioned)
        with st.expander("💡 Quick Questions - Click to Ask"):
            # Generate contextual quick questions based on topic
            if "MCP" in st.session_state.topic.upper():
                quick_questions = [
                    "What is MCP and how does it work?",
                    "How do I create an MCP server?",
                    "What are the key components of MCP?",
                    "Can you show me a practical example?",
                    "What are common MCP development challenges?"
                ]
            else:
                quick_questions = [
                    f"What are the key concepts in {st.session_state.topic}?",
                    "Can you summarize the main points?",
                    "What are the best practices mentioned?",
                    "How do I get started with this topic?",
                    "What are common challenges discussed?"
                ]
            
            # Display questions in columns for better layout
            cols = st.columns(2)
            for i, q in enumerate(quick_questions):
                with cols[i % 2]:
                    if st.button(q, key=f"quick_{hash(q)}", use_container_width=True):
                        st.session_state[f"submit_{i}"] = q
        
        # Chat management
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**💬 Chat Messages:** {len(msgs.messages)}")
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                msgs.clear()
                msgs.add_ai_message(f"👋 Chat cleared! I'm ready for a fresh conversation about {st.session_state.topic}.")
                st.rerun()
        
        # Check for submitted questions (from buttons or manual input)
        submitted_question = None
        
        # Check for button submissions
        for i in range(len(quick_questions)):
            if f"submit_{i}" in st.session_state:
                submitted_question = st.session_state[f"submit_{i}"]
                del st.session_state[f"submit_{i}"]
                break
        
        # PROMINENT CHAT INPUT
        st.markdown("### 💬 Ask Your Question:")
        if question := st.chat_input(f"Type your question about {st.session_state.topic} here... (I remember our conversation!)"):
            submitted_question = question
        
        # Process any submitted question
        if submitted_question:
            # Generate response with history awareness
            # RunnableWithMessageHistory automatically manages adding messages to msgs
            try:
                # Wrap RAG chain with message history
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
                    config={"configurable": {"session_id": "youtube_agent"}}
                )
                
                # Only manually add sources if they exist
                if 'context' in response and response['context']:
                    sources_seen = set()
                    sources_text = "**📚 Sources Used:**\n"
                    for doc in response['context']:
                        title = doc.metadata.get('title', 'Unknown')
                        channel = doc.metadata.get('channel', 'Unknown')
                        url = doc.metadata.get('url', '#')
                        
                        source_key = f"{title}|{channel}"
                        if source_key not in sources_seen:
                            sources_text += f"- **{title}** by {channel} - [🔗 Watch]({url})\n"
                            sources_seen.add(source_key)
                    
                    # Add sources as a separate message
                    msgs.add_ai_message(sources_text)
                
            except Exception as e:
                error_msg = f"❌ Error processing question: {e}"
                msgs.add_ai_message(error_msg)
        
        # Display chat history AFTER processing
        for msg in msgs.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)
    
    # FOOTER
    st.markdown("---")
    st.markdown("**YouTube Channel Agent MVP** | Agent Engineering Bootcamp Week 2")

if __name__ == "__main__":
    main() 