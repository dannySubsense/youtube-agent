"""
Vector database management for YouTube Agent System.
Extracted from MVP's AdvancedRAG class with enhanced type safety and error handling.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from config.settings import Settings


@dataclass
class DatabaseInfo:
    """Information about a vector database."""
    name: str
    path: str
    topic: str
    document_count: int
    created_at: Optional[str] = None


class VectorDatabaseError(Exception):
    """Custom exception for vector database related errors."""
    pass


class VectorDatabaseManager:
    """
    Manages Chroma vector databases for YouTube Agent System.
    
    Extracted from MVP's AdvancedRAG class with:
    - Enhanced error handling and validation
    - Proper type hints for all operations
    - Separation of concerns from RAG chain creation
    - Improved database discovery and management
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize VectorDatabaseManager with settings.
        
        Args:
            settings: Application settings containing embedding configuration
        """
        self.settings = settings
        self.embedding_function = OpenAIEmbeddings(model=settings.embedding_model)
        self.data_directory = Path(settings.data_directory)
        self.data_directory.mkdir(exist_ok=True)
    
    def create_knowledge_base(
        self, 
        documents: List[str], 
        metadatas: List[Dict[str, Any]], 
        topic: str
    ) -> Tuple[Chroma, str]:
        """
        Create a new vector database from documents and metadata.
        
        Args:
            documents: List of text chunks to embed
            metadatas: List of metadata dictionaries for each document
            topic: Topic name for the knowledge base
            
        Returns:
            Tuple of (vectorstore, database_path)
            
        Raises:
            VectorDatabaseError: If database creation fails
        """
        if not documents:
            raise VectorDatabaseError("No documents provided for knowledge base creation")
        
        if len(documents) != len(metadatas):
            raise VectorDatabaseError("Documents and metadatas lists must have same length")
        
        try:
            # Create database path
            db_path = self._generate_database_path(topic)
            
            # Create vector store
            vectorstore = Chroma.from_texts(
                texts=documents,
                metadatas=metadatas,
                embedding=self.embedding_function,
                persist_directory=str(db_path)
            )
            
            return vectorstore, str(db_path)
            
        except Exception as e:
            raise VectorDatabaseError(f"Failed to create knowledge base: {e}") from e
    
    def load_existing_database(self, topic: str) -> Optional[Tuple[Chroma, str]]:
        """
        Load an existing vector database by topic.
        
        Args:
            topic: Topic name to load
            
        Returns:
            Tuple of (vectorstore, database_path) or None if not found
        """
        db_path = self._generate_database_path(topic)
        
        if not db_path.exists():
            return None
        
        try:
            vectorstore = Chroma(
                persist_directory=str(db_path),
                embedding_function=self.embedding_function
            )
            return vectorstore, str(db_path)
            
        except Exception as e:
            raise VectorDatabaseError(f"Failed to load database '{topic}': {e}") from e
    
    def discover_databases(self) -> List[DatabaseInfo]:
        """
        Discover all available vector databases in the data directory.
        
        Returns:
            List of DatabaseInfo objects for available databases
        """
        databases = []
        
        if not self.data_directory.exists():
            return databases
        
        for item in self.data_directory.iterdir():
            if item.is_dir() and item.name.endswith("_db"):
                # Convert from file format back to readable format
                topic_name = item.name.replace("_db", "").replace("_", " ").title()
                
                # Try to get document count
                document_count = 0
                try:
                    vectorstore = Chroma(
                        persist_directory=str(item),
                        embedding_function=self.embedding_function
                    )
                    # Get collection info if possible
                    collection = vectorstore._collection
                    document_count = collection.count() if collection else 0
                except:
                    document_count = 0
                
                db_info = DatabaseInfo(
                    name=topic_name,
                    path=str(item),
                    topic=topic_name,
                    document_count=document_count
                )
                databases.append(db_info)
        
        # Sort by name for consistent ordering
        databases.sort(key=lambda x: x.name)
        return databases
    
    def delete_database(self, topic: str) -> bool:
        """
        Delete a vector database.
        
        Args:
            topic: Topic name to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        db_path = self._generate_database_path(topic)
        
        if not db_path.exists():
            return False
        
        try:
            import shutil
            shutil.rmtree(db_path)
            return True
        except Exception:
            return False
    
    def get_database_stats(self, vectorstore: Chroma) -> Dict[str, Any]:
        """
        Get statistics about a vector database.
        
        Args:
            vectorstore: Chroma vectorstore instance
            
        Returns:
            Dictionary containing database statistics
        """
        try:
            collection = vectorstore._collection
            
            if not collection:
                return {"error": "No collection found"}
            
            # Get all documents with metadata
            all_results = collection.get(include=['metadatas'])
            
            # Calculate statistics
            total_chunks = len(all_results['metadatas'])
            
            # Count unique videos
            unique_videos = set()
            for metadata in all_results['metadatas']:
                video_id = metadata.get('video_id', 'unknown')
                unique_videos.add(video_id)
            
            # Extract video titles and channels
            videos_info = {}
            for metadata in all_results['metadatas']:
                title = metadata.get('title', 'Unknown')
                channel = metadata.get('channel', 'Unknown')
                
                if title not in videos_info and title != 'Unknown':
                    videos_info[title] = channel
            
            return {
                "total_chunks": total_chunks,
                "unique_videos": len(unique_videos),
                "videos_info": videos_info,
                "chunk_count_per_video": total_chunks / len(unique_videos) if unique_videos else 0
            }
            
        except Exception as e:
            return {"error": f"Failed to get database stats: {e}"}
    
    def _generate_database_path(self, topic: str) -> Path:
        """
        Generate standardized database path for a topic.
        
        Args:
            topic: Topic name
            
        Returns:
            Path object for the database directory
        """
        # Convert topic to filesystem-safe format
        safe_topic = topic.replace(' ', '_').upper()
        db_name = f"{safe_topic}_db"
        return self.data_directory / db_name
    
    def prepare_documents_for_vectorstore(
        self, 
        processed_data: List[Dict[str, Any]], 
        topic: str,
        chunk_size: Optional[int] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[Dict[str, str]]]:
        """
        Prepare processed video data for vector database creation.
        
        Args:
            processed_data: List of processed video data
            topic: Topic name for metadata
            chunk_size: Size of text chunks (uses settings default if None)
            
        Returns:
            Tuple of (documents, metadatas, successful_videos, failed_videos)
        """
        if chunk_size is None:
            chunk_size = self.settings.transcript_chunk_size
        
        documents = []
        metadatas = []
        successful_videos = []
        failed_videos = []
        
        for item in processed_data:
            if "Transcript not available" not in item.get('transcript', ''):
                # Chunk the transcript
                transcript = item['transcript']
                
                for i in range(0, len(transcript), chunk_size):
                    chunk = transcript[i:i + chunk_size]
                    documents.append(chunk)
                    metadatas.append({
                        'title': item['title'],
                        'channel': item.get('channel', 'Unknown'),
                        'video_id': item['video_id'],
                        'url': item.get('url', ''),
                        'duration': item.get('duration', 'Unknown'),
                        'view_count': item.get('view_count', 0),
                        'like_count': item.get('like_count', 0),
                        'published_at': item.get('published_at', ''),
                        'topic': topic,
                        'chunk_index': i // chunk_size
                    })
                
                successful_videos.append(item['title'])
            else:
                failed_videos.append({
                    'title': item['title'],
                    'error': item['transcript']
                })
        
        return documents, metadatas, successful_videos, failed_videos


def create_vector_database_manager(settings: Settings) -> VectorDatabaseManager:
    """
    Factory function to create VectorDatabaseManager instance.
    
    Args:
        settings: Application settings
        
    Returns:
        Configured VectorDatabaseManager instance
    """
    return VectorDatabaseManager(settings) 