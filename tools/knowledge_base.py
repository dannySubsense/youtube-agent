"""
Knowledge base and RAG chain management for YouTube Agent System.
Extracted from MVP's AdvancedRAG class with enhanced type safety and modularity.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from config.settings import Settings


@dataclass
class RAGChainConfig:
    """Configuration for RAG chain creation."""
    retrieval_k: int = 3
    temperature: float = 0.7
    model: str = "gpt-4o-mini"


class KnowledgeBaseError(Exception):
    """Custom exception for knowledge base related errors."""
    pass


class KnowledgeBaseManager:
    """
    Manages RAG chains and conversational AI for YouTube Agent System.
    
    Extracted from MVP's AdvancedRAG class with:
    - Enhanced type safety and error handling
    - Configurable RAG chain parameters
    - Separation of concerns from vector database management
    - Support for conversation memory and history-aware retrieval
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize KnowledgeBaseManager with settings.
        
        Args:
            settings: Application settings containing LLM configuration
        """
        self.settings = settings
        self.llm = ChatOpenAI(
            model=settings.llm_model, 
            temperature=settings.llm_temperature
        )
    
    def create_advanced_rag_chain(
        self, 
        vectorstore: Chroma, 
        topic: str,
        config: Optional[RAGChainConfig] = None
    ) -> Any:
        """
        Create history-aware RAG chain with memory integration.
        
        Args:
            vectorstore: Chroma vectorstore for retrieval
            topic: Topic name for contextualization
            config: Optional RAG chain configuration
            
        Returns:
            Configured RAG chain ready for conversation
            
        Raises:
            KnowledgeBaseError: If chain creation fails
        """
        if config is None:
            config = RAGChainConfig(
                retrieval_k=self.settings.retrieval_k,
                temperature=self.settings.llm_temperature,
                model=self.settings.llm_model
            )
        
        try:
            # Create retriever with configurable parameters
            retriever = vectorstore.as_retriever(search_kwargs={"k": config.retrieval_k})
            
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
            raise KnowledgeBaseError(f"Failed to create RAG chain: {e}") from e
    
    def create_conversational_rag_chain(
        self,
        vectorstore: Chroma,
        topic: str,
        session_id: str,
        chat_message_history: Any,
        config: Optional[RAGChainConfig] = None
    ) -> RunnableWithMessageHistory:
        """
        Create a conversational RAG chain with persistent message history.
        
        Args:
            vectorstore: Chroma vectorstore for retrieval
            topic: Topic name for contextualization  
            session_id: Unique session identifier
            chat_message_history: Chat message history instance
            config: Optional RAG chain configuration
            
        Returns:
            RunnableWithMessageHistory for conversational interaction
        """
        # Create base RAG chain
        rag_chain = self.create_advanced_rag_chain(vectorstore, topic, config)
        
        # Wrap with message history
        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_message_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        return conversational_chain
    
    def generate_dynamic_greeting(
        self, 
        vectorstore: Chroma, 
        topic: str
    ) -> str:
        """
        Generate a dynamic greeting based on actual database content.
        
        Args:
            vectorstore: Chroma vectorstore to analyze
            topic: Topic name for personalization
            
        Returns:
            Personalized greeting string
        """
        try:
            # Get actual videos from the current database
            collection = vectorstore._collection
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
            
            # Generate video list
            video_list = ""
            for i, (title, channel) in enumerate(videos_info.items(), 1):
                video_list += f"🎥 \"{title}\" by {channel}\n"
            
            # Generate topic-specific capabilities
            capabilities = self._generate_topic_capabilities(topic)
            question_ending = self._generate_question_ending(topic)
            
            greeting = f"""👋 Hi! I'm your YouTube Channel Agent with access to **{video_count} {topic} tutorial videos**!

I have detailed knowledge from these videos:
{video_list.rstrip()}

I can help you with:
{capabilities}

{question_ending}"""
            
            return greeting
            
        except Exception:
            # Fallback to generic greeting if database reading fails
            return f"""👋 Hi! I'm your YouTube Channel Agent with access to tutorial videos about **{topic}**!

I'm ready to answer your questions and help you learn. What would you like to know about {topic}?"""
    
    def _generate_topic_capabilities(self, topic: str) -> str:
        """Generate topic-specific capability descriptions."""
        topic_upper = topic.upper()
        
        if "MCP" in topic_upper:
            return """✅ Step-by-step MCP server creation
✅ Code examples and implementation details  
✅ API integration techniques
✅ Best practices from the tutorials"""
        elif "FINBERT" in topic_upper or "FINANCE" in topic_upper:
            return """✅ Financial sentiment analysis concepts
✅ FinBERT model implementation
✅ Financial data processing techniques
✅ Best practices for financial NLP"""
        else:
            return f"""✅ Key concepts and fundamentals
✅ Practical implementation examples
✅ Best practices and techniques
✅ Step-by-step guidance"""
    
    def _generate_question_ending(self, topic: str) -> str:
        """Generate topic-specific question ending."""
        topic_upper = topic.upper()
        
        if "MCP" in topic_upper:
            return "What would you like to learn about MCP development?"
        elif "FINBERT" in topic_upper or "FINANCE" in topic_upper:
            return f"What would you like to learn about {topic}?"
        else:
            return f"What would you like to learn about {topic}?"
    
    def generate_dynamic_questions(
        self, 
        vectorstore: Chroma, 
        topic: str,
        num_questions: int = 4
    ) -> list[str]:
        """
        Generate contextual quick questions based on actual database content.
        
        Args:
            vectorstore: Chroma vectorstore to sample content from
            topic: Topic name for context
            num_questions: Number of questions to generate
            
        Returns:
            List of relevant questions based on database content
        """
        try:
            # Get documents directly from the vectorstore without embedding computation
            # Use the internal collection to get raw documents
            collection = vectorstore._collection
            
            # Get all documents (limited sample for performance)
            try:
                all_docs = collection.get(
                    include=['documents', 'metadatas'],
                    limit=50  # Limit to avoid performance issues
                )
            except Exception as collection_error:
                print(f"Failed to get documents from collection: {collection_error}")
                return self._generate_fallback_questions(topic)
            
            if not all_docs.get('documents') or not all_docs['documents']:
                return self._generate_fallback_questions(topic)
            
            # Extract content and metadata for context
            content_samples = []
            video_titles = set()
            
            for doc, meta in zip(all_docs['documents'], all_docs['metadatas']):
                if doc and len(doc.strip()) > 50:  # Skip very short content
                    content_samples.append(doc[:500])  # First 500 chars
                    video_titles.add(meta.get('title', 'Unknown'))
                
                if len(content_samples) >= 8:  # Limit sample size
                    break
            
            if not content_samples:
                return self._generate_fallback_questions(topic)
            
            # Create sample context for LLM
            context_text = "\n\n".join(content_samples)
            video_list = "\n".join(f"- {title}" for title in list(video_titles)[:5])
            
            # Generate questions using LLM
            prompt = f"""Based on the following video content from a {topic} knowledge base, generate {num_questions} diverse, engaging questions that would help someone explore this topic.

Video titles in the database:
{video_list}

Sample content from videos:
{context_text}

Generate questions that:
1. Cover different aspects of the topic
2. Range from beginner to intermediate level
3. Are specific enough to get useful answers
4. Would help someone learning about {topic}

Return ONLY the questions, one per line, without numbering or bullets:"""

            try:
                response = self.llm.invoke(prompt)
                questions = [q.strip() for q in response.content.split('\n') if q.strip()]
                
                # Filter and validate questions
                valid_questions = []
                for q in questions:
                    if len(q) > 10 and '?' in q and len(valid_questions) < num_questions:
                        # Clean up the question
                        q = q.strip('- •123456789.').strip()
                        if q and not q.lower().startswith(('question', 'q:')):
                            valid_questions.append(q)
                
                if valid_questions:
                    return valid_questions[:num_questions]
                else:
                    return self._generate_fallback_questions(topic)
                    
            except Exception as llm_error:
                print(f"LLM generation failed: {llm_error}")
                return self._generate_fallback_questions(topic)
                
        except Exception as e:
            print(f"Dynamic question generation failed: {e}")
            return self._generate_fallback_questions(topic)
    
    def _generate_fallback_questions(self, topic: str) -> list[str]:
        """Generate generic fallback questions when dynamic generation fails."""
        return [
            f"What are the key concepts in {topic}?",
            f"How do I get started with {topic}?",
            f"What are the best practices for {topic}?",
            f"Can you explain the most important aspects of {topic}?"
        ]
    
    def extract_sources_from_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract source video information from RAG chain response.
        
        Args:
            response: Response from RAG chain invocation
            
        Returns:
            Dictionary containing source information and citations
        """
        sources = {
            "videos": [],
            "citations": [],
            "context_used": False
        }
        
        try:
            if "context" in response:
                sources["context_used"] = True
                
                # Extract unique sources from context
                seen_videos = set()
                for doc in response["context"]:
                    if hasattr(doc, 'metadata'):
                        metadata = doc.metadata
                        video_id = metadata.get('video_id')
                        
                        if video_id and video_id not in seen_videos:
                            seen_videos.add(video_id)
                            sources["videos"].append({
                                "title": metadata.get('title', 'Unknown'),
                                "channel": metadata.get('channel', 'Unknown'),
                                "url": metadata.get('url', ''),
                                "video_id": video_id
                            })
                
                # Generate citations
                for i, video in enumerate(sources["videos"], 1):
                    citation = f"[{i}] \"{video['title']}\" by {video['channel']}"
                    sources["citations"].append(citation)
            
        except Exception as e:
            sources["error"] = f"Failed to extract sources: {e}"
        
        return sources


def create_knowledge_base_manager(settings: Settings) -> KnowledgeBaseManager:
    """
    Factory function to create KnowledgeBaseManager instance.
    
    Args:
        settings: Application settings
        
    Returns:
        Configured KnowledgeBaseManager instance
    """
    return KnowledgeBaseManager(settings) 