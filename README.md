# YouTube Agent - Intelligent Knowledge Base Builder

A comprehensive YouTube video analysis and knowledge base building system that combines the power of AI with efficient video content processing. The system enables intelligent content discovery, transcript analysis, and knowledge base curation with an intuitive user interface.

## 🚀 Project Overview

YouTube Agent is a modular system that helps you:

- 🔍 **Search and discover** YouTube videos by topic
- 📜 **Extract and process** video transcripts automatically  
- 🧠 **Build knowledge bases** using RAG (Retrieval-Augmented Generation)
- 💬 **Chat with your video content** using AI-powered conversations
- ⚡ **Quick Questions** feature for instant insights
- 🎯 **Smart content evaluation** with technology freshness scoring

## 🛠️ Architecture

The system uses a modular architecture with clear separation of concerns:

- **`core/`** - Core business logic (video search, transcript processing, local vector database)
- **`mcp_integration/`** - Model Context Protocol integration layer
- **`agents/`** - AI agent orchestration using LangChain
- **`api/`** - FastAPI application layer with REST endpoints
- **`ui/`** - User interface components (Streamlit dashboard)
- **`config/`** - Configuration and settings management
- **`tests/`** - Comprehensive test suite

## 🎬 YouTube MCP Server Integration

This project integrates with a powerful **YouTube MCP Server** that provides real-time YouTube Data API access. The MCP server features:

### 14 Complete Functions

1. **`get_video_details`** - Comprehensive video information including metadata and statistics
2. **`get_playlist_details`** - Retrieve playlist information and metadata
3. **`get_playlist_items`** - List videos within playlists with details
4. **`get_channel_details`** - Channel information including subscriber count and descriptions
5. **`get_video_categories`** - Available video categories for specific regions
6. **`get_channel_videos`** - Recent videos from YouTube channels
7. **`search_videos`** - Advanced YouTube video search with filters
8. **`get_trending_videos`** - Trending content for specific regions
9. **`get_video_comments`** - Video comments with sorting options
10. **`analyze_video_engagement`** - Engagement metrics and insights
11. **`get_channel_playlists`** - List playlists from YouTube channels
12. **`get_video_caption_info`** - Available caption/transcript information
13. **`evaluate_video_for_knowledge_base`** - **Intelligent content evaluation with freshness scoring**
14. **`get_video_transcript`** - **Extract full transcript content from videos**

### Key MCP Features

✅ **Real-time data** from YouTube Data API v3  
✅ **Comprehensive error handling** and API quota management  
✅ **Multiple URL format support** (youtube.com, youtu.be, @usernames, channel IDs)  
✅ **Intelligent content evaluation** with technology freshness scoring  
✅ **Flexible search and filtering** options  
✅ **Engagement analysis** with industry benchmarks  
✅ **Regional content support** for trending and categories  
✅ **MCP protocol compliance** for seamless AI integration

**YouTube MCP Server Repository**: [https://github.com/dannySubsense/youtube-mcp-server](https://github.com/dannySubsense/youtube-mcp-server)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- YouTube Data API v3 key
- OpenAI API key
- **No external database required** - uses local Chroma vector database

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd youtube-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # or using uv
   uv pip install -r requirements.txt
   ```

3. **Set up credentials**
   Create a `credentials.yml` file in the project root with your API keys:
   ```yaml
   youtube: 'your-youtube-api-key-here'
   openai: 'your-openai-api-key-here'
   ```

### Running the Application

#### Option 1: Modular Architecture (Recommended)
```bash
uv run streamlit run ui/streamlit/streamlit_runner.py
```

#### Option 2: MVP Version (Reference)
```bash
uv run streamlit run youtube_agent_mvp.py
```

## 🧪 Testing

### Test Structure

The project includes comprehensive testing across multiple layers:

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_ui_components.py      # UI component tests
│   ├── test_mcp_client.py         # MCP client functionality tests
│   ├── test_mcp_server.py         # MCP server integration tests
│   ├── test_mcp_property_based.py # Property-based testing
│   └── test_video_search.py       # Video search logic tests
├── integration/             # Integration tests for workflows
│   ├── test_ui_workflow.py        # End-to-end UI workflow tests
│   └── test_modular_integration.py # Modular architecture integration
├── test_mcp_integration.py  # MCP protocol integration tests
└── fixtures/               # Test data and fixtures
```

### Running Tests

#### Run All Tests
```bash
# Using pytest
pytest

# Using uv
uv run pytest
```

#### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# MCP integration tests
pytest tests/test_mcp_integration.py

# UI workflow tests
pytest tests/integration/test_ui_workflow.py
```

#### Run Tests with Coverage
```bash
pytest --cov=. --cov-report=html
```

#### Property-Based Testing
```bash
# Run property-based tests for robust validation
pytest tests/unit/test_mcp_property_based.py -v
```

### Test Features

- **Unit Tests**: Test individual functions and components in isolation
- **Integration Tests**: Test complete workflows and component interactions  
- **Property-Based Tests**: Use Hypothesis for robust edge case testing
- **UI Workflow Tests**: Simulate user interactions and validate behavior
- **MCP Protocol Tests**: Validate Model Context Protocol integration
- **Fixtures**: Reusable test data for consistent testing

### Test Gates

The project follows Test-Driven Development (TDD) with mandatory gates:

- ✅ **`black --check`** - Code formatting
- ✅ **`flake8`** - Linting 
- ✅ **`mypy`** - Type checking
- ✅ **`pytest`** - All tests must pass

## 🏗️ Development

### Code Quality

```bash
# Format code
black .

# Check linting
flake8

# Type checking  
mypy .

# Run all quality checks
black . && flake8 && mypy . && pytest
```



## 📊 Features

### Core Functionality

- **Topic-Based Video Discovery**: Search YouTube for videos by topic or keyword
- **Intelligent Transcript Processing**: Extract, clean, and process video transcripts
- **Local Vector Database**: Store and index video content using local Chroma database (no external database required)
- **Advanced RAG**: Retrieve relevant content for AI-powered conversations
- **Conversation Memory**: Maintain context across chat sessions

### User Interface

- **Streamlit Dashboard**: Intuitive web interface for all operations
- **Quick Questions**: Pre-generated questions for instant insights
- **Real-time Chat**: Interactive conversations with your video knowledge base
- **Video Source Display**: Clear attribution and source tracking
- **Database Management**: Easy switching between different local knowledge bases (stored in `data/` directory)

### AI Integration

- **LangChain Orchestration**: Sophisticated agent coordination
- **OpenAI Integration**: GPT-powered analysis and conversations
- **MCP Protocol**: Seamless integration with Model Context Protocol
- **Content Evaluation**: Smart scoring for knowledge base curation

## 🔧 Configuration

### Credentials Configuration

Create a `credentials.yml` file in the project root:

```yaml
# Required
youtube: 'your-youtube-api-key-here'
openai: 'your-openai-api-key-here'
```

**Note**: The system uses a local `credentials.yml` file instead of environment variables for API key management. This file is automatically ignored by git for security.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the development philosophy and code guidelines
4. Ensure all tests pass (`pytest`)
5. Run quality checks (`black . && flake8 && mypy .`)
6. Commit changes using Conventional Commits format
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YouTube Data API v3** for video data access
- **OpenAI** for AI-powered analysis
- **LangChain** for agent orchestration
- **Streamlit** for the user interface
- **Chroma** for local vector database functionality
- **Model Context Protocol** for AI integration standards

---

**Ready to build intelligent knowledge bases from YouTube content? Get started today!** 🚀
