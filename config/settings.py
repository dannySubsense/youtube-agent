"""
Settings and configuration management for YouTube Agent System.
Extracted from MVP's credential loading logic with enhanced error handling.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Settings:
    """Configuration settings for YouTube Agent System."""
    
    # API Keys
    youtube_api_key: str
    openai_api_key: str
    
    # Database Configuration
    data_directory: str = "data"
    max_videos_per_knowledge_base: int = 5
    
    # Processing Configuration
    transcript_chunk_size: int = 1000
    rate_limit_delay: float = 1.5
    max_retries: int = 3
    
    # Vector Database Configuration
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    retrieval_k: int = 3
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if not self.youtube_api_key:
            raise ValueError("YouTube API key is required")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Ensure data directory exists
        Path(self.data_directory).mkdir(exist_ok=True)
        
        # Set environment variables for libraries that expect them
        os.environ["OPENAI_API_KEY"] = self.openai_api_key


def load_credentials(credentials_path: str = "credentials.yml") -> Dict[str, Any]:
    """
    Load credentials from YAML file.
    
    Args:
        credentials_path: Path to credentials file
        
    Returns:
        Dictionary containing API keys and configuration
        
    Raises:
        FileNotFoundError: If credentials file doesn't exist
        ValueError: If credentials file is invalid
    """
    try:
        with open(credentials_path, 'r') as f:
            credentials = yaml.safe_load(f)
        
        if not credentials:
            raise ValueError("Credentials file is empty")
        
        required_keys = ['youtube', 'openai']
        missing_keys = [key for key in required_keys if key not in credentials]
        if missing_keys:
            raise ValueError(f"Missing required keys in credentials: {missing_keys}")
        
        return credentials
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Credentials file '{credentials_path}' not found. "
            "Please create a credentials.yml file with 'youtube' and 'openai' keys."
        )
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in credentials file: {e}")


def get_settings(credentials_path: str = "credentials.yml") -> Settings:
    """
    Get application settings with credentials loaded.
    
    Args:
        credentials_path: Path to credentials file
        
    Returns:
        Settings instance with loaded configuration
    """
    credentials = load_credentials(credentials_path)
    
    return Settings(
        youtube_api_key=credentials['youtube'],
        openai_api_key=credentials['openai']
    )


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_global_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings 