"""
Streamlit runner script for YouTube Agent System.
Maintains same launch experience as MVP while using modular backend.
uv run streamlit run ui/streamlit/streamlit_runner.py
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import and run the dashboard
from ui.streamlit.dashboard import main

if __name__ == "__main__":
    # Set working directory to project root for file access
    os.chdir(project_root)
    
    # Run the main application
    main() 