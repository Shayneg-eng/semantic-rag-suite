import os
"""
Configuration settings for the Document Analysis Agent
"""

# API Configuration
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Reading Configuration  
PREVIEW_LENGTH = 2000
MAX_SNIPPET_LENGTH = 100  # Reduced from 200

# Search Configuration
FUZZY_THRESHOLD = 0.7

# Agent Configuration
MAX_ITERATIONS = 20
HISTORY_LIMIT = 10

# File Paths
DOCUMENTS_FOLDER = "./data"
FOCUS_DOCS_FILE = "./data/focus_docs.json"
