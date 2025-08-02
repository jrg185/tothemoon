"""
Data collection and management module for Reddit Options App
"""

from .reddit_scraper import (
    RedditScraper,
    RedditPost,
    RedditComment
)
from .firebase_manager import FirebaseManager
from .llm_ticker_extractor import EnhancedTickerExtractor, LLMTickerExtractor

# ML Database Manager (with proper error handling)
try:
    from .ml_database_manager import MLDatabaseManager
    ML_DATABASE_AVAILABLE = True
    print("✅ MLDatabaseManager imported successfully")
except ImportError as e:
    print(f"⚠️ MLDatabaseManager not available: {e}")
    MLDatabaseManager = None
    ML_DATABASE_AVAILABLE = False

__all__ = [
    # Main classes
    'RedditScraper',
    'FirebaseManager',
    'EnhancedTickerExtractor',
    'LLMTickerExtractor',

    # Data models
    'RedditPost',
    'RedditComment',

    # ML Database (with availability flag)
    'MLDatabaseManager',
    'ML_DATABASE_AVAILABLE'
]

# Module version
__version__ = '2.1.0'