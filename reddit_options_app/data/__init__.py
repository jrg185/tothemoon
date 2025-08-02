"""
Data collection and management module for Reddit Options App

This module provides classes and utilities for:
- Reddit data scraping from r/wallstreetbets
- Firebase/Firestore database operations
- LLM-powered stock ticker extraction from text
- ML prediction tracking and performance monitoring
- Data models for posts and comments
"""

from .reddit_scraper import (
    RedditScraper,
    RedditPost,
    RedditComment
)
from .firebase_manager import FirebaseManager
from .llm_ticker_extractor import EnhancedTickerExtractor, LLMTickerExtractor

# ML Database Manager (with graceful fallback)
try:
    from .ml_database_manager import MLDatabaseManager
    ML_DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML Database Manager not available: {e}")
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

    # ML Database (if available)
    'MLDatabaseManager',
    'ML_DATABASE_AVAILABLE'
]

# Module version
__version__ = '2.1.0'

# Module metadata
__author__ = 'Reddit Options App'
__description__ = 'LLM-powered data collection and management with ML prediction tracking for options trading sentiment analysis'