"""
Data collection and management module for Reddit Options App

This module provides classes and utilities for:
- Reddit data scraping from r/wallstreetbets
- Firebase/Firestore database operations
- LLM-powered stock ticker extraction from text
- Data models for posts and comments
"""

from .reddit_scraper import (
    RedditScraper,
    RedditPost,
    RedditComment
)
from .firebase_manager import FirebaseManager
from .llm_ticker_extractor import EnhancedTickerExtractor, LLMTickerExtractor

__all__ = [
    # Main classes
    'RedditScraper',
    'FirebaseManager',
    'EnhancedTickerExtractor',
    'LLMTickerExtractor',

    # Data models
    'RedditPost',
    'RedditComment'
]

# Module version
__version__ = '2.0.0'

# Module metadata
__author__ = 'Reddit Options App'
__description__ = 'LLM-powered data collection and management for options trading sentiment analysis'