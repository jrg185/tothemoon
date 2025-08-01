"""
Pytest configuration and fixtures
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta

@pytest.fixture
def sample_reddit_data():
    """Sample Reddit post data for testing"""
    return {
        'id': 'test123',
        'title': 'TSLA to the moon! 🚀',
        'selftext': 'Tesla earnings looking great, buying calls',
        'score': 150,
        'upvote_ratio': 0.85,
        'num_comments': 25,
        'created_utc': datetime.now().timestamp(),
        'author': 'test_user'
    }

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'Date': dates,
        'Open': [100 + i for i in range(30)],
        'High': [105 + i for i in range(30)],
        'Low': [95 + i for i in range(30)],
        'Close': [102 + i for i in range(30)],
        'Volume': [1000000 + i*10000 for i in range(30)]
    })