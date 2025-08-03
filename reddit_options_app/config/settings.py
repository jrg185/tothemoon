"""
Application configuration settings
"""
import os
import json
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# Configuration loader that works with both Streamlit and standalone scripts
def get_secret(key, default=None):
    """Get configuration value from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for dashboard)
        import streamlit as st
        return st.secrets.get(key, default)
    except (ImportError, FileNotFoundError):
        # Fallback to environment variables (for standalone scripts)
        return os.getenv(key, default)

# Reddit API Configuration
REDDIT_CONFIG = {
    'client_id': get_secret('REDDIT_CLIENT_ID'),
    'client_secret': get_secret('REDDIT_CLIENT_SECRET'),
    'user_agent': get_secret('REDDIT_USER_AGENT'),
    'subreddit': 'wallstreetbets',
    'posts_limit': 100,
    'comments_limit': 50
}

# Financial APIs
FINANCIAL_APIS = {
    'alpha_vantage': get_secret('ALPHA_VANTAGE_API_KEY'),
    'finnhub': get_secret('FINNHUB_API_KEY')
}

# LLM Configuration
LLM_CONFIG = {
    'openai': {
        'api_key': get_secret('OPENAI_API_KEY'),
        'model': 'gpt-4-turbo',  # More capable model for complex reasoning
        'max_tokens': 1000,
        'temperature': 0.3
    },
    'xai': {
        'api_key': get_secret('XAI_API_KEY'),
        'base_url': 'https://api.x.ai/v1',
        'model': 'grok-3-mini',  # Updated to grok-3-mini
        'max_tokens': 1000,
        'temperature': 0.3
    },
    'default_provider': 'xai'  # Changed from 'openai' to 'xai'
}

# Firebase Configuration
FIREBASE_CONFIG = {
    'project_id': get_secret('FIREBASE_PROJECT_ID'),
    'credentials': {
        'type': 'service_account',
        'project_id': get_secret('FIREBASE_PROJECT_ID'),
        'private_key_id': get_secret('FIREBASE_PRIVATE_KEY_ID'),
        'private_key': get_secret('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
        'client_email': get_secret('FIREBASE_CLIENT_EMAIL'),
        'client_id': get_secret('FIREBASE_CLIENT_ID'),
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
        'client_x509_cert_url': get_secret('FIREBASE_CLIENT_CERT_URL')
    },
    'collections': {
        'reddit_posts': 'reddit_posts',
        'sentiment_data': 'sentiment_analysis',
        'stock_data': 'stock_data',
        'options_strategies': 'options_strategies',
        'predictions': 'ml_predictions'
    }
}

# Application Settings
APP_CONFIG = {
    'debug': str(get_secret('DEBUG', 'False')).lower() == 'true',
    'log_level': get_secret('LOG_LEVEL', 'INFO'),
    'streamlit_port': int(get_secret('STREAMLIT_PORT', '8501'))
}

# Sentiment Analysis Settings
SENTIMENT_CONFIG = {
    'models': ['vader', 'textblob', 'finbert'],
    'min_score_threshold': 0.1,
    'confidence_threshold': 0.7,
    'use_llm_analysis': True,  # Use LLM for complex sentiment
    'llm_batch_size': 10
}

# Options Trading Settings
OPTIONS_CONFIG = {
    'max_dte': 45,  # Days to expiration
    'min_volume': 100,
    'max_risk_per_trade': 0.02,  # 2% of portfolio
    'strategies': ['long_call', 'long_put', 'call_spread', 'put_spread'],
    'min_probability': 0.6  # Minimum win probability
}

# Validation
def validate_config():
    """Validate that required configuration is present"""
    required_vars = [
        'REDDIT_CLIENT_ID',
        'REDDIT_CLIENT_SECRET',
        'REDDIT_USER_AGENT',
        'FIREBASE_PROJECT_ID',
        'OPENAI_API_KEY'
    ]

    missing_vars = [var for var in required_vars if not get_secret(var)]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    # Validate Firebase credentials
    firebase_required = [
        'FIREBASE_PROJECT_ID',
        'FIREBASE_PRIVATE_KEY',
        'FIREBASE_CLIENT_EMAIL'
    ]

    firebase_missing = [var for var in firebase_required if not get_secret(var)]
    if firebase_missing:
        raise ValueError(f"Missing Firebase credentials: {firebase_missing}")

    return True

def get_firebase_credentials_dict():
    """Get Firebase credentials as a dictionary for initialization"""
    return FIREBASE_CONFIG['credentials']

if __name__ == "__main__":
    validate_config()
    print("✅ Configuration validated successfully!")
    print(f"📊 Firebase Project: {FIREBASE_CONFIG['project_id']}")
    print(f"🤖 Primary LLM: {LLM_CONFIG['default_provider']}")