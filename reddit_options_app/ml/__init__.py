"""
Machine learning models and predictions for advanced trading analytics
"""

try:
    from .enhanced_trading_analyst import EnhancedTradingAnalyst
    from .ml_price_forecaster import MLPriceForecaster

    __all__ = ['EnhancedTradingAnalyst', 'MLPriceForecaster']

except ImportError as e:
    # Graceful fallback if dependencies aren't installed
    print(f"Warning: ML components not fully available: {e}")
    __all__ = []

# Version info
__version__ = '1.0.0'