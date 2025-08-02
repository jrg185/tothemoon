"""
Machine learning models and predictions for advanced trading analytics
"""

# Import ML components with proper error handling
ML_COMPONENTS_AVAILABLE = True
MISSING_COMPONENTS = []

try:
    from .enhanced_trading_analyst import EnhancedTradingAnalyst
    ENHANCED_ANALYST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: EnhancedTradingAnalyst not available: {e}")
    EnhancedTradingAnalyst = None
    ENHANCED_ANALYST_AVAILABLE = False
    MISSING_COMPONENTS.append('EnhancedTradingAnalyst')

try:
    from .ml_price_forecaster import MLPriceForecaster
    ML_FORECASTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MLPriceForecaster not available: {e}")
    MLPriceForecaster = None
    ML_FORECASTER_AVAILABLE = False
    MISSING_COMPONENTS.append('MLPriceForecaster')

# Import ML Database Manager from data module
try:
    from data.ml_database_manager import MLDatabaseManager
    ML_DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MLDatabaseManager not available: {e}")
    MLDatabaseManager = None
    ML_DATABASE_AVAILABLE = False
    MISSING_COMPONENTS.append('MLDatabaseManager')

# Determine overall ML availability
ML_COMPONENTS_AVAILABLE = ENHANCED_ANALYST_AVAILABLE and ML_FORECASTER_AVAILABLE and ML_DATABASE_AVAILABLE

# Export available components
__all__ = []

if ENHANCED_ANALYST_AVAILABLE:
    __all__.append('EnhancedTradingAnalyst')

if ML_FORECASTER_AVAILABLE:
    __all__.append('MLPriceForecaster')

if ML_DATABASE_AVAILABLE:
    __all__.append('MLDatabaseManager')

# Add availability flags for other modules to check
__all__.extend([
    'ML_COMPONENTS_AVAILABLE',
    'ENHANCED_ANALYST_AVAILABLE',
    'ML_FORECASTER_AVAILABLE',
    'ML_DATABASE_AVAILABLE',
    'MISSING_COMPONENTS'
])

# Version info
__version__ = '2.0.0'

# Status summary
if ML_COMPONENTS_AVAILABLE:
    print("✅ All ML components available")
else:
    print(f"⚠️ Some ML components missing: {MISSING_COMPONENTS}")
    available_components = [comp for comp in __all__ if not comp.startswith('ML_') and not comp == 'MISSING_COMPONENTS']
    print("📝 Available components:", available_components)