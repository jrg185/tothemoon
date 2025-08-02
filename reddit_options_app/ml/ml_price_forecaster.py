"""
ML Price Forecasting Engine - FIXED with yfinance
Replace your existing ml_price_forecaster.py with this version
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import logging
import warnings
from typing import Dict, List, Tuple, Optional

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available")

# yfinance for historical data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    print("✅ yfinance available")
except ImportError:
    YFINANCE_AVAILABLE = False
    print("❌ yfinance not available - install with: pip install yfinance")

from config.settings import FINANCIAL_APIS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPriceForecaster:
    """ML forecaster using yfinance instead of Alpha Vantage"""

    def __init__(self):
        """Initialize the ML forecaster"""

        # Check data availability
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available - ML forecaster will be limited")

        # Models storage
        self.models = {}
        self.scalers = {}
        self.model_performance = {}

        # Model configurations (simplified for reliability)
        self.model_configs = {
            'random_forest': {
                'n_estimators': 50,
                'max_depth': 8,
                'min_samples_split': 5,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 50,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }

        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'n_estimators': 50,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbosity': 0  # Reduce XGBoost logging
            }

        logger.info("ML Price Forecaster initialized with yfinance")

    def get_historical_data(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Get historical data using yfinance (FIXED VERSION)"""

        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available - cannot get historical data")
            return pd.DataFrame()

        try:
            logger.info(f"Fetching {ticker} data using yfinance...")

            # Calculate period based on days requested
            if days <= 30:
                period = "1mo"
            elif days <= 90:
                period = "3mo"
            elif days <= 180:
                period = "6mo"
            elif days <= 365:
                period = "1y"
            else:
                period = "2y"

            # Create ticker object and fetch data
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)

            if hist.empty:
                logger.warning(f"No data returned for {ticker} from yfinance")
                return pd.DataFrame()

            # Convert to expected format (matching your original structure)
            df = pd.DataFrame({
                'open': hist['Open'],
                'high': hist['High'],
                'low': hist['Low'],
                'close': hist['Close'],
                'volume': hist['Volume']
            })

            # Keep the datetime index
            df.index = hist.index

            # Limit to requested number of days
            df = df.tail(days)

            # Remove any rows with NaN values
            df = df.dropna()

            logger.info(f"✅ Retrieved {len(df)} days of data for {ticker} from yfinance")

            if len(df) > 0:
                latest_close = df['close'].iloc[-1]
                logger.info(f"   Latest close price: ${latest_close:.2f}")

            return df

        except Exception as e:
            logger.error(f"❌ yfinance failed for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for ML features"""

        try:
            # Simple Moving Averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()

            # Price momentum
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)

            # Volatility
            df['volatility_10'] = df['close'].rolling(window=10).std()
            df['volatility_20'] = df['close'].rolling(window=20).std()

            # Price position in recent range
            df['high_20'] = df['high'].rolling(window=20).max()
            df['low_20'] = df['low'].rolling(window=20).min()
            df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])

            # Volume indicators
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']

            # RSI (simplified)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Lag features
            df['close_lag_1'] = df['close'].shift(1)
            df['close_lag_5'] = df['close'].shift(5)
            df['volume_lag_1'] = df['volume'].shift(1)

            # Target variable (next day return)
            df['target_return'] = df['close'].shift(-1) / df['close'] - 1

            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
        """Prepare data for ML training"""

        # Feature columns (exclude target and non-feature columns)
        feature_cols = [
            'sma_5', 'sma_10', 'sma_20',
            'momentum_5', 'momentum_10',
            'volatility_10', 'volatility_20',
            'price_position', 'volume_ratio', 'rsi',
            'close_lag_1', 'close_lag_5', 'volume_lag_1'
        ]

        # Only use columns that exist and have data
        available_cols = [col for col in feature_cols if col in df.columns and not df[col].isna().all()]

        if not available_cols:
            raise ValueError("No valid feature columns available")

        logger.info(f"Using {len(available_cols)} features: {available_cols}")

        # Prepare features and target
        X = df[available_cols].values
        y = df['target_return'].values

        # Remove rows with NaN values
        valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_rows]
        y = y[valid_rows]

        if len(X) < 30:
            raise ValueError(f"Insufficient data after cleaning: {len(X)} rows")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info(f"Prepared training data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

        return X_scaled, y, scaler, available_cols

    def train_models(self, ticker: str, sentiment_data: List[Dict] = None,
                     market_data: Dict = None) -> Dict:
        """Train ML models for a ticker"""

        logger.info(f"🤖 Training models for {ticker}")

        try:
            # Get historical data using yfinance
            df = self.get_historical_data(ticker, days=200)

            if df.empty:
                raise ValueError("No historical data available")

            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)

            # Prepare training data
            X, y, scaler, feature_cols = self.prepare_training_data(df)

            # Train models
            models = {}
            performance = {}

            # Split data (80% train, 20% test)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            for model_name, config in self.model_configs.items():
                try:
                    logger.info(f"   Training {model_name}...")

                    # Initialize model
                    if model_name == 'random_forest':
                        model = RandomForestRegressor(**config)
                    elif model_name == 'gradient_boosting':
                        model = GradientBoostingRegressor(**config)
                    elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                        model = xgb.XGBRegressor(**config)
                    else:
                        continue

                    # Train model
                    model.fit(X_train, y_train)

                    # Evaluate
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test) if len(X_test) > 0 else train_score

                    models[model_name] = model
                    performance[model_name] = {
                        'train_r2': train_score,
                        'test_r2': test_score,
                        'data_points': len(X)
                    }

                    logger.info(f"   ✅ {model_name}: Train R² = {train_score:.3f}, Test R² = {test_score:.3f}")

                except Exception as e:
                    logger.error(f"   ❌ {model_name} training failed: {e}")

            if not models:
                raise ValueError("No models trained successfully")

            # Store everything
            self.models[ticker] = {
                'models': models,
                'scaler': scaler,
                'feature_columns': feature_cols
            }

            self.model_performance[ticker] = {
                'performance': performance,
                'training_date': datetime.now().isoformat(),
                'data_points': len(df)
            }

            logger.info(f"✅ Training complete for {ticker}: {len(models)} models trained")

            return {
                'success': True,
                'models_trained': len(models),
                'performance': performance,
                'data_points': len(df)
            }

        except Exception as e:
            logger.error(f"❌ Training failed for {ticker}: {e}")
            return {'success': False, 'error': str(e)}

    def predict_price_movement(self, ticker: str, current_data: pd.DataFrame = None,
                             sentiment_data: List[Dict] = None) -> Dict:
        """Predict price movement for a ticker"""

        try:
            # Check if models exist
            if ticker not in self.models:
                # Try to train models first
                logger.info(f"No models found for {ticker}, training...")
                train_result = self.train_models(ticker, sentiment_data)
                if not train_result['success']:
                    raise ValueError(f"No models available and training failed: {train_result.get('error')}")

            # Get recent data if not provided
            if current_data is None or current_data.empty:
                current_data = self.get_historical_data(ticker, days=50)
                if current_data.empty:
                    raise ValueError("No current data available for prediction")

            # Calculate indicators
            df = self.calculate_technical_indicators(current_data)

            # Get latest features
            feature_cols = self.models[ticker]['feature_columns']

            # Check if we have the required features
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")

            # Get the latest row with all features
            latest_data = df[feature_cols].dropna().tail(1)
            if latest_data.empty:
                raise ValueError("No valid feature data for prediction")

            latest_features = latest_data.values

            # Scale features
            scaler = self.models[ticker]['scaler']
            latest_scaled = scaler.transform(latest_features)

            # Make predictions with all models
            models = self.models[ticker]['models']
            predictions = {}

            for model_name, model in models.items():
                try:
                    pred = model.predict(latest_scaled)[0]
                    predictions[model_name] = float(pred)
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")

            if not predictions:
                raise ValueError("All model predictions failed")

            # Ensemble prediction (simple average)
            ensemble_pred = np.mean(list(predictions.values()))

            # Current price
            current_price = float(current_data['close'].iloc[-1])

            # Predicted price
            predicted_price = current_price * (1 + ensemble_pred)

            # Calculate confidence based on model agreement
            pred_values = list(predictions.values())
            pred_std = np.std(pred_values)
            confidence = max(0.3, min(0.9, 1 - (pred_std * 5)))

            result = {
                'ticker': ticker,
                'current_price': current_price,
                'predicted_return': float(ensemble_pred),
                'predicted_price': float(predicted_price),
                'price_change': float(predicted_price - current_price),
                'price_change_pct': float((predicted_price - current_price) / current_price * 100),
                'confidence': float(confidence),
                'individual_predictions': predictions,
                'prediction_date': datetime.now().isoformat(),
                'direction': 'up' if ensemble_pred > 0 else 'down',
                'magnitude': 'high' if abs(ensemble_pred) > 0.02 else 'moderate' if abs(ensemble_pred) > 0.01 else 'low',
                'models_used': list(predictions.keys())
            }

            logger.info(f"✅ Prediction for {ticker}: {ensemble_pred*100:.2f}% ({result['direction']})")

            return result

        except Exception as e:
            logger.error(f"❌ Prediction failed for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'prediction_date': datetime.now().isoformat()
            }

    def test_functionality(self, ticker: str = 'AAPL') -> Dict:
        """Test the forecaster functionality"""

        logger.info(f"🧪 Testing ML forecaster with {ticker}")

        results = {
            'ticker': ticker,
            'yfinance_available': YFINANCE_AVAILABLE,
            'tests': {}
        }

        # Test 1: Data retrieval
        try:
            df = self.get_historical_data(ticker, days=50)
            results['tests']['data_retrieval'] = {
                'success': not df.empty,
                'data_points': len(df),
                'latest_price': float(df['close'].iloc[-1]) if not df.empty else None
            }
        except Exception as e:
            results['tests']['data_retrieval'] = {'success': False, 'error': str(e)}

        # Test 2: Model training (only if data retrieval worked)
        if results['tests']['data_retrieval']['success']:
            try:
                train_result = self.train_models(ticker)
                results['tests']['model_training'] = {
                    'success': train_result['success'],
                    'models_trained': train_result.get('models_trained', 0),
                    'error': train_result.get('error')
                }
            except Exception as e:
                results['tests']['model_training'] = {'success': False, 'error': str(e)}

            # Test 3: Prediction (only if training worked)
            if results['tests']['model_training']['success']:
                try:
                    prediction = self.predict_price_movement(ticker)
                    results['tests']['prediction'] = {
                        'success': 'error' not in prediction,
                        'predicted_change': prediction.get('price_change_pct'),
                        'confidence': prediction.get('confidence'),
                        'error': prediction.get('error')
                    }
                except Exception as e:
                    results['tests']['prediction'] = {'success': False, 'error': str(e)}

        return results


def main():
    """Test the fixed forecaster"""

    print("🚀 Testing Fixed ML Price Forecaster with yfinance")
    print("=" * 60)

    forecaster = MLPriceForecaster()

    # Run comprehensive test
    test_results = forecaster.test_functionality('AAPL')

    print(f"\n📊 Test Results for {test_results['ticker']}:")
    print(f"yfinance available: {test_results['yfinance_available']}")

    for test_name, result in test_results['tests'].items():
        status = "✅" if result['success'] else "❌"
        print(f"\n{status} {test_name.replace('_', ' ').title()}:")

        if result['success']:
            for key, value in result.items():
                if key != 'success' and value is not None:
                    print(f"   {key}: {value}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)

    if all(test['success'] for test in test_results['tests'].values()):
        print("🎉 ALL TESTS PASSED! ML Forecaster is working with yfinance")
    else:
        print("⚠️  Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()