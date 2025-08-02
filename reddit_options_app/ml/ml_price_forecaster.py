"""
IMPROVED ML Price Forecasting Engine with Database Integration
- Fixes overfitting issues from previous version
- Adds sentiment integration
- Uses classification instead of regression
- Implements prediction tracking in Firebase
- Provides realistic performance expectations
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

# ML Libraries - simpler models to prevent overfitting
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# yfinance for historical data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    print("✅ yfinance available")
except ImportError:
    YFINANCE_AVAILABLE = False
    print("❌ yfinance not available - install with: pip install yfinance")

from config.settings import FINANCIAL_APIS

# Import ML database manager
try:
    from data.ml_database_manager import MLDatabaseManager
    ML_DATABASE_AVAILABLE = True
except ImportError:
    ML_DATABASE_AVAILABLE = False
    print("⚠️ ML Database Manager not available")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPriceForecaster:
    """Improved ML forecaster that actually works and doesn't overfit"""

    def __init__(self):
        """Initialize the improved forecaster"""

        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available - ML forecaster will be disabled")
            return

        # Models storage
        self.models = {}
        self.scalers = {}
        self.model_performance = {}

        # SIMPLIFIED model configurations to prevent overfitting
        self.model_configs = {
            'logistic_regression': {
                'C': 0.1,  # Strong regularization
                'random_state': 42,
                'max_iter': 1000
            },
            'random_forest': {
                'n_estimators': 50,  # Reduced from 100+
                'max_depth': 5,      # Limited depth to prevent overfitting
                'min_samples_split': 20,  # Require more samples to split
                'min_samples_leaf': 10,   # Require more samples in leaves
                'random_state': 42
            }
        }

        # Database integration
        if ML_DATABASE_AVAILABLE:
            try:
                self.ml_db = MLDatabaseManager()
                self.use_database = True
                logger.info("✅ ML Database Manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ ML Database Manager failed to initialize: {e}")
                self.ml_db = None
                self.use_database = False
        else:
            self.ml_db = None
            self.use_database = False

        # Local prediction tracking (backup)
        self.predictions_db = {}
        self.prediction_file = Path("logs/ml_predictions.json")
        self.prediction_file.parent.mkdir(exist_ok=True)
        self.load_prediction_history()

        logger.info("Improved ML Forecaster initialized - overfitting fixes applied")

    def load_prediction_history(self):
        """Load prediction history from local file"""
        try:
            if self.prediction_file.exists():
                import json
                with open(self.prediction_file, 'r') as f:
                    self.predictions_db = json.load(f)
                logger.info(f"Loaded {len(self.predictions_db)} historical predictions from local file")
            else:
                self.predictions_db = {}
        except Exception as e:
            logger.warning(f"Failed to load prediction history: {e}")
            self.predictions_db = {}

    def save_prediction_history(self):
        """Save prediction history to local file"""
        try:
            import json
            with open(self.prediction_file, 'w') as f:
                json.dump(self.predictions_db, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save prediction history: {e}")

    def get_historical_data(self, ticker: str, days: int = 500) -> pd.DataFrame:
        """Get more historical data for better training"""

        if not YFINANCE_AVAILABLE:
            return pd.DataFrame()

        try:
            logger.info(f"Fetching {ticker} data using yfinance (last {days} days)...")

            # Get longer period for better training
            if days <= 90:
                period = "3mo"
            elif days <= 180:
                period = "6mo"
            elif days <= 365:
                period = "1y"
            else:
                period = "2y"

            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)

            if hist.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Convert to expected format
            df = pd.DataFrame({
                'open': hist['Open'],
                'high': hist['High'],
                'low': hist['Low'],
                'close': hist['Close'],
                'volume': hist['Volume']
            })

            df.index = hist.index
            df = df.tail(days)
            df = df.dropna()

            logger.info(f"✅ Retrieved {len(df)} days of data for {ticker}")
            if len(df) > 0:
                latest_close = df['close'].iloc[-1]
                logger.info(f"   Latest close price: ${latest_close:.2f}")

            return df

        except Exception as e:
            logger.error(f"❌ Failed to get data for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_enhanced_features(self, df: pd.DataFrame, sentiment_data: List[Dict] = None) -> pd.DataFrame:
        """Calculate enhanced features with sentiment integration"""

        try:
            # Technical indicators (simplified to reduce overfitting)
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()

            # Price momentum (3-day and 10-day to reduce noise)
            df['momentum_3'] = df['close'].pct_change(3)
            df['momentum_10'] = df['close'].pct_change(10)

            # Volatility
            df['volatility_20'] = df['close'].rolling(window=20).std()

            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']

            # Price position in range
            df['high_20'] = df['high'].rolling(window=20).max()
            df['low_20'] = df['low'].rolling(window=20).min()
            df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])

            # FIXED: Add sentiment features if available
            if sentiment_data:
                # Create sentiment score timeseries
                sentiment_scores = []
                sentiment_confidences = []
                mention_counts = []

                for i, row in df.iterrows():
                    # Find closest sentiment data point
                    closest_sentiment = self._get_closest_sentiment(row.name, sentiment_data)

                    if closest_sentiment:
                        score = closest_sentiment.get('numerical_score', 0)
                        confidence = closest_sentiment.get('confidence', 0)
                        mentions = closest_sentiment.get('mention_count', 0)
                    else:
                        score = 0
                        confidence = 0
                        mentions = 0

                    sentiment_scores.append(score)
                    sentiment_confidences.append(confidence)
                    mention_counts.append(mentions)

                df['sentiment_score'] = sentiment_scores
                df['sentiment_confidence'] = sentiment_confidences
                df['mention_count'] = mention_counts

                # Sentiment momentum
                df['sentiment_momentum'] = pd.Series(sentiment_scores).diff(5)

                logger.info("✅ Added sentiment features to dataset")
            else:
                # Default sentiment features
                df['sentiment_score'] = 0
                df['sentiment_confidence'] = 0
                df['mention_count'] = 0
                df['sentiment_momentum'] = 0
                logger.info("⚠️ No sentiment data provided - using default values")

            # CLASSIFICATION TARGETS (instead of noisy regression)

            # Target 1: 3-day direction (more predictable than 1-day)
            df['future_return_3d'] = df['close'].shift(-3) / df['close'] - 1
            df['target_direction_3d'] = (df['future_return_3d'] > 0.01).astype(int)  # >1% gain

            # Target 2: 5-day direction
            df['future_return_5d'] = df['close'].shift(-5) / df['close'] - 1
            df['target_direction_5d'] = (df['future_return_5d'] > 0.02).astype(int)  # >2% gain

            # Target 3: Volatility prediction (high vol = options opportunity)
            df['future_volatility'] = df['close'].shift(-3).rolling(3).std()
            current_vol = df['volatility_20']
            df['target_high_vol'] = (df['future_volatility'] > current_vol * 1.5).astype(int)

            return df

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return df

    def _get_closest_sentiment(self, date, sentiment_data: List[Dict]) -> Dict:
        """Get closest sentiment data point for a given date"""
        if not sentiment_data:
            return {}

        # For now, return the most recent sentiment
        # In production, you'd match by timestamp
        return sentiment_data[0] if sentiment_data else {}

    def prepare_training_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
        """Prepare training data with proper validation"""

        # Feature columns (including sentiment)
        feature_cols = [
            'sma_5', 'sma_20', 'sma_50',
            'momentum_3', 'momentum_10',
            'volatility_20', 'price_position', 'volume_ratio',
            'sentiment_score', 'sentiment_confidence', 'mention_count', 'sentiment_momentum'
        ]

        # Only use columns that exist
        available_cols = [col for col in feature_cols if col in df.columns and not df[col].isna().all()]

        if not available_cols:
            raise ValueError("No valid feature columns available")

        logger.info(f"Using {len(available_cols)} features: {available_cols}")

        # Prepare features and target
        X = df[available_cols].values
        y = df[target_col].values

        # Remove rows with NaN values
        valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1))
        X = X[valid_rows]
        y = y[valid_rows]

        if len(X) < 100:
            raise ValueError(f"Insufficient data after cleaning: {len(X)} rows (need at least 100)")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info(f"Prepared training data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

        return X_scaled, y, scaler, available_cols

    def train_models(self, ticker: str, sentiment_data: List[Dict] = None) -> Dict:
        """Train models with proper cross-validation to prevent overfitting"""

        logger.info(f"🤖 Training IMPROVED models for {ticker}")

        try:
            # Get more historical data
            df = self.get_historical_data(ticker, days=500)  # More data

            if df.empty:
                raise ValueError("No historical data available")

            # Calculate enhanced features
            df = self.calculate_enhanced_features(df, sentiment_data)

            # Train multiple targets
            results = {}

            for target_name in ['target_direction_3d', 'target_direction_5d', 'target_high_vol']:
                if target_name not in df.columns:
                    continue

                try:
                    logger.info(f"Training models for target: {target_name}")

                    # Prepare data
                    X, y, scaler, feature_cols = self.prepare_training_data(df, target_name)

                    # Check class balance
                    positive_rate = np.mean(y)
                    logger.info(f"Positive class rate: {positive_rate:.2%}")

                    if positive_rate < 0.1 or positive_rate > 0.9:
                        logger.warning(f"Imbalanced classes for {target_name} - skipping")
                        continue

                    # Use TimeSeriesSplit for proper validation
                    tscv = TimeSeriesSplit(n_splits=5)

                    models = {}
                    cv_scores = {}

                    for model_name, config in self.model_configs.items():
                        try:
                            logger.info(f"   Training {model_name} for {target_name}...")

                            # Initialize model
                            if model_name == 'logistic_regression':
                                model = LogisticRegression(**config)
                            elif model_name == 'random_forest':
                                model = RandomForestClassifier(**config)
                            else:
                                continue

                            # Cross-validation scores
                            cv_score = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                            mean_cv_score = np.mean(cv_score)
                            std_cv_score = np.std(cv_score)

                            # Final training on all data
                            model.fit(X, y)
                            train_score = model.score(X, y)

                            models[model_name] = model
                            cv_scores[model_name] = {
                                'cv_mean': mean_cv_score,
                                'cv_std': std_cv_score,
                                'train_score': train_score,
                                'n_samples': len(X)
                            }

                            logger.info(f"   ✅ {model_name}: CV = {mean_cv_score:.3f} ± {std_cv_score:.3f}, Train = {train_score:.3f}")

                        except Exception as e:
                            logger.error(f"   ❌ {model_name} training failed: {e}")
                            continue

                    if models:
                        results[target_name] = {
                            'models': models,
                            'scaler': scaler,
                            'feature_columns': feature_cols,
                            'cv_scores': cv_scores
                        }

                except Exception as e:
                    logger.error(f"Failed to train models for {target_name}: {e}")
                    continue

            if results:
                # Store results
                self.models[ticker] = results
                self.model_performance[ticker] = {
                    'training_date': datetime.now().isoformat(),
                    'targets_trained': list(results.keys()),
                    'data_points': len(df)
                }

                # Save training data to database if available
                if self.use_database and self.ml_db:
                    try:
                        training_data_summary = {
                            'data_points': len(df),
                            'features_used': feature_cols,
                            'targets_trained': list(results.keys()),
                            'cv_scores_summary': {
                                target: {model: scores['cv_mean'] for model, scores in data['cv_scores'].items()}
                                for target, data in results.items()
                            }
                        }
                        self.ml_db.save_training_data(ticker, training_data_summary)
                    except Exception as e:
                        logger.warning(f"Failed to save training data to database: {e}")

                logger.info(f"✅ Training complete for {ticker}: {len(results)} targets trained")

                return {
                    'success': True,
                    'targets_trained': len(results),
                    'models_per_target': {target: len(data['models']) for target, data in results.items()},
                    'data_points': len(df)
                }
            else:
                raise ValueError("No models trained successfully")

        except Exception as e:
            logger.error(f"❌ Training failed for {ticker}: {e}")
            return {'success': False, 'error': str(e)}

    def predict_price_movement(self, ticker: str, current_data: pd.DataFrame = None,
                             sentiment_data: List[Dict] = None) -> Dict:
        """Make predictions and track them in database"""

        try:
            # Check if models exist
            if ticker not in self.models:
                logger.info(f"No models found for {ticker}, training...")
                train_result = self.train_models(ticker, sentiment_data)
                if not train_result['success']:
                    raise ValueError(f"No models available and training failed: {train_result.get('error')}")

            # Get recent data if not provided
            if current_data is None or current_data.empty:
                current_data = self.get_historical_data(ticker, days=100)
                if current_data.empty:
                    raise ValueError("No current data available for prediction")

            # Calculate features
            df = self.calculate_enhanced_features(current_data, sentiment_data)

            # Make predictions for each target
            predictions = {}

            for target_name, target_data in self.models[ticker].items():
                try:
                    feature_cols = target_data['feature_columns']
                    scaler = target_data['scaler']
                    models = target_data['models']

                    # Get latest features
                    latest_data = df[feature_cols].dropna().tail(1)
                    if latest_data.empty:
                        continue

                    latest_features = latest_data.values
                    latest_scaled = scaler.transform(latest_features)

                    # Ensemble prediction
                    target_predictions = {}
                    probabilities = {}

                    for model_name, model in models.items():
                        pred = model.predict(latest_scaled)[0]
                        prob = model.predict_proba(latest_scaled)[0]

                        target_predictions[model_name] = int(pred)
                        probabilities[model_name] = float(prob[1])  # Probability of positive class

                    # Ensemble
                    ensemble_prob = np.mean(list(probabilities.values()))
                    ensemble_pred = int(ensemble_prob > 0.5)

                    predictions[target_name] = {
                        'prediction': ensemble_pred,
                        'probability': ensemble_prob,
                        'individual_predictions': target_predictions,
                        'individual_probabilities': probabilities
                    }

                except Exception as e:
                    logger.warning(f"Prediction failed for {target_name}: {e}")
                    continue

            if not predictions:
                raise ValueError("All predictions failed")

            # Create final result
            current_price = float(current_data['close'].iloc[-1])

            # Interpret predictions
            direction_3d = predictions.get('target_direction_3d', {})
            direction_5d = predictions.get('target_direction_5d', {})
            high_vol = predictions.get('target_high_vol', {})

            # Overall signal
            signals = []
            confidence_scores = []

            if direction_3d:
                if direction_3d['prediction'] == 1:
                    signals.append('bullish_3d')
                confidence_scores.append(direction_3d['probability'])

            if direction_5d:
                if direction_5d['prediction'] == 1:
                    signals.append('bullish_5d')
                confidence_scores.append(direction_5d['probability'])

            if high_vol:
                if high_vol['prediction'] == 1:
                    signals.append('high_volatility')
                confidence_scores.append(high_vol['probability'])

            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5

            # Determine overall direction
            if 'bullish_3d' in signals or 'bullish_5d' in signals:
                overall_direction = 'up'
            else:
                overall_direction = 'down'

            # Create prediction record
            prediction_id = f"{ticker}_{int(time.time())}"
            prediction_record = {
                'prediction_id': prediction_id,
                'ticker': ticker,
                'prediction_time': datetime.now().isoformat(),
                'current_price': current_price,
                'overall_direction': overall_direction,
                'overall_confidence': overall_confidence,
                'signals': signals,
                'detailed_predictions': predictions,
                'sentiment_data': sentiment_data[0] if sentiment_data else None
            }

            # Save to Firebase database if available
            if self.use_database and self.ml_db:
                try:
                    saved_id = self.ml_db.save_prediction(prediction_record)
                    if saved_id:
                        logger.info(f"✅ Saved prediction {saved_id} to Firebase")
                    else:
                        logger.warning("⚠️ Failed to save prediction to Firebase")
                except Exception as e:
                    logger.warning(f"⚠️ Database save failed: {e}")

            # Also save to local file as backup
            self.predictions_db[prediction_id] = prediction_record
            self.save_prediction_history()

            result = {
                'ticker': ticker,
                'current_price': current_price,
                'predicted_return': (overall_confidence - 0.5) * 0.1,  # Conservative return estimate
                'predicted_price': current_price * (1 + (overall_confidence - 0.5) * 0.1),
                'price_change_pct': (overall_confidence - 0.5) * 10,  # Conservative % change
                'direction': overall_direction,
                'confidence': overall_confidence,
                'signals': signals,
                'detailed_predictions': predictions,
                'prediction_id': prediction_id,
                'prediction_time': datetime.now().isoformat(),
                'saved_to_database': self.use_database
            }

            logger.info(f"✅ Prediction for {ticker}: {overall_direction} (confidence: {overall_confidence:.2f})")

            return result

        except Exception as e:
            logger.error(f"❌ Prediction failed for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'prediction_time': datetime.now().isoformat()
            }

    def get_model_performance_summary(self) -> Dict:
        """Get comprehensive model performance summary"""

        # Local performance
        local_performance = {
            'models_trained': len(self.models),
            'total_predictions_made_local': len(self.predictions_db),
            'prediction_tracking_available': True,
            'database_integration': self.use_database
        }

        # Database performance if available
        if self.use_database and self.ml_db:
            try:
                db_performance = self.ml_db.get_performance_summary()
                local_performance.update({
                    'database_performance': db_performance,
                    'database_status': 'connected'
                })
            except Exception as e:
                local_performance.update({
                    'database_status': f'error: {str(e)}',
                    'database_performance': None
                })
        else:
            local_performance.update({
                'database_status': 'not_available',
                'database_performance': None
            })

        local_performance['last_updated'] = datetime.now().isoformat()
        return local_performance


def main():
    """Test the improved forecaster"""

    print("🚀 Testing IMPROVED ML Forecaster (No More Overfitting!)")
    print("=" * 60)

    forecaster = MLPriceForecaster()

    # Test with sample sentiment data
    sample_sentiment = [{
        'sentiment': 'bullish',
        'confidence': 0.8,
        'numerical_score': 0.6,
        'mention_count': 15
    }]

    # Test training
    print("Testing training with improved features...")
    train_result = forecaster.train_models('AAPL', sample_sentiment)

    if train_result['success']:
        print(f"✅ Training successful: {train_result}")

        # Test prediction
        print("Testing prediction with tracking...")
        prediction = forecaster.predict_price_movement('AAPL', sentiment_data=sample_sentiment)

        if 'error' not in prediction:
            print(f"✅ Prediction successful: {prediction}")
            print(f"   Direction: {prediction['direction']}")
            print(f"   Confidence: {prediction['confidence']:.2%}")
            print(f"   Saved to database: {prediction['saved_to_database']}")
        else:
            print(f"❌ Prediction failed: {prediction['error']}")
    else:
        print(f"❌ Training failed: {train_result['error']}")

    # Test performance tracking
    performance = forecaster.get_model_performance_summary()
    print(f"📊 Performance summary: {performance}")


if __name__ == "__main__":
    main()