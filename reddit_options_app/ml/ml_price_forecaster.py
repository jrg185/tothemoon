"""
ML Price Forecasting Engine
Advanced machine learning models for stock price prediction using multiple data sources
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import requests
import time
import logging
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
import json

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
from scipy import stats

# Deep Learning (if torch is available)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, LSTM models will be disabled")

from config.settings import FINANCIAL_APIS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for price prediction"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.fitted = False

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""

        # Price-based indicators
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std_dev = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']

        # Volatility indicators
        df['volatility_10'] = df['close'].rolling(window=10).std()
        df['volatility_20'] = df['close'].rolling(window=20).std()

        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']
            df['price_volume'] = df['close'] * df['volume']

        # Price action features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['range_pct'] = (df['high'] - df['low']) / df['close']

        # Support/Resistance levels
        df['resistance_20'] = df['high'].rolling(window=20).max()
        df['support_20'] = df['low'].rolling(window=20).min()
        df['distance_to_resistance'] = (df['resistance_20'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support_20']) / df['close']

        return df

    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged features for time series modeling"""

        for lag in lags:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
            df[f'return_lag_{lag}'] = df['close'].pct_change().shift(lag)

        return df

    def add_market_features(self, df: pd.DataFrame, market_data: Dict) -> pd.DataFrame:
        """Add market-wide features"""

        # Time-based features
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        df['quarter'] = pd.to_datetime(df.index).quarter

        # Market regime features (if market data available)
        if market_data:
            # VIX levels, market trends, etc.
            df['market_trend'] = market_data.get('market_trend', 0)
            df['vix_level'] = market_data.get('vix', 20)
            df['sector_performance'] = market_data.get('sector_performance', 0)

        return df

    def add_sentiment_features(self, df: pd.DataFrame, sentiment_data: List[Dict]) -> pd.DataFrame:
        """Add sentiment-based features"""

        # Convert sentiment data to DataFrame
        if sentiment_data:
            sentiment_df = pd.DataFrame(sentiment_data)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp']).dt.date

            # Aggregate daily sentiment
            daily_sentiment = sentiment_df.groupby('date').agg({
                'numerical_score': ['mean', 'std', 'count'],
                'confidence': 'mean',
                'mention_count': 'sum'
            }).reset_index()

            # Flatten column names
            daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'sentiment_count',
                                       'confidence_mean', 'mention_count_total']

            # Add sentiment change rate
            daily_sentiment['sentiment_change'] = daily_sentiment['sentiment_mean'].diff()
            daily_sentiment['mention_momentum'] = daily_sentiment['mention_count_total'].pct_change()

            # Merge with price data
            df_reset = df.reset_index()
            df_reset['date'] = pd.to_datetime(df_reset['date']).dt.date

            merged = pd.merge(df_reset, daily_sentiment, on='date', how='left')
            merged = merged.set_index('date')

            # Fill missing sentiment data
            sentiment_cols = ['sentiment_mean', 'sentiment_std', 'sentiment_count',
                              'confidence_mean', 'mention_count_total', 'sentiment_change', 'mention_momentum']
            for col in sentiment_cols:
                if col in merged.columns:
                    df[col] = merged[col].fillna(0)

        return df

    def engineer_features(self, df: pd.DataFrame, sentiment_data: List[Dict] = None,
                          market_data: Dict = None) -> pd.DataFrame:
        """Complete feature engineering pipeline"""

        logger.info("Starting feature engineering...")

        # Technical indicators
        df = self.calculate_technical_indicators(df)

        # Lag features
        df = self.add_lag_features(df)

        # Market features
        df = self.add_market_features(df, market_data or {})

        # Sentiment features
        if sentiment_data:
            df = self.add_sentiment_features(df, sentiment_data)

        # Target variable (next day return)
        df['target_return'] = df['close'].shift(-1) / df['close'] - 1
        df['target_price'] = df['close'].shift(-1)

        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)

        logger.info(f"Feature engineering complete: {initial_rows} -> {final_rows} rows")
        logger.info(f"Features created: {len(df.columns)} total columns")

        return df


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, output_size: int = 1):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.linear(output)

        return output


class MLPriceForecaster:
    """Advanced ML-based price forecasting system"""

    def __init__(self):
        """Initialize the ML forecaster"""

        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.model_performance = {}

        # Financial APIs
        self.finnhub_key = FINANCIAL_APIS.get('finnhub')
        self.alpha_vantage_key = FINANCIAL_APIS.get('alpha_vantage')

        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }

        logger.info("ML Price Forecaster initialized")

    def get_historical_data(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Get historical price data"""

        try:
            if not self.alpha_vantage_key:
                raise ValueError("Alpha Vantage API key not configured")

            # Get daily data
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={self.alpha_vantage_key}"
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                raise ValueError(f"API request failed: {response.status_code}")

            data = response.json()

            if 'Time Series (Daily)' not in data:
                raise ValueError(f"No data returned for {ticker}")

            time_series = data['Time Series (Daily)']

            # Convert to DataFrame
            df_data = []
            for date, values in time_series.items():
                df_data.append({
                    'date': pd.to_datetime(date),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': float(values['5. volume'])
                })

            df = pd.DataFrame(df_data)
            df = df.sort_values('date').set_index('date')

            # Limit to requested days
            df = df.tail(days)

            logger.info(f"Retrieved {len(df)} days of data for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {e}")
            return pd.DataFrame()

    def prepare_training_data(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""

        # Select feature columns (exclude target columns and date)
        feature_cols = [col for col in df.columns if not col.startswith('target_') and col != 'date']

        X = df[feature_cols].values
        y = df['target_return'].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # For LSTM, create sequences
        X_sequences = []
        y_sequences = []

        for i in range(sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i - sequence_length:i])
            y_sequences.append(y[i])

        return np.array(X_sequences), np.array(y_sequences), scaler

    def train_traditional_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train traditional ML models"""

        models = {}
        performance = {}

        # Reshape X for traditional models (use last observation in sequence)
        X_2d = X[:, -1, :] if len(X.shape) == 3 else X

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)

        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Training {model_name}...")

                if model_name == 'random_forest':
                    model = RandomForestRegressor(**config)
                elif model_name == 'xgboost':
                    model = xgb.XGBRegressor(**config)
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingRegressor(**config)

                # Cross-validation
                cv_scores = cross_val_score(model, X_2d, y, cv=tscv,
                                            scoring='neg_mean_squared_error')

                # Train on full dataset
                model.fit(X_2d, y)

                # Store model and performance
                models[model_name] = model
                performance[model_name] = {
                    'cv_mse': -cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_score': model.score(X_2d, y)
                }

                logger.info(f"{model_name} - CV MSE: {-cv_scores.mean():.6f} (+/- {cv_scores.std() * 2:.6f})")

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")

        return models, performance

    def train_lstm_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[LSTMModel], Dict]:
        """Train LSTM model"""

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping LSTM training")
            return None, {}

        try:
            logger.info("Training LSTM model...")

            # Split data
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

            # Initialize model
            input_size = X.shape[2]
            model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2)

            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            model.train()
            for epoch in range(50):
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if epoch % 10 == 0:
                    logger.info(f"LSTM Epoch {epoch}, Loss: {total_loss / len(train_loader):.6f}")

            # Evaluate
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                test_pred = model(X_test_tensor)

                train_mse = criterion(train_pred, y_train_tensor).item()
                test_mse = criterion(test_pred, y_test_tensor).item()

            performance = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': r2_score(y_train, train_pred.numpy().flatten()),
                'test_r2': r2_score(y_test, test_pred.numpy().flatten())
            }

            logger.info(f"LSTM - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")

            return model, performance

        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return None, {}

    def train_models(self, ticker: str, sentiment_data: List[Dict] = None,
                     market_data: Dict = None) -> Dict:
        """Train all models for a ticker"""

        logger.info(f"Training models for {ticker}")

        try:
            # Get historical data
            df = self.get_historical_data(ticker, days=500)
            if df.empty:
                raise ValueError("No historical data available")

            # Feature engineering
            df = self.feature_engineer.engineer_features(df, sentiment_data, market_data)

            if len(df) < 50:
                raise ValueError("Insufficient data after feature engineering")

            # Prepare training data
            X, y, scaler = self.prepare_training_data(df)

            # Train traditional models
            traditional_models, traditional_performance = self.train_traditional_models(X, y)

            # Train LSTM model
            lstm_model, lstm_performance = self.train_lstm_model(X, y)

            # Store models and performance
            self.models[ticker] = {
                'traditional': traditional_models,
                'lstm': lstm_model,
                'scaler': scaler,
                'feature_columns': [col for col in df.columns if not col.startswith('target_')]
            }

            self.model_performance[ticker] = {
                'traditional': traditional_performance,
                'lstm': lstm_performance,
                'training_date': datetime.now().isoformat(),
                'data_points': len(df)
            }

            logger.info(f"Model training complete for {ticker}")

            return {
                'success': True,
                'models_trained': len(traditional_models) + (1 if lstm_model else 0),
                'performance': self.model_performance[ticker]
            }

        except Exception as e:
            logger.error(f"Error training models for {ticker}: {e}")
            return {'success': False, 'error': str(e)}

    def predict_price_movement(self, ticker: str, current_data: pd.DataFrame,
                               sentiment_data: List[Dict] = None,
                               market_data: Dict = None) -> Dict:
        """Predict price movement for a ticker"""

        try:
            if ticker not in self.models:
                raise ValueError(f"No trained models found for {ticker}")

            # Feature engineering on current data
            df = self.feature_engineer.engineer_features(current_data, sentiment_data, market_data)

            # Get the latest features
            feature_cols = self.models[ticker]['feature_columns']
            latest_features = df[feature_cols].iloc[-1:].values

            # Scale features
            scaler = self.models[ticker]['scaler']
            latest_scaled = scaler.transform(latest_features)

            # Predictions from traditional models
            predictions = {}
            traditional_models = self.models[ticker]['traditional']

            for model_name, model in traditional_models.items():
                pred = model.predict(latest_scaled)[0]
                predictions[model_name] = pred

            # LSTM prediction
            if self.models[ticker]['lstm'] and TORCH_AVAILABLE:
                lstm_model = self.models[ticker]['lstm']
                sequence_length = 10

                if len(df) >= sequence_length:
                    # Create sequence for LSTM
                    sequence_data = df[feature_cols].tail(sequence_length).values
                    sequence_scaled = scaler.transform(sequence_data)
                    sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0)

                    lstm_model.eval()
                    with torch.no_grad():
                        lstm_pred = lstm_model(sequence_tensor).item()
                        predictions['lstm'] = lstm_pred

            # Ensemble prediction (weighted average)
            if predictions:
                weights = {
                    'random_forest': 0.25,
                    'xgboost': 0.30,
                    'gradient_boosting': 0.25,
                    'lstm': 0.20
                }

                ensemble_pred = 0
                total_weight = 0

                for model_name, pred in predictions.items():
                    weight = weights.get(model_name, 0.1)
                    ensemble_pred += pred * weight
                    total_weight += weight

                ensemble_pred = ensemble_pred / total_weight if total_weight > 0 else 0

                # Convert return to price prediction
                current_price = current_data['close'].iloc[-1]
                predicted_price = current_price * (1 + ensemble_pred)

                # Calculate confidence based on model agreement
                pred_values = list(predictions.values())
                confidence = 1 - (np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8))
                confidence = max(0, min(1, confidence))

                return {
                    'ticker': ticker,
                    'current_price': current_price,
                    'predicted_return': ensemble_pred,
                    'predicted_price': predicted_price,
                    'price_change': predicted_price - current_price,
                    'price_change_pct': (predicted_price - current_price) / current_price * 100,
                    'confidence': confidence,
                    'individual_predictions': predictions,
                    'prediction_date': datetime.now().isoformat(),
                    'direction': 'up' if ensemble_pred > 0 else 'down',
                    'magnitude': 'high' if abs(ensemble_pred) > 0.02 else 'moderate' if abs(
                        ensemble_pred) > 0.01 else 'low'
                }
            else:
                raise ValueError("No valid predictions generated")

        except Exception as e:
            logger.error(f"Error predicting for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'prediction_date': datetime.now().isoformat()
            }

    def get_model_performance(self, ticker: str) -> Dict:
        """Get performance metrics for trained models"""

        if ticker in self.model_performance:
            return self.model_performance[ticker]
        else:
            return {'error': f'No performance data available for {ticker}'}

    def save_models(self, filepath: str):
        """Save trained models to disk"""

        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'models': self.models,
                    'performance': self.model_performance,
                    'feature_engineer': self.feature_engineer
                }, f)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self, filepath: str):
        """Load trained models from disk"""

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.model_performance = data['performance']
                self.feature_engineer = data['feature_engineer']
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")


def main():
    """Test the ML price forecaster"""

    forecaster = MLPriceForecaster()

    # Test with a sample ticker
    ticker = 'AAPL'

    print(f"🤖 Testing ML Price Forecaster with {ticker}")

    # Train models
    print("Training models...")
    result = forecaster.train_models(ticker)

    if result['success']:
        print(f"✅ Successfully trained {result['models_trained']} models")

        # Get current data for prediction
        current_data = forecaster.get_historical_data(ticker, days=50)

        if not current_data.empty:
            # Make prediction
            print("Making prediction...")
            prediction = forecaster.predict_price_movement(ticker, current_data)

            if 'error' not in prediction:
                print(f"\n📊 Prediction Results for {ticker}:")
                print(f"Current Price: ${prediction['current_price']:.2f}")
                print(f"Predicted Price: ${prediction['predicted_price']:.2f}")
                print(f"Expected Change: {prediction['price_change_pct']:.2f}%")
                print(f"Direction: {prediction['direction']}")
                print(f"Magnitude: {prediction['magnitude']}")
                print(f"Confidence: {prediction['confidence']:.2f}")
            else:
                print(f"❌ Prediction failed: {prediction['error']}")
        else:
            print("❌ Could not get current data for prediction")
    else:
        print(f"❌ Model training failed: {result['error']}")


if __name__ == "__main__":
    main()