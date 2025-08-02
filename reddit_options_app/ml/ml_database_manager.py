"""
ML Database Manager for Reddit Options App
Manages ML predictions and training data in Firebase
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import logging
import json

from data.firebase_manager import FirebaseManager

logger = logging.getLogger(__name__)


class MLDatabaseManager:
    """Manages ML predictions and training data in Firebase"""

    def __init__(self, firebase_manager: FirebaseManager = None):
        """Initialize with existing Firebase manager"""
        self.firebase_manager = firebase_manager or FirebaseManager()

        # Collection names for ML data
        self.predictions_collection = 'ml_predictions'
        self.performance_collection = 'ml_model_performance'
        self.training_data_collection = 'ml_training_data'

        logger.info("ML Database Manager initialized")

    def save_prediction(self, prediction_data: Dict) -> str:
        """Save ML prediction to Firebase"""
        try:
            # Add metadata
            prediction_data.update({
                'created_at': datetime.now(timezone.utc).isoformat(),
                'model_version': 'v2.0',
                'status': 'pending'
            })

            # Save to Firebase
            doc_id = prediction_data['prediction_id']
            result = self.firebase_manager.save_document(
                self.predictions_collection,
                prediction_data,
                doc_id
            )

            logger.info(f"✅ Saved prediction {doc_id} to Firebase")
            return doc_id

        except Exception as e:
            logger.error(f"❌ Failed to save prediction: {e}")
            return None

    def update_prediction_result(self, prediction_id: str, actual_data: Dict) -> bool:
        """Update prediction with actual results"""
        try:
            # Get existing prediction
            existing = self.firebase_manager.get_document(
                self.predictions_collection,
                prediction_id
            )

            if not existing:
                logger.warning(f"Prediction {prediction_id} not found")
                return False

            # Add actual results
            update_data = existing.copy()
            update_data.update(actual_data)
            update_data.update({
                'status': 'checked',
                'check_time': datetime.now(timezone.utc).isoformat()
            })

            # Calculate accuracy
            correct_predictions = 0
            total_predictions = 0

            for target in ['3d', '5d']:
                if f'prediction_correct_{target}' in update_data:
                    total_predictions += 1
                    if update_data[f'prediction_correct_{target}']:
                        correct_predictions += 1

            if total_predictions > 0:
                update_data['accuracy_score'] = correct_predictions / total_predictions

            # Save updated prediction
            self.firebase_manager.save_document(
                self.predictions_collection,
                update_data,
                prediction_id
            )

            logger.info(f"✅ Updated prediction {prediction_id} with actual results")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update prediction {prediction_id}: {e}")
            return False

    def get_recent_predictions(self, ticker: str = None, days: int = 30) -> List[Dict]:
        """Get recent predictions for analysis"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            cutoff_iso = cutoff_time.isoformat()

            filters = [('prediction_time', '>=', cutoff_iso)]
            if ticker:
                filters.append(('ticker', '==', ticker))

            predictions = self.firebase_manager.query_documents(
                self.predictions_collection,
                filters=filters,
                order_by='prediction_time',
                desc=True,
                limit=500
            )

            logger.info(f"Retrieved {len(predictions)} recent predictions")
            return predictions

        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return []

    def calculate_model_performance(self, ticker: str) -> Dict:
        """Calculate and save model performance metrics"""
        try:
            # Get recent predictions for this ticker
            recent_predictions = self.get_recent_predictions(ticker, days=30)

            if not recent_predictions:
                return {'error': 'No recent predictions found'}

            # Calculate performance metrics
            total_predictions = len(recent_predictions)
            checked_predictions = [p for p in recent_predictions if p.get('status') == 'checked']

            if not checked_predictions:
                return {'error': 'No checked predictions found'}

            # Overall accuracy
            correct_total = sum(1 for p in checked_predictions if p.get('accuracy_score', 0) > 0.5)
            overall_accuracy = correct_total / len(checked_predictions)

            # Target-specific accuracy
            accuracy_3d = []
            accuracy_5d = []

            for p in checked_predictions:
                if 'prediction_correct_3d' in p:
                    accuracy_3d.append(p['prediction_correct_3d'])
                if 'prediction_correct_5d' in p:
                    accuracy_5d.append(p['prediction_correct_5d'])

            performance_data = {
                'model_id': f"{ticker}_v2_{datetime.now().strftime('%Y%m%d')}",
                'ticker': ticker,
                'model_version': 'v2.0',
                'last_performance_check': datetime.now(timezone.utc).isoformat(),
                'total_predictions': total_predictions,
                'checked_predictions': len(checked_predictions),
                'correct_predictions': correct_total,
                'overall_accuracy': overall_accuracy,
                'accuracy_3d': sum(accuracy_3d) / len(accuracy_3d) if accuracy_3d else 0,
                'accuracy_5d': sum(accuracy_5d) / len(accuracy_5d) if accuracy_5d else 0,
                'needs_retraining': overall_accuracy < 0.55,  # Retrain if accuracy < 55%
                'performance_trend': 'stable'
            }

            # Save performance data
            self.firebase_manager.save_document(
                self.performance_collection,
                performance_data,
                performance_data['model_id']
            )

            logger.info(f"✅ Calculated performance for {ticker}: {overall_accuracy:.2%} accuracy")
            return performance_data

        except Exception as e:
            logger.error(f"❌ Failed to calculate performance for {ticker}: {e}")
            return {'error': str(e)}

    def save_training_data(self, ticker: str, training_data: Dict) -> str:
        """Save training data for future reference"""
        try:
            data_id = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            training_record = {
                'data_id': data_id,
                'ticker': ticker,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'model_version': 'v2.0',
                **training_data
            }

            self.firebase_manager.save_document(
                self.training_data_collection,
                training_record,
                data_id
            )

            logger.info(f"✅ Saved training data {data_id} to Firebase")
            return data_id

        except Exception as e:
            logger.error(f"❌ Failed to save training data: {e}")
            return None

    def get_performance_summary(self) -> Dict:
        """Get overall ML system performance summary"""
        try:
            # Get recent performance records
            recent_performance = self.firebase_manager.query_documents(
                self.performance_collection,
                order_by='last_performance_check',
                desc=True,
                limit=20
            )

            if not recent_performance:
                return {'error': 'No performance data found'}

            # Calculate summary statistics
            accuracies = [p.get('overall_accuracy', 0) for p in recent_performance]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

            models_needing_retraining = sum(1 for p in recent_performance if p.get('needs_retraining', False))

            best_model = max(recent_performance, key=lambda x: x.get('overall_accuracy', 0))

            return {
                'total_models': len(recent_performance),
                'average_accuracy': avg_accuracy,
                'models_needing_retraining': models_needing_retraining,
                'best_performing_ticker': best_model.get('ticker', 'N/A'),
                'best_accuracy': best_model.get('overall_accuracy', 0),
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {'error': str(e)}

    def check_stale_predictions(self, days_old: int = 5) -> Dict:
        """Check and update predictions that are old enough to verify"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_old)

            # Get pending predictions older than cutoff
            stale_predictions = self.firebase_manager.query_documents(
                self.predictions_collection,
                filters=[
                    ('status', '==', 'pending'),
                    ('prediction_time', '<=', cutoff_time.isoformat())
                ],
                limit=50
            )

            updated_count = 0
            results = []

            for prediction in stale_predictions:
                prediction_id = prediction['prediction_id']
                ticker = prediction['ticker']

                # Mark for manual verification (in real implementation, you'd fetch actual prices)
                update_data = {
                    'status': 'needs_verification',
                    'check_attempted': datetime.now(timezone.utc).isoformat()
                }

                if self.update_prediction_result(prediction_id, update_data):
                    updated_count += 1
                    results.append({
                        'prediction_id': prediction_id,
                        'ticker': ticker,
                        'status': 'marked_for_verification'
                    })

            return {
                'stale_predictions_found': len(stale_predictions),
                'predictions_updated': updated_count,
                'results': results
            }

        except Exception as e:
            logger.error(f"Failed to check stale predictions: {e}")
            return {'error': str(e)}


def test_ml_database_manager():
    """Test the ML database manager"""

    print("🧪 Testing ML Database Manager")
    print("=" * 50)

    try:
        # Initialize
        ml_db = MLDatabaseManager()

        # Test saving a prediction
        test_prediction = {
            'prediction_id': f'TEST_AAPL_{int(datetime.now().timestamp())}',
            'ticker': 'AAPL',
            'prediction_time': datetime.now().isoformat(),
            'current_price': 150.25,
            'predicted_direction': 'up',
            'confidence': 0.68,
            'detailed_predictions': {
                'target_direction_3d': {'prediction': 1, 'probability': 0.72},
                'target_direction_5d': {'prediction': 1, 'probability': 0.68}
            }
        }

        # Save prediction
        prediction_id = ml_db.save_prediction(test_prediction)

        if prediction_id:
            print(f"✅ Saved test prediction: {prediction_id}")

            # Test updating with results
            test_results = {
                'actual_price_3d': 155.20,
                'actual_return_3d': 0.033,
                'prediction_correct_3d': True,
                'actual_price_5d': 148.90,
                'actual_return_5d': -0.009,
                'prediction_correct_5d': False
            }

            if ml_db.update_prediction_result(prediction_id, test_results):
                print("✅ Updated prediction with test results")
            else:
                print("❌ Failed to update prediction")
        else:
            print("❌ Failed to save test prediction")

        # Test getting recent predictions
        recent = ml_db.get_recent_predictions('AAPL', days=7)
        print(f"📊 Found {len(recent)} recent AAPL predictions")

        # Test performance summary
        summary = ml_db.get_performance_summary()
        print(f"📈 Performance summary: {summary}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_ml_database_manager()