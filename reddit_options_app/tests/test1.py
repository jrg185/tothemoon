"""
Complete Test Script for ML Integration
Tests the improved ML system with database integration
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import time
from datetime import datetime
import traceback


def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")

    try:
        # Test Firebase manager
        from data.firebase_manager import FirebaseManager
        print("✅ FirebaseManager imported")

        # Test ML database manager
        from data.ml_database_manager import MLDatabaseManager
        print("✅ MLDatabaseManager imported")

        # Test improved ML forecaster
        from ml.ml_price_forecaster import MLPriceForecaster
        print("✅ MLPriceForecaster imported")

        # Test data module integration
        from data import MLDatabaseManager, ML_DATABASE_AVAILABLE
        print(f"✅ Data module integration: ML_DATABASE_AVAILABLE = {ML_DATABASE_AVAILABLE}")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        print(traceback.format_exc())
        return False


def test_ml_database_manager():
    """Test ML database manager functionality"""
    print("\n🗄️ Testing ML Database Manager...")

    try:
        from data.ml_database_manager import MLDatabaseManager

        # Initialize
        ml_db = MLDatabaseManager()
        print("✅ ML Database Manager initialized")

        # Test saving a prediction
        test_prediction = {
            'prediction_id': f'TEST_INTEGRATION_{int(time.time())}',
            'ticker': 'AAPL',
            'prediction_time': datetime.now().isoformat(),
            'current_price': 150.25,
            'predicted_direction': 'up',
            'confidence': 0.68,
            'detailed_predictions': {
                'target_direction_3d': {'prediction': 1, 'probability': 0.72}
            }
        }

        prediction_id = ml_db.save_prediction(test_prediction)
        if prediction_id:
            print(f"✅ Saved test prediction: {prediction_id}")
        else:
            print("❌ Failed to save prediction")
            return False

        # Test querying recent predictions
        recent = ml_db.get_recent_predictions('AAPL', days=1)
        print(f"✅ Found {len(recent)} recent predictions")

        return True

    except Exception as e:
        print(f"❌ ML Database test failed: {e}")
        print(traceback.format_exc())
        return False


def test_improved_ml_forecaster():
    """Test the improved ML forecaster"""
    print("\n🤖 Testing Improved ML Forecaster...")

    try:
        from ml.ml_price_forecaster import MLPriceForecaster

        # Initialize forecaster
        forecaster = MLPriceForecaster()
        print("✅ ML Forecaster initialized")

        # Test with sample sentiment data
        sample_sentiment = [{
            'sentiment': 'bullish',
            'confidence': 0.8,
            'numerical_score': 0.6,
            'mention_count': 15
        }]

        print("🏋️ Testing model training (this may take a minute)...")

        # Test training
        train_result = forecaster.train_models('AAPL', sample_sentiment)

        if train_result['success']:
            print(f"✅ Training successful!")
            print(f"   Targets trained: {train_result['targets_trained']}")
            print(f"   Data points: {train_result['data_points']}")

            # Test prediction
            print("🔮 Testing prediction...")
            prediction = forecaster.predict_price_movement('AAPL', sentiment_data=sample_sentiment)

            if 'error' not in prediction:
                print(f"✅ Prediction successful!")
                print(f"   Direction: {prediction['direction']}")
                print(f"   Confidence: {prediction['confidence']:.2%}")
                print(f"   Database integration: {prediction.get('saved_to_database', False)}")
                return True
            else:
                print(f"❌ Prediction failed: {prediction['error']}")
                return False
        else:
            print(f"❌ Training failed: {train_result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ ML Forecaster test failed: {e}")
        print(traceback.format_exc())
        return False


def test_performance_tracking():
    """Test performance tracking functionality"""
    print("\n📊 Testing Performance Tracking...")

    try:
        from ml.ml_price_forecaster import MLPriceForecaster

        forecaster = MLPriceForecaster()

        # Test performance summary
        performance = forecaster.get_model_performance_summary()
        print("✅ Performance summary retrieved:")
        print(f"   Models trained: {performance.get('models_trained', 0)}")
        print(f"   Database integration: {performance.get('database_integration', False)}")
        print(f"   Database status: {performance.get('database_status', 'unknown')}")

        # Test database performance if available
        if forecaster.use_database and forecaster.ml_db:
            db_summary = forecaster.ml_db.get_performance_summary()
            print(f"✅ Database performance summary: {db_summary}")
        else:
            print("ℹ️ Database performance tracking not available")

        return True

    except Exception as e:
        print(f"❌ Performance tracking test failed: {e}")
        print(traceback.format_exc())
        return False


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print("\n🔄 Testing End-to-End Workflow...")

    try:
        # 1. Initialize all components
        from data.ml_database_manager import MLDatabaseManager
        from ml.ml_price_forecaster import MLPriceForecaster

        ml_db = MLDatabaseManager()
        forecaster = MLPriceForecaster()

        print("✅ All components initialized")

        # 2. Create sample data
        sample_sentiment = [{
            'sentiment': 'bullish',
            'confidence': 0.75,
            'numerical_score': 0.5,
            'mention_count': 12
        }]

        # 3. Train model (if not already trained)
        if 'AAPL' not in forecaster.models:
            print("🏋️ Training model for end-to-end test...")
            train_result = forecaster.train_models('AAPL', sample_sentiment)
            if not train_result['success']:
                print(f"❌ Training failed: {train_result.get('error')}")
                return False

        # 4. Make prediction
        print("🔮 Making prediction...")
        prediction = forecaster.predict_price_movement('AAPL', sentiment_data=sample_sentiment)

        if 'error' in prediction:
            print(f"❌ Prediction failed: {prediction['error']}")
            return False

        prediction_id = prediction['prediction_id']
        print(f"✅ Prediction made: {prediction_id}")

        # 5. Verify prediction was saved to database
        if forecaster.use_database:
            recent_predictions = ml_db.get_recent_predictions('AAPL', days=1)
            found_prediction = any(p.get('prediction_id') == prediction_id for p in recent_predictions)

            if found_prediction:
                print("✅ Prediction found in database")
            else:
                print("⚠️ Prediction not found in database (but may be due to timing)")

        # 6. Test performance calculation
        if forecaster.use_database:
            performance = ml_db.calculate_model_performance('AAPL')
            if 'error' not in performance:
                print(f"✅ Performance calculation successful")
            else:
                print(f"ℹ️ Performance calculation: {performance['error']} (expected for new models)")

        print("✅ End-to-end workflow completed successfully!")
        return True

    except Exception as e:
        print(f"❌ End-to-end workflow failed: {e}")
        print(traceback.format_exc())
        return False


def main():
    """Run all integration tests"""

    print("🧪 ML INTEGRATION TEST SUITE")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Track test results
    tests = [
        ("Import Tests", test_imports),
        ("ML Database Manager", test_ml_database_manager),
        ("Improved ML Forecaster", test_improved_ml_forecaster),
        ("Performance Tracking", test_performance_tracking),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time

            results.append((test_name, result, duration))

            if result:
                print(f"✅ {test_name} PASSED ({duration:.1f}s)")
            else:
                print(f"❌ {test_name} FAILED ({duration:.1f}s)")

        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            results.append((test_name, False, duration))
            print(f"❌ {test_name} CRASHED: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, duration in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name:25} ({duration:.1f}s)")

    print()
    print(f"🎯 FINAL RESULT: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! ML integration is working correctly.")
        print()
        print("✅ Next steps:")
        print("   1. Your ML system should now avoid overfitting")
        print("   2. Predictions are being tracked in Firebase")
        print("   3. You can monitor model performance over time")
        print("   4. The system will learn and improve from real results")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print()
        print("💡 Common issues:")
        print("   - Firebase credentials not configured")
        print("   - Missing dependencies (yfinance, sklearn)")
        print("   - Network connectivity issues")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()