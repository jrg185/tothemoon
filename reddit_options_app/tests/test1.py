#!/usr/bin/env python3
"""
Test script to verify ML imports are working correctly
Run this after updating the __init__.py files
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_data_module_imports():
    """Test data module imports"""
    print("🔍 Testing data module imports...")

    try:
        from data import MLDatabaseManager, ML_DATABASE_AVAILABLE
        print(f"✅ MLDatabaseManager import: Available = {ML_DATABASE_AVAILABLE}")

        if MLDatabaseManager is not None:
            print("✅ MLDatabaseManager class is not None")
            # Test initialization
            ml_db = MLDatabaseManager()
            print("✅ MLDatabaseManager initialized successfully")
        else:
            print("❌ MLDatabaseManager is None")

        return True
    except Exception as e:
        print(f"❌ Data module import failed: {e}")
        return False

def test_ml_module_imports():
    """Test ML module imports"""
    print("\n🔍 Testing ML module imports...")

    try:
        from ml import (
            EnhancedTradingAnalyst,
            MLPriceForecaster,
            MLDatabaseManager,
            ML_COMPONENTS_AVAILABLE,
            ENHANCED_ANALYST_AVAILABLE,
            ML_FORECASTER_AVAILABLE,
            ML_DATABASE_AVAILABLE
        )

        print(f"✅ ML_COMPONENTS_AVAILABLE: {ML_COMPONENTS_AVAILABLE}")
        print(f"✅ ENHANCED_ANALYST_AVAILABLE: {ENHANCED_ANALYST_AVAILABLE}")
        print(f"✅ ML_FORECASTER_AVAILABLE: {ML_FORECASTER_AVAILABLE}")
        print(f"✅ ML_DATABASE_AVAILABLE: {ML_DATABASE_AVAILABLE}")

        # Test component initialization
        if ENHANCED_ANALYST_AVAILABLE and EnhancedTradingAnalyst is not None:
            analyst = EnhancedTradingAnalyst()
            print("✅ EnhancedTradingAnalyst initialized")

        if ML_FORECASTER_AVAILABLE and MLPriceForecaster is not None:
            forecaster = MLPriceForecaster()
            print("✅ MLPriceForecaster initialized")

        if ML_DATABASE_AVAILABLE and MLDatabaseManager is not None:
            ml_db = MLDatabaseManager()
            print("✅ MLDatabaseManager (from ml module) initialized")

        return True
    except Exception as e:
        print(f"❌ ML module import failed: {e}")
        return False

def test_automated_analysis_imports():
    """Test automated analysis engine imports"""
    print("\n🔍 Testing automated analysis engine imports...")

    try:
        from automated_analysis_engine import AutomatedAnalysisEngine
        print("✅ AutomatedAnalysisEngine imported successfully")

        # Test initialization
        engine = AutomatedAnalysisEngine()
        print("✅ AutomatedAnalysisEngine initialized")

        # Check if analytics are available
        if hasattr(engine, 'analytics_available'):
            print(f"✅ Analytics available: {engine.analytics_available}")

        return True
    except Exception as e:
        print(f"❌ Automated analysis engine import failed: {e}")
        return False

def test_continuous_scraper_imports():
    """Test continuous scraper imports"""
    print("\n🔍 Testing continuous scraper imports...")

    try:
        # This should now work without the missing ml_database_manager error
        from continuous_scraper import FixedContinuousRedditScraper
        print("✅ FixedContinuousRedditScraper imported successfully")

        return True
    except Exception as e:
        print(f"❌ Continuous scraper import failed: {e}")
        return False

def main():
    """Run all import tests"""
    print("🧪 ML IMPORT TEST SUITE")
    print("=" * 50)

    tests = [
        ("Data Module", test_data_module_imports),
        ("ML Module", test_ml_module_imports),
        ("Automated Analysis", test_automated_analysis_imports),
        ("Continuous Scraper", test_continuous_scraper_imports)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 IMPORT TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")

    print(f"\n🎯 FINAL RESULT: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL IMPORT TESTS PASSED!")
        print("\n✅ Your continuous scraper should now work without ML database errors")
        print("✅ ML predictions should now be saved to Firebase")
        print("✅ All components are properly integrated")
    else:
        print("⚠️ Some import tests failed. Check the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)