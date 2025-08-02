"""
Test script for Reddit Options App components
Run this from the project root directory
"""

import sys
from pathlib import Path

# Ensure we're in the right directory
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_config():
    """Test configuration loading"""
    print("🔧 Testing Configuration...")
    try:
        from config.settings import APP_CONFIG, LLM_CONFIG, FIREBASE_CONFIG
        print("✅ Config imported successfully")
        print(f"   Debug mode: {APP_CONFIG.get('debug')}")
        print(f"   LLM provider: {LLM_CONFIG.get('default_provider')}")
        print(f"   Firebase project: {FIREBASE_CONFIG.get('project_id', 'Not set')}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_firebase_manager():
    """Test Firebase manager (without actual connection)"""
    print("\n🔥 Testing Firebase Manager...")
    try:
        from data.firebase_manager import FirebaseManager
        print("✅ FirebaseManager imported successfully")

        # Test class methods exist
        manager_methods = ['save_document', 'batch_save', 'query_documents', 'get_trending_tickers']
        for method in manager_methods:
            if hasattr(FirebaseManager, method):
                print(f"✅ Method {method} found")
            else:
                print(f"❌ Method {method} missing")
                return False

        return True
    except Exception as e:
        print(f"❌ Firebase manager test failed: {e}")
        return False


def test_reddit_scraper():
    """Test Reddit scraper (without actual connection)"""
    print("\n🔴 Testing Reddit Scraper...")
    try:
        from data.reddit_scraper import RedditScraper, TickerExtractor
        print("✅ RedditScraper imported successfully")

        # Test ticker extraction
        extractor = TickerExtractor()
        test_text = "Buying $TSLA calls and AAPL puts, also watching GME"
        tickers = extractor.extract_tickers(test_text)
        print(f"✅ Ticker extraction test: {tickers}")

        if 'TSLA' in tickers and 'AAPL' in tickers:
            print("✅ Ticker extraction working correctly")
            return True
        else:
            print("❌ Ticker extraction not working properly")
            return False

    except Exception as e:
        print(f"❌ Reddit scraper test failed: {e}")
        return False


def test_firebase_connection():
    """Test actual Firebase connection (requires credentials)"""
    print("\n🔥 Testing Firebase Connection...")
    try:
        from data.firebase_manager import FirebaseManager

        # This will fail if credentials aren't set up
        fm = FirebaseManager()
        print("✅ Firebase connection successful!")
        return True

    except Exception as e:
        print(f"❌ Firebase connection failed: {e}")
        print("   This is expected if you haven't set up Firebase credentials yet")
        return False


def main():
    """Run all tests"""
    print("🧪 Reddit Options App Component Tests")
    print("=" * 50)

    # Track results
    results = []

    # Test 1: Configuration
    results.append(test_config())

    # Test 2: Firebase Manager (import only)
    results.append(test_firebase_manager())

    # Test 3: Reddit Scraper (import only)
    results.append(test_reddit_scraper())

    # Test 4: Firebase Connection (requires credentials)
    results.append(test_firebase_connection())

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    test_names = ["Config", "Firebase Manager", "Reddit Scraper", "Firebase Connection"]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")

    passed = sum(results)
    total = len(results)
    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed >= 3:  # First 3 should pass even without credentials
        print("🎉 Core components are working! Ready for development.")
    else:
        print("🔧 Some components need attention before proceeding.")


if __name__ == "__main__":
    main()