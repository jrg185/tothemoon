"""
API Diagnostic Script
Tests Alpha Vantage and Finnhub APIs to identify data issues
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests
import json
from datetime import datetime
import time
from config.settings import FINANCIAL_APIS


def test_alpha_vantage_api():
    """Test Alpha Vantage API connection and response"""

    print("🔍 Testing Alpha Vantage API...")

    api_key = FINANCIAL_APIS.get('alpha_vantage')

    if not api_key:
        print("❌ Alpha Vantage API key not found in config")
        return False

    print(f"✅ API key found: {api_key[:10]}...")

    # Test with AAPL
    test_ticker = 'AAPL'
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={test_ticker}&apikey={api_key}"

    try:
        print(f"📡 Making request to Alpha Vantage for {test_ticker}...")
        response = requests.get(url, timeout=30)

        print(f"📊 Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print(f"📄 Response keys: {list(data.keys())}")

            # Check for error messages
            if 'Error Message' in data:
                print(f"❌ API Error: {data['Error Message']}")
                return False

            if 'Note' in data:
                print(f"⚠️ API Note: {data['Note']}")
                print("This usually indicates rate limiting.")
                return False

            if 'Information' in data:
                print(f"ℹ️ API Information: {data['Information']}")

            # Check for time series data
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                print(f"✅ Time Series data found: {len(time_series)} days")

                # Show sample data
                latest_date = max(time_series.keys())
                latest_data = time_series[latest_date]
                print(f"📈 Latest data ({latest_date}): {latest_data}")

                return True
            else:
                print(f"❌ No 'Time Series (Daily)' found in response")
                print(f"Full response: {json.dumps(data, indent=2)}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False


def test_finnhub_api():
    """Test Finnhub API connection and response"""

    print("\n🔍 Testing Finnhub API...")

    api_key = FINANCIAL_APIS.get('finnhub')

    if not api_key:
        print("❌ Finnhub API key not found in config")
        return False

    print(f"✅ API key found: {api_key[:10]}...")

    # Test with AAPL
    test_ticker = 'AAPL'
    url = f"https://finnhub.io/api/v1/quote?symbol={test_ticker}&token={api_key}"

    try:
        print(f"📡 Making request to Finnhub for {test_ticker}...")
        response = requests.get(url, timeout=10)

        print(f"📊 Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print(f"📄 Response: {data}")

            # Check for valid price data
            if 'c' in data and data['c'] is not None and data['c'] > 0:
                print(f"✅ Valid price data: ${data['c']}")
                return True
            else:
                print(f"❌ Invalid or missing price data")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False


def test_sample_requests():
    """Test sample requests to understand response format"""

    print("\n🔍 Testing sample API requests...")

    # Test Alpha Vantage with different symbols
    test_symbols = ['AAPL', 'MSFT', 'TSLA', 'GME', 'VOO', 'PLTR']

    for symbol in test_symbols[:3]:  # Test first 3 to avoid rate limits
        print(f"\n📊 Testing {symbol}...")

        api_key = FINANCIAL_APIS.get('alpha_vantage')
        if api_key:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"

            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    data = response.json()

                    if 'Time Series (Daily)' in data:
                        print(f"✅ {symbol}: Data available")
                    elif 'Note' in data:
                        print(f"⚠️ {symbol}: Rate limited - {data['Note']}")
                    elif 'Error Message' in data:
                        print(f"❌ {symbol}: Error - {data['Error Message']}")
                    else:
                        print(f"❓ {symbol}: Unexpected response format")
                        print(f"Keys: {list(data.keys())}")
                else:
                    print(f"❌ {symbol}: HTTP {response.status_code}")

            except Exception as e:
                print(f"❌ {symbol}: Exception - {str(e)}")

        # Wait between requests to avoid rate limiting
        time.sleep(12)  # Alpha Vantage free tier: 5 requests per minute


def check_ml_forecaster_fix():
    """Check if ML forecaster can be fixed with current data"""

    print("\n🔍 Checking ML Forecaster fix options...")

    # Option 1: Disable ML forecaster completely
    print("💡 Option 1: Disable ML forecaster in advanced analytics")

    # Option 2: Use yfinance instead of Alpha Vantage
    print("💡 Option 2: Switch to yfinance for historical data")

    try:
        import yfinance as yf
        print("✅ yfinance is available as alternative")

        # Test yfinance
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="5d")

        if not hist.empty:
            print(f"✅ yfinance test successful: {len(hist)} days of data")
            print(f"Latest close: ${hist['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ yfinance returned empty data")

    except ImportError:
        print("❌ yfinance not available")
    except Exception as e:
        print(f"❌ yfinance test failed: {str(e)}")

    return False


def main():
    """Run all API diagnostics"""

    print("🚀 Starting API Diagnostic...")
    print("=" * 50)

    # Test APIs
    av_success = test_alpha_vantage_api()
    fh_success = test_finnhub_api()

    # Test sample requests
    test_sample_requests()

    # Check fix options
    yf_available = check_ml_forecaster_fix()

    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY:")
    print(f"Alpha Vantage API: {'✅ Working' if av_success else '❌ Failed'}")
    print(f"Finnhub API: {'✅ Working' if fh_success else '❌ Failed'}")
    print(f"yfinance Alternative: {'✅ Available' if yf_available else '❌ Not available'}")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")

    if not av_success:
        print("1. ❌ Alpha Vantage API is not working - ML forecaster will fail")
        print("   - Check API key configuration")
        print("   - Verify Alpha Vantage account status")
        print("   - Consider rate limiting issues")

        if yf_available:
            print("2. ✅ Use yfinance as alternative for historical data")
        else:
            print("2. ❌ Disable ML forecaster until Alpha Vantage is fixed")

    if not fh_success:
        print("3. ❌ Finnhub API is not working - price data will be unavailable")
        print("   - Check API key configuration")
        print("   - Verify Finnhub account status")

    print("\n🔧 IMMEDIATE FIXES:")
    print("1. Use the debug dashboard to isolate issues")
    print("2. Disable advanced analytics if APIs are down")
    print("3. Check Firebase data availability")
    print("4. Remove auto-refresh to prevent infinite loops")


if __name__ == "__main__":
    main()