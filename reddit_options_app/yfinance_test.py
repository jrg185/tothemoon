"""
Quick yfinance test script
Run this first to verify yfinance works on your system
"""


def test_yfinance():
    """Test if yfinance works and can get data"""

    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError:
        print("❌ yfinance not installed")
        print("Install with: pip install yfinance")
        return False

    # Test data retrieval
    test_tickers = ['AAPL', 'MSFT', 'TSLA']

    for ticker in test_tickers:
        try:
            print(f"\n📊 Testing {ticker}...")

            # Get 1 month of data
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="1mo")

            if hist.empty:
                print(f"❌ No data for {ticker}")
            else:
                latest_close = hist['Close'].iloc[-1]
                print(f"✅ {ticker}: {len(hist)} days, latest: ${latest_close:.2f}")

        except Exception as e:
            print(f"❌ {ticker} failed: {e}")

    print("\n🎯 yfinance test complete!")
    return True


if __name__ == "__main__":
    test_yfinance()