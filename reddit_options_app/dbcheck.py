"""
Database Data Verification Script
Run this to see exactly what data is in your Firebase database
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.firebase_manager import FirebaseManager
from datetime import datetime, timezone


def check_database_contents():
    """Check what's actually in your Firebase database"""

    print("🔍 CHECKING FIREBASE DATABASE CONTENTS")
    print("=" * 60)

    try:
        fm = FirebaseManager()

        # 1. Check recent posts
        print("\n📊 RECENT POSTS (last 24 hours):")
        recent_posts = fm.get_recent_posts(limit=10, hours=24, use_cache=False)

        if recent_posts:
            print(f"✅ Found {len(recent_posts)} recent posts")
            for i, post in enumerate(recent_posts[:3], 1):
                title = post.get('title', 'No title')[:50] + '...'
                tickers = post.get('tickers', [])
                score = post.get('score', 0)
                created = post.get('created_datetime', 'Unknown time')
                print(f"   {i}. {title}")
                print(f"      Tickers: {tickers}, Score: {score}")
                print(f"      Created: {created}")
        else:
            print("❌ NO RECENT POSTS FOUND")
            print("   Your continuous scraper may not be running or has no data")

        # 2. Check trending tickers
        print(f"\n📈 TRENDING TICKERS (24h):")
        trending = fm.get_trending_tickers(hours=24, min_mentions=1, use_cache=False)

        if trending:
            print(f"✅ Found {len(trending)} trending tickers")
            for ticker_data in trending[:5]:
                ticker = ticker_data.get('ticker', 'Unknown')
                mentions = ticker_data.get('mention_count', 0)
                avg_score = ticker_data.get('avg_score', 0)
                print(f"   • {ticker}: {mentions} mentions, avg score: {avg_score:.1f}")
        else:
            print("❌ NO TRENDING TICKERS FOUND")

        # 3. Check sentiment data
        print(f"\n🧠 SENTIMENT ANALYSIS:")
        sentiment_data = fm.get_sentiment_overview(hours=24, use_cache=False)

        if sentiment_data:
            print(f"✅ Found sentiment data for {len(sentiment_data)} tickers")
            for sent in sentiment_data[:5]:
                ticker = sent.get('ticker', 'Unknown')
                sentiment = sent.get('sentiment', 'neutral')
                confidence = sent.get('confidence', 0)
                mentions = sent.get('mention_count', 0)
                print(f"   • {ticker}: {sentiment} ({confidence:.2f} confidence, {mentions} mentions)")
        else:
            print("❌ NO SENTIMENT DATA FOUND")

        # 4. Check database quota usage
        print(f"\n📊 FIREBASE QUOTA STATUS:")
        quota_status = fm.get_quota_status()
        print(f"   Daily reads: {quota_status.get('daily_reads', 0)}")
        print(f"   Daily limit: {quota_status.get('daily_limit', 0)}")
        print(f"   Quota healthy: {quota_status.get('quota_healthy', False)}")

        # 5. Check raw collection counts
        print(f"\n🗄️ RAW COLLECTION DATA:")

        # Check reddit_posts collection
        try:
            all_posts = fm.query_documents('reddit_posts', limit=1, use_cache=False)
            if all_posts:
                print(f"✅ reddit_posts collection has data")
                latest_post = all_posts[0]
                print(f"   Latest post ID: {latest_post.get('id', 'Unknown')}")
                print(f"   Latest post tickers: {latest_post.get('tickers', [])}")
            else:
                print("❌ reddit_posts collection is EMPTY")
        except Exception as e:
            print(f"❌ Error checking reddit_posts: {e}")

        # Check sentiment_analysis collection
        try:
            sentiment_records = fm.query_documents('sentiment_analysis', limit=1, use_cache=False)
            if sentiment_records:
                print(f"✅ sentiment_analysis collection has data")
            else:
                print("❌ sentiment_analysis collection is EMPTY")
        except Exception as e:
            print(f"❌ Error checking sentiment_analysis: {e}")

        # VERDICT
        print(f"\n" + "=" * 60)
        print("🎯 VERDICT:")

        has_posts = len(recent_posts) > 0
        has_trending = len(trending) > 0
        has_sentiment = len(sentiment_data) > 0

        if has_posts and has_trending and has_sentiment:
            print("✅ YOUR DATABASE HAS REAL DATA!")
            print("   The dashboard is showing actual scraped Reddit data")
            print("   Your continuous scraper has been working")
        elif has_posts:
            print("⚠️ PARTIAL DATA FOUND")
            print("   You have Reddit posts but may be missing processed sentiment")
            print("   Your scraper is working but sentiment analysis may need attention")
        else:
            print("❌ NO DATA FOUND - USING FALLBACKS OR EMPTY")
            print("   Your dashboard might be showing:")
            print("   1. Empty results (no tickers)")
            print("   2. Cached data from previous runs")
            print("   3. Some other data source")
            print()
            print("🔧 TO FIX:")
            print("   1. Run your continuous scraper: python continuous_scraper.py --test")
            print("   2. Check your Reddit API credentials")
            print("   3. Verify Firebase connection")

        return has_posts, has_trending, has_sentiment

    except Exception as e:
        print(f"❌ DATABASE CHECK FAILED: {e}")
        print("   There may be a connection issue with Firebase")
        return False, False, False


def check_dashboard_data_source():
    """Check what the dashboard would actually return"""

    print(f"\n🖥️ DASHBOARD DATA SOURCE CHECK:")
    print("=" * 40)

    try:
        # Import your dashboard class
        sys.path.append('.')
        from dashboard import CleanTradingDashboard

        dashboard = CleanTradingDashboard()

        # Get the same data the dashboard gets
        data = dashboard.get_trading_opportunities(max_tickers=5)

        opportunities = data.get('opportunities', [])

        if opportunities:
            print(f"✅ Dashboard returns {len(opportunities)} opportunities:")
            for opp in opportunities[:3]:
                ticker = opp.get('ticker', 'Unknown')
                sentiment = opp.get('sentiment', 'neutral')
                mentions = opp.get('mention_count_24h', 0)
                price = opp.get('current_price', 0)
                print(f"   • {ticker}: {sentiment}, {mentions} mentions, ${price:.2f}")
        else:
            print("❌ Dashboard returns NO opportunities")
            print("   This confirms database is empty or scraper not running")

        return len(opportunities) > 0

    except Exception as e:
        print(f"❌ Dashboard check failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting comprehensive database verification...")

    # Check database contents
    has_posts, has_trending, has_sentiment = check_database_contents()

    # Check what dashboard sees
    dashboard_has_data = check_dashboard_data_source()

    print(f"\n🏁 FINAL SUMMARY:")
    print(f"   Database has posts: {has_posts}")
    print(f"   Database has trending: {has_trending}")
    print(f"   Database has sentiment: {has_sentiment}")
    print(f"   Dashboard shows data: {dashboard_has_data}")

    if not (has_posts or dashboard_has_data):
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Start your continuous scraper to populate database")
        print(f"   2. Run: python continuous_scraper.py --test")
        print(f"   3. Wait 5-10 minutes for data to accumulate")
        print(f"   4. Refresh your dashboard")