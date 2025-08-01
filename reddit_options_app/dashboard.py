"""
Optimized Trading Dashboard with reduced Firebase API calls
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
import time
import requests
from typing import Dict, List, Optional

# Use optimized Firebase manager
from data.firebase_manager import OptimizedFirebaseManager as FirebaseManager
from config.settings import APP_CONFIG, FINANCIAL_APIS
import numpy as np

# Configure Streamlit
st.set_page_config(
    page_title="WSB Options Trader",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Trading-focused CSS (same as before)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00FF88;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .hot-stock {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        border: 2px solid #00FF88;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .bearish-stock {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        border: 2px solid #FF4444;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .neutral-stock {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        border: 2px solid #888888;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .price-up { color: #00FF88; font-weight: bold; }
    .price-down { color: #FF4444; font-weight: bold; }
    .price-neutral { color: #FFFFFF; }
    
    .metric-box {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #00FF88;
        margin: 5px 0;
    }
    
    .alert-box {
        background-color: #2D1B00;
        border: 2px solid #FF8C00;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
    
    .ticker-item {
        background-color: #1E1E1E;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #FF4B4B;
    }
    
    .cache-info {
        background-color: #0E1117;
        padding: 8px;
        border-radius: 4px;
        border-left: 3px solid #4CAF50;
        font-size: 0.8em;
        color: #888;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


class OptimizedTradingDashboard:
    """Optimized trading dashboard with reduced API calls"""

    def __init__(self):
        self.firebase_manager = FirebaseManager()
        self.alpha_vantage_key = FINANCIAL_APIS.get('alpha_vantage')
        self.finnhub_key = FINANCIAL_APIS.get('finnhub')

        # Rate limiting for price API calls
        self.price_cache = {}
        self.price_cache_duration = 300  # 5 minutes
        self.last_price_requests = {}

    def get_stock_price_data(self, ticker: str) -> dict:
        """Get current price data with caching and rate limiting"""
        # Check cache first
        cache_key = ticker.upper()
        current_time = time.time()

        if (cache_key in self.price_cache and
            current_time - self.price_cache[cache_key]['timestamp'] < self.price_cache_duration):
            return self.price_cache[cache_key]['data']

        # Rate limiting - max 1 request per ticker per 30 seconds
        if (cache_key in self.last_price_requests and
            current_time - self.last_price_requests[cache_key] < 30):
            return self._get_default_price_data()

        try:
            if not self.finnhub_key:
                return self._get_default_price_data()

            # Use Finnhub for real-time data
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}"
            response = requests.get(url, timeout=3)

            if response.status_code == 200:
                data = response.json()
                price_data = {
                    'current_price': data.get('c') or 0,
                    'change': data.get('d') or 0,
                    'change_percent': data.get('dp') or 0,
                    'high': data.get('h') or 0,
                    'low': data.get('l') or 0,
                    'open': data.get('o') or 0,
                    'prev_close': data.get('pc') or 0
                }

                # Cache the result
                self.price_cache[cache_key] = {
                    'data': price_data,
                    'timestamp': current_time
                }
                self.last_price_requests[cache_key] = current_time

                return price_data

        except Exception as e:
            # Don't log every failure, just cache misses
            pass

        return self._get_default_price_data()

    def _get_default_price_data(self):
        """Return default price data structure"""
        return {
            'current_price': 0,
            'change': 0,
            'change_percent': 0,
            'high': 0,
            'low': 0,
            'open': 0,
            'prev_close': 0
        }

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_trading_opportunities(_self):
        """Get comprehensive trading data with caching"""
        try:
            # Use optimized Firebase manager with caching
            fm = _self.firebase_manager

            # Get data with smaller limits and caching
            recent_posts = fm.get_recent_posts(limit=200, hours=24, use_cache=True)  # Reduced from 1000
            trending_24h = fm.get_trending_tickers(hours=24, min_mentions=2, use_cache=True)
            trending_1h = fm.get_trending_tickers(hours=1, min_mentions=1, use_cache=True)
            sentiment_overview = fm.get_sentiment_overview(hours=24, use_cache=True)

            # Process only top opportunities to reduce price API calls
            enhanced_opportunities = []

            # Limit to top 20 tickers to reduce price API calls
            top_sentiment_items = sentiment_overview[:20]

            for sentiment_item in top_sentiment_items:
                ticker = sentiment_item.get('ticker')
                if ticker:
                    # Get price data (with caching)
                    price_data = _self.get_stock_price_data(ticker)

                    # Find in trending data
                    trending_info = next((t for t in trending_24h if t.get('ticker') == ticker), {})
                    recent_trending = next((t for t in trending_1h if t.get('ticker') == ticker), {})

                    # Calculate opportunity score
                    opportunity_score = _self.calculate_opportunity_score(
                        sentiment_item, price_data, trending_info, recent_trending
                    )

                    enhanced_opportunities.append({
                        'ticker': ticker,
                        'sentiment': sentiment_item.get('sentiment', 'neutral'),
                        'confidence': float(sentiment_item.get('confidence', 0)),
                        'numerical_score': float(sentiment_item.get('numerical_score', 0)),
                        'mention_count_24h': int(trending_info.get('mention_count', 0)),
                        'mention_count_1h': int(recent_trending.get('mention_count', 0)) if recent_trending else 0,
                        'current_price': float(price_data['current_price']),
                        'change_percent': float(price_data['change_percent']),
                        'volume_trend': 'increasing' if recent_trending else 'stable',
                        'opportunity_score': opportunity_score,
                        'price_data': price_data
                    })

            # Sort by opportunity score
            enhanced_opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)

            return {
                'opportunities': enhanced_opportunities,
                'total_tickers': len(enhanced_opportunities),
                'hot_stocks': [op for op in enhanced_opportunities if op['opportunity_score'] > 50],
                'bullish_plays': [op for op in enhanced_opportunities if op['sentiment'] == 'bullish' and op['confidence'] > 0.5],
                'bearish_plays': [op for op in enhanced_opportunities if op['sentiment'] == 'bearish' and op['confidence'] > 0.5],
                'momentum_plays': [op for op in enhanced_opportunities if op['mention_count_1h'] > 0 or abs(op['change_percent']) > 1],
                'recent_posts': recent_posts[:50],  # Limit recent posts display
                'cache_info': fm.get_cache_stats()
            }

        except Exception as e:
            st.error(f"Error getting trading opportunities: {e}")
            return {
                'opportunities': [],
                'total_tickers': 0,
                'hot_stocks': [],
                'bullish_plays': [],
                'bearish_plays': [],
                'momentum_plays': [],
                'recent_posts': [],
                'cache_info': {}
            }

    def calculate_opportunity_score(self, sentiment_data, price_data, trending_data, recent_trending):
        """Calculate a trading opportunity score (0-100)"""
        score = 0

        try:
            # Sentiment factor (0-30 points)
            confidence = float(sentiment_data.get('confidence', 0))
            sentiment = sentiment_data.get('sentiment', 'neutral')

            if sentiment in ['bullish', 'bearish']:
                score += confidence * 30

            # Mention volume factor (0-25 points)
            mentions_24h = int(trending_data.get('mention_count', 0))
            mentions_1h = int(recent_trending.get('mention_count', 0)) if recent_trending else 0

            if mentions_24h > 10:
                score += 15
            elif mentions_24h > 5:
                score += 10
            elif mentions_24h > 2:
                score += 5

            # Recent momentum factor (0-20 points)
            if mentions_1h > 0:
                score += min(mentions_1h * 5, 20)

            # Price movement factor (0-25 points)
            change_percent = price_data.get('change_percent', 0)
            if change_percent is not None:
                abs_change = abs(float(change_percent))
                if abs_change > 5:
                    score += 25
                elif abs_change > 3:
                    score += 15
                elif abs_change > 1:
                    score += 10

        except (ValueError, TypeError) as e:
            pass

        return min(score, 100)

    def render_cache_info(self, cache_info: Dict):
        """Render cache information"""
        if cache_info:
            st.markdown(f"""
            <div class="cache-info">
            📊 <strong>Cache Status:</strong> {cache_info.get('valid_cached_queries', 0)} valid queries cached | 
            ⏱️ Cache duration: {cache_info.get('cache_duration_seconds', 0)}s | 
            💾 Potential API calls saved: {cache_info.get('cache_hit_potential', '0/0')}
            </div>
            """, unsafe_allow_html=True)

    def render_hot_opportunities(self, data):
        """Render top trading opportunities"""
        st.markdown("## 🔥 Hot Trading Opportunities")

        hot_stocks = data['hot_stocks'][:6]

        if not hot_stocks:
            st.info("No high-scoring opportunities found. Showing top opportunities...")
            hot_stocks = data['opportunities'][:6]

        if not hot_stocks:
            st.info("No trading opportunities available. Make sure sentiment analysis is running.")
            return

        # Create 2 columns for hot stocks
        col1, col2 = st.columns(2)

        for i, stock in enumerate(hot_stocks):
            col = col1 if i % 2 == 0 else col2

            with col:
                sentiment_emoji = {'bullish': '🐂', 'bearish': '🐻', 'neutral': '😐'}[stock['sentiment']]

                # Determine price color
                if stock['change_percent'] > 0:
                    price_class = "price-up"
                    price_arrow = "📈"
                elif stock['change_percent'] < 0:
                    price_class = "price-down"
                    price_arrow = "📉"
                else:
                    price_class = "price-neutral"
                    price_arrow = "➖"

                # Stock card
                card_class = "hot-stock" if stock['sentiment'] == 'bullish' else "bearish-stock" if stock['sentiment'] == 'bearish' else "neutral-stock"

                price_display = f"${stock['current_price']:.2f}" if stock['current_price'] > 0 else "Price N/A"

                st.markdown(f"""
                <div class="{card_class}">
                    <h3>{sentiment_emoji} {stock['ticker']} - Score: {stock['opportunity_score']:.0f}/100</h3>
                    <p><span class="{price_class}">{price_display} {price_arrow} {stock['change_percent']:.1f}%</span></p>
                    <p><strong>Sentiment:</strong> {stock['sentiment'].title()} ({stock['confidence']:.2f} confidence)</p>
                    <p><strong>Mentions:</strong> {stock['mention_count_24h']} (24h) | {stock['mention_count_1h']} (1h)</p>
                    <p><strong>Play:</strong> {'CALLS' if stock['sentiment'] == 'bullish' else 'PUTS' if stock['sentiment'] == 'bearish' else 'NEUTRAL'}</p>
                </div>
                """, unsafe_allow_html=True)

    def render_trading_metrics(self, data):
        """Render key trading metrics"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🔥 Hot Opportunities", len(data['hot_stocks']))

        with col2:
            bullish_count = len(data['bullish_plays'])
            st.metric("🐂 Bullish Plays", bullish_count)

        with col3:
            bearish_count = len(data['bearish_plays'])
            st.metric("🐻 Bearish Plays", bearish_count)

        with col4:
            momentum_count = len(data['momentum_plays'])
            st.metric("⚡ Momentum Plays", momentum_count)

    def render_options_opportunities(self, data):
        """Render specific options trading opportunities"""
        st.markdown("## 💰 Options Trading Opportunities")

        # Separate into calls and puts
        call_opportunities = data['bullish_plays'][:5]
        put_opportunities = data['bearish_plays'][:5]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📞 CALL Options (Bullish)")
            if call_opportunities:
                for stock in call_opportunities:
                    confidence_bar = "🟢" * max(1, int(stock['confidence'] * 10))
                    price_display = f"${stock['current_price']:.2f}" if stock['current_price'] > 0 else "N/A"
                    st.markdown(f"""
                    <div class="metric-box">
                        <strong>{stock['ticker']}</strong> - {price_display}<br>
                        Confidence: {confidence_bar} ({stock['confidence']:.2f})<br>
                        Change: <span class="{'price-up' if stock['change_percent'] > 0 else 'price-down'}">{stock['change_percent']:.1f}%</span><br>
                        Mentions: {stock['mention_count_24h']} (24h)
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No high-confidence bullish plays found")

        with col2:
            st.markdown("### 📉 PUT Options (Bearish)")
            if put_opportunities:
                for stock in put_opportunities:
                    confidence_bar = "🔴" * max(1, int(stock['confidence'] * 10))
                    price_display = f"${stock['current_price']:.2f}" if stock['current_price'] > 0 else "N/A"
                    st.markdown(f"""
                    <div class="metric-box">
                        <strong>{stock['ticker']}</strong> - {price_display}<br>
                        Confidence: {confidence_bar} ({stock['confidence']:.2f})<br>
                        Change: <span class="{'price-up' if stock['change_percent'] > 0 else 'price-down'}">{stock['change_percent']:.1f}%</span><br>
                        Mentions: {stock['mention_count_24h']} (24h)
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No high-confidence bearish plays found")

    def render_recent_posts(self, data):
        """Render recent posts with tickers"""
        st.markdown("## 📋 Recent WSB Activity")

        recent_posts = data.get('recent_posts', [])

        if recent_posts:
            # Show top 15 most recent posts with tickers
            ticker_posts = [post for post in recent_posts if post.get('tickers')]
            ticker_posts.sort(key=lambda x: x.get('created_utc', 0), reverse=True)

            if ticker_posts:
                for i, post in enumerate(ticker_posts[:15]):
                    timestamp = datetime.fromtimestamp(post.get('created_utc', 0), tz=timezone.utc)
                    time_ago = datetime.now(timezone.utc) - timestamp

                    if time_ago.total_seconds() < 3600:
                        time_str = f"{int(time_ago.total_seconds() / 60)} min ago"
                    else:
                        time_str = f"{int(time_ago.total_seconds() / 3600)} hrs ago"

                    tickers = ', '.join(post.get('tickers', []))

                    # Create Reddit URL from permalink
                    permalink = post.get('permalink', '')
                    reddit_url = f"https://reddit.com{permalink}" if permalink else "#"

                    # Truncate title for display
                    title = post.get('title', 'No title')
                    display_title = title[:80] + '...' if len(title) > 80 else title

                    st.markdown(f"""
                    <div class="ticker-item">
                    <a href="{reddit_url}" target="_blank" style="color: #FF6B6B; text-decoration: none;">
                    <strong>{display_title}</strong>
                    </a><br>
                    🎯 <strong>{tickers}</strong> | ⭐ {post.get('score', 0)} | ⏰ {time_str} | 
                    <a href="{reddit_url}" target="_blank" style="color: #4ECDC4; font-size: 0.9em;">🔗 View Thread</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🔍 No recent posts with tickers found")
        else:
            st.info("🔍 No recent posts available")

    def render_sidebar(self, data):
        """Render optimized sidebar with rate limiting"""
        st.sidebar.markdown("## 💰 Trading Control Panel")

        # Auto-refresh with longer intervals
        refresh_options = {
            "Disabled": 0,
            "Every 5 minutes": 300,
            "Every 10 minutes": 600,
            "Every 15 minutes": 900
        }

        selected_refresh = st.sidebar.selectbox(
            "🔄 Auto-refresh interval",
            list(refresh_options.keys()),
            index=0  # Default to disabled
        )

        auto_refresh_seconds = refresh_options[selected_refresh]

        if st.sidebar.button("🔄 Refresh Data"):
            # Clear caches
            self.firebase_manager.clear_cache()
            st.cache_data.clear()
            st.rerun()

        st.sidebar.markdown("---")

        # Cache information
        cache_info = data.get('cache_info', {})
        if cache_info:
            st.sidebar.markdown("## 📊 Cache Status")
            st.sidebar.markdown(f"Valid queries: {cache_info.get('valid_cached_queries', 0)}")
            st.sidebar.markdown(f"Total cached: {cache_info.get('total_cached_queries', 0)}")

            if st.sidebar.button("🗑️ Clear Cache"):
                self.firebase_manager.clear_cache()
                st.cache_data.clear()
                st.sidebar.success("Cache cleared!")

        st.sidebar.markdown("---")

        # Market overview
        st.sidebar.markdown("## 📊 Market Overview")
        st.sidebar.metric("Total Opportunities", data['total_tickers'])
        st.sidebar.metric("Hot Stocks", len(data['hot_stocks']))
        st.sidebar.metric("Bullish Sentiment", len(data['bullish_plays']))
        st.sidebar.metric("Bearish Sentiment", len(data['bearish_plays']))

        # Top movers
        st.sidebar.markdown("## 🚀 Top Movers")
        top_movers = sorted(data['opportunities'][:10],
                          key=lambda x: abs(x['change_percent']) if x['change_percent'] is not None else 0,
                          reverse=True)

        for mover in top_movers[:5]:
            if mover['change_percent'] != 0:
                change_color = "🟢" if mover['change_percent'] > 0 else "🔴"
                st.sidebar.markdown(f"{change_color} **{mover['ticker']}**: {mover['change_percent']:+.1f}%")

        return auto_refresh_seconds

    def run(self):
        """Main dashboard run function"""
        # Header
        st.markdown('<h1 class="main-header">💰 WSB Options Trader Dashboard</h1>', unsafe_allow_html=True)

        # Get trading data (cached)
        with st.spinner("Loading trading opportunities..."):
            data = self.get_trading_opportunities()

        # Render cache info
        self.render_cache_info(data.get('cache_info', {}))

        # Render sidebar
        auto_refresh_seconds = self.render_sidebar(data)

        # Main content
        self.render_trading_metrics(data)
        st.markdown("---")

        self.render_hot_opportunities(data)
        st.markdown("---")

        self.render_options_opportunities(data)
        st.markdown("---")

        self.render_recent_posts(data)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        💰 Live Trading Intelligence • 🤖 Powered by WSB Sentiment • 📊 Optimized for Low API Usage<br>
        ⚠️ Not financial advice. Trade at your own risk.
        </div>
        """, unsafe_allow_html=True)

        # Auto-refresh with longer intervals
        if auto_refresh_seconds > 0:
            time.sleep(auto_refresh_seconds)
            st.rerun()


def main():
    """Main function"""
    dashboard = OptimizedTradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()