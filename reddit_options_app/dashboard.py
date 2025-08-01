"""
Trading-Focused WSB Sentiment Dashboard
Fixed version with post listings and better error handling
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
from data.firebase_manager import FirebaseManager
from config.settings import APP_CONFIG, FINANCIAL_APIS
import numpy as np

# Configure Streamlit
st.set_page_config(
    page_title="WSB Options Trader",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Trading-focused CSS
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
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    """Trading-focused dashboard for actionable opportunities"""

    def __init__(self):
        self.firebase_manager = FirebaseManager()
        self.alpha_vantage_key = FINANCIAL_APIS.get('alpha_vantage')
        self.finnhub_key = FINANCIAL_APIS.get('finnhub')

    def get_stock_price_data(self, ticker: str) -> dict:
        """Get current price and basic metrics for a ticker with better error handling"""
        try:
            if not self.finnhub_key:
                return self._get_default_price_data()

            # Use Finnhub for real-time data
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}"
            response = requests.get(url, timeout=3)

            if response.status_code == 200:
                data = response.json()
                return {
                    'current_price': data.get('c') or 0,
                    'change': data.get('d') or 0,
                    'change_percent': data.get('dp') or 0,
                    'high': data.get('h') or 0,
                    'low': data.get('l') or 0,
                    'open': data.get('o') or 0,
                    'prev_close': data.get('pc') or 0
                }
        except Exception as e:
            # Don't show warning for every ticker, just log
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

    def get_trading_opportunities(self):
        """Get comprehensive trading data with better error handling"""
        try:
            # Get sentiment data
            recent_posts = self.firebase_manager.get_recent_posts(limit=1000, hours=24)
            trending_24h = self.firebase_manager.get_trending_tickers(hours=24, min_mentions=2)
            trending_1h = self.firebase_manager.get_trending_tickers(hours=1, min_mentions=1)
            sentiment_overview = self.firebase_manager.get_sentiment_overview(hours=24)

            # Enhance with price data
            enhanced_opportunities = []

            for sentiment_item in sentiment_overview:
                ticker = sentiment_item.get('ticker')
                if ticker:
                    # Get price data
                    price_data = self.get_stock_price_data(ticker)

                    # Find in trending data
                    trending_info = next((t for t in trending_24h if t.get('ticker') == ticker), {})
                    recent_trending = next((t for t in trending_1h if t.get('ticker') == ticker), {})

                    # Calculate opportunity score
                    opportunity_score = self.calculate_opportunity_score(
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
                'hot_stocks': [op for op in enhanced_opportunities if op['opportunity_score'] > 50],  # Lowered threshold
                'bullish_plays': [op for op in enhanced_opportunities if op['sentiment'] == 'bullish' and op['confidence'] > 0.5],
                'bearish_plays': [op for op in enhanced_opportunities if op['sentiment'] == 'bearish' and op['confidence'] > 0.5],
                'momentum_plays': [op for op in enhanced_opportunities if op['mention_count_1h'] > 0 or abs(op['change_percent']) > 1],
                'recent_posts': recent_posts  # Add this for post listings
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
                'recent_posts': []
            }

    def calculate_opportunity_score(self, sentiment_data, price_data, trending_data, recent_trending):
        """Calculate a trading opportunity score (0-100) with safe math operations"""
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

            # Price movement factor (0-25 points) - FIXED
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
            # If any conversion fails, return minimum score
            pass

        return min(score, 100)

    def render_hot_opportunities(self, data):
        """Render top trading opportunities"""
        st.markdown("## 🔥 Hot Trading Opportunities")

        hot_stocks = data['hot_stocks'][:6]  # Top 6

        if not hot_stocks:
            st.info("No high-scoring opportunities found. Lowering threshold or waiting for more data...")
            # Show top opportunities anyway
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
        """Render recent posts with tickers - RESTORED FROM PREVIOUS VERSION"""
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

    def render_momentum_tracker(self, data):
        """Track momentum changes"""
        st.markdown("## ⚡ Momentum Tracker")

        momentum_stocks = data['momentum_plays'][:10]

        if momentum_stocks:
            df_data = []
            for stock in momentum_stocks:
                df_data.append({
                    'ticker': stock['ticker'],
                    'sentiment_score': stock['numerical_score'],
                    'mentions_1h': max(1, stock['mention_count_1h']),  # Avoid zero for bubble size
                    'price_change': stock['change_percent'],
                    'opportunity_score': stock['opportunity_score']
                })

            df = pd.DataFrame(df_data)

            # Create bubble chart
            fig = px.scatter(df,
                           x='price_change',
                           y='sentiment_score',
                           size='mentions_1h',
                           color='opportunity_score',
                           hover_name='ticker',
                           title='📊 Momentum vs Sentiment vs Price Movement',
                           labels={
                               'price_change': 'Price Change %',
                               'sentiment_score': 'Sentiment Score',
                               'opportunity_score': 'Opportunity Score'
                           },
                           color_continuous_scale='RdYlGn')

            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No momentum plays detected. Check back for updates!")

    def render_sidebar(self, data):
        """Render trading-focused sidebar"""
        st.sidebar.markdown("## 💰 Trading Control Panel")

        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)

        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()

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

        # Alerts
        st.sidebar.markdown("## 🚨 Trading Alerts")

        high_confidence_plays = [op for op in data['opportunities']
                               if op['confidence'] > 0.7 and
                               (op['mention_count_1h'] > 0 or op['mention_count_24h'] > 5)]

        if high_confidence_plays:
            for play in high_confidence_plays[:3]:
                st.sidebar.markdown(f"""
                <div class="alert-box">
                <strong>🔥 {play['ticker']}</strong><br>
                {play['sentiment'].upper()} - {play['confidence']:.0%} confidence<br>
                Recent mentions: {play['mention_count_1h']} (1h)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.sidebar.info("No high-confidence alerts")

        return auto_refresh

    def run(self):
        """Main dashboard run function"""
        # Header
        st.markdown('<h1 class="main-header">💰 WSB Options Trader Dashboard</h1>', unsafe_allow_html=True)

        # Get trading data
        with st.spinner("Loading trading opportunities..."):
            data = self.get_trading_opportunities()

        # Render sidebar
        auto_refresh = self.render_sidebar(data)

        # Main content
        self.render_trading_metrics(data)
        st.markdown("---")

        self.render_hot_opportunities(data)
        st.markdown("---")

        self.render_options_opportunities(data)
        st.markdown("---")

        self.render_momentum_tracker(data)
        st.markdown("---")

        # ADD BACK THE RECENT POSTS SECTION
        self.render_recent_posts(data)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        💰 Live Trading Intelligence • 🤖 Powered by WSB Sentiment • 📊 Real-time Market Data<br>
        ⚠️ Not financial advice. Trade at your own risk.
        </div>
        """, unsafe_allow_html=True)

        # Auto-refresh
        if auto_refresh:
            time.sleep(30)
            st.rerun()


def main():
    """Main function"""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()