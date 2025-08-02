"""
WSB Options Trading Dashboard - Production Version
Fixed import issues and infinite loop, ready for production use
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
import numpy as np
import logging

from data.firebase_manager import FirebaseManager
from config.settings import APP_CONFIG, FINANCIAL_APIS

# Advanced Analytics with better error handling
ADVANCED_ANALYTICS_AVAILABLE = False
AI_ANALYST_AVAILABLE = False
ML_FORECASTER_AVAILABLE = False

# Try to import AI analyst
try:
    from ml.enhanced_trading_analyst import EnhancedTradingAnalyst
    AI_ANALYST_AVAILABLE = True
    print("✅ AI Analyst available")
except ImportError as e:
    print(f"⚠️ AI Analyst not available: {e}")

# Try to import ML forecaster
try:
    from ml.ml_price_forecaster import MLPriceForecaster
    ML_FORECASTER_AVAILABLE = True
    print("✅ ML Forecaster available")
except ImportError as e:
    print(f"⚠️ ML Forecaster not available: {e}")

# Set overall advanced analytics availability
ADVANCED_ANALYTICS_AVAILABLE = AI_ANALYST_AVAILABLE or ML_FORECASTER_AVAILABLE

# Configure Streamlit
st.set_page_config(
    page_title="WSB Options Trader",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00FF88;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .enhanced-card {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        border: 2px solid #00FF88;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .ai-analysis {
        background: linear-gradient(135deg, #1a1a4a, #2d2d6d);
        border: 2px solid #4488FF;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .ml-forecast {
        background: linear-gradient(135deg, #1a4a1a, #2d6d2d);
        border: 2px solid #44FF88;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
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
    
    .ticker-item {
        background-color: #1E1E1E;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)


class ProductionTradingDashboard:
    """Production trading dashboard with fixed imports and no infinite loops"""

    def __init__(self):
        self.firebase_manager = FirebaseManager()
        self.finnhub_key = FINANCIAL_APIS.get('finnhub')

        # Price API caching
        self.price_cache = {}
        self.price_cache_duration = 300  # 5 minutes
        self.last_price_requests = {}

        # Advanced analytics (with safe initialization)
        self.ai_analyst = None
        self.ml_forecaster = None
        self.advanced_cache = {}
        self.advanced_cache_duration = 900  # 15 minutes

        # Initialize available components
        if AI_ANALYST_AVAILABLE:
            try:
                self.ai_analyst = EnhancedTradingAnalyst()
                logging.info("✅ AI analyst initialized")
            except Exception as e:
                logging.error(f"❌ AI analyst failed to initialize: {e}")
                self.ai_analyst = None

        if ML_FORECASTER_AVAILABLE:
            try:
                self.ml_forecaster = MLPriceForecaster()
                logging.info("✅ ML forecaster initialized")
            except Exception as e:
                logging.error(f"❌ ML forecaster failed to initialize: {e}")
                self.ml_forecaster = None

    def get_stock_price_data(self, ticker: str) -> dict:
        """Get current price data with caching"""
        cache_key = ticker.upper()
        current_time = time.time()

        # Check cache first
        if (cache_key in self.price_cache and
            current_time - self.price_cache[cache_key]['timestamp'] < self.price_cache_duration):
            return self.price_cache[cache_key]['data']

        # Rate limiting check
        if (cache_key in self.last_price_requests and
            current_time - self.last_price_requests[cache_key] < 30):
            return self._get_default_price_data()

        try:
            if not self.finnhub_key:
                return self._get_default_price_data()

            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()

                # Validate data
                if 'c' in data and data['c'] is not None and data['c'] > 0:
                    price_data = {
                        'current_price': data.get('c', 0),
                        'change': data.get('d', 0),
                        'change_percent': data.get('dp', 0),
                        'high': data.get('h', 0),
                        'low': data.get('l', 0),
                        'open': data.get('o', 0),
                        'prev_close': data.get('pc', 0)
                    }

                    # Cache successful result
                    self.price_cache[cache_key] = {
                        'data': price_data,
                        'timestamp': current_time
                    }
                    self.last_price_requests[cache_key] = current_time

                    return price_data

        except Exception as e:
            logging.warning(f"Price API failed for {ticker}: {e}")

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

    def get_enhanced_analysis(self, ticker: str, sentiment_data: Dict, enable_advanced: bool = False) -> Dict:
        """Get enhanced analysis for a ticker (with safe error handling)"""

        if not enable_advanced:
            return {}

        cache_key = f"enhanced_{ticker}_{int(time.time() / self.advanced_cache_duration)}"

        if cache_key in self.advanced_cache:
            return self.advanced_cache[cache_key]

        result = {}

        # AI Analysis
        if self.ai_analyst:
            try:
                ai_analysis = self.ai_analyst.analyze_trading_opportunity(
                    ticker=ticker,
                    sentiment_data=sentiment_data
                )
                result['ai_analysis'] = ai_analysis
            except Exception as e:
                logging.error(f"AI analysis failed for {ticker}: {e}")
                result['ai_analysis'] = {'error': str(e)}

        # ML Forecast
        if self.ml_forecaster:
            try:
                # Get historical data for ML prediction
                historical_data = self.ml_forecaster.get_historical_data(ticker, days=100)

                if not historical_data.empty:
                    ml_forecast = self.ml_forecaster.predict_price_movement(
                        ticker=ticker,
                        current_data=historical_data,
                        sentiment_data=[sentiment_data]
                    )
                    result['ml_forecast'] = ml_forecast
                else:
                    result['ml_forecast'] = {'error': 'No historical data available'}

            except Exception as e:
                logging.error(f"ML forecast failed for {ticker}: {e}")
                result['ml_forecast'] = {'error': str(e)}

        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()

        # Cache result
        self.advanced_cache[cache_key] = result
        return result

    @st.cache_data(ttl=300)
    def get_trading_opportunities(_self, enable_advanced: bool = False):
        """Get comprehensive trading data"""
        try:
            fm = _self.firebase_manager

            # Get data from Firebase
            recent_posts = fm.get_recent_posts(limit=100, hours=24, use_cache=True)
            trending_24h = fm.get_trending_tickers(hours=24, min_mentions=2, use_cache=True)
            trending_1h = fm.get_trending_tickers(hours=1, min_mentions=1, use_cache=True)
            sentiment_overview = fm.get_sentiment_overview(hours=24, use_cache=True)

            opportunities = []

            # Process sentiment data (limit to prevent API overload)
            max_items = 5 if enable_advanced else 10
            top_sentiment_items = sentiment_overview[:max_items]

            for sentiment_item in top_sentiment_items:
                ticker = sentiment_item.get('ticker')
                if ticker:
                    # Get price data
                    price_data = _self.get_stock_price_data(ticker)

                    # Find in trending data
                    trending_info = next((t for t in trending_24h if t.get('ticker') == ticker), {})
                    recent_trending = next((t for t in trending_1h if t.get('ticker') == ticker), {})

                    # Calculate opportunity score
                    opportunity_score = _self.calculate_opportunity_score(
                        sentiment_item, price_data, trending_info, recent_trending
                    )

                    # Get enhanced analysis if enabled and available
                    enhanced_analysis = {}
                    if enable_advanced and (AI_ANALYST_AVAILABLE or ML_FORECASTER_AVAILABLE):
                        enhanced_analysis = _self.get_enhanced_analysis(ticker, sentiment_item, True)

                    opportunity = {
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
                        'price_data': price_data,
                        'enhanced_analysis': enhanced_analysis
                    }

                    opportunities.append(opportunity)

            # Sort by opportunity score
            opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)

            return {
                'opportunities': opportunities,
                'total_tickers': len(opportunities),
                'hot_stocks': [op for op in opportunities if op['opportunity_score'] > 50],
                'bullish_plays': [op for op in opportunities if op['sentiment'] == 'bullish' and op['confidence'] > 0.5],
                'bearish_plays': [op for op in opportunities if op['sentiment'] == 'bearish' and op['confidence'] > 0.5],
                'momentum_plays': [op for op in opportunities if op['mention_count_1h'] > 0 or abs(op['change_percent']) > 1],
                'recent_posts': recent_posts[:25],
                'cache_info': fm.get_cache_stats(),
                'quota_status': fm.get_quota_status(),
                'advanced_enabled': enable_advanced,
                'ai_available': AI_ANALYST_AVAILABLE,
                'ml_available': ML_FORECASTER_AVAILABLE
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
                'error': str(e)
            }

    def calculate_opportunity_score(self, sentiment_data, price_data, trending_data, recent_trending):
        """Calculate a trading opportunity score (0-100)"""
        score = 0

        try:
            # Sentiment contribution (0-30 points)
            confidence = float(sentiment_data.get('confidence', 0))
            sentiment = sentiment_data.get('sentiment', 'neutral')

            if sentiment in ['bullish', 'bearish']:
                score += confidence * 30

            # Mentions contribution (0-35 points)
            mentions_24h = int(trending_data.get('mention_count', 0))
            mentions_1h = int(recent_trending.get('mention_count', 0)) if recent_trending else 0

            if mentions_24h > 10:
                score += 15
            elif mentions_24h > 5:
                score += 10
            elif mentions_24h > 2:
                score += 5

            if mentions_1h > 0:
                score += min(mentions_1h * 5, 20)

            # Price movement contribution (0-35 points)
            change_percent = price_data.get('change_percent', 0)
            if change_percent is not None:
                abs_change = abs(float(change_percent))
                if abs_change > 5:
                    score += 25
                elif abs_change > 3:
                    score += 15
                elif abs_change > 1:
                    score += 10

        except (ValueError, TypeError):
            pass

        return min(score, 100)

    def render_opportunity_card(self, stock: Dict):
        """Render enhanced opportunity card"""

        sentiment_emoji = {'bullish': '🐂', 'bearish': '🐻', 'neutral': '😐'}[stock['sentiment']]

        # Price styling
        if stock['change_percent'] > 0:
            price_class = "price-up"
            price_arrow = "📈"
        elif stock['change_percent'] < 0:
            price_class = "price-down"
            price_arrow = "📉"
        else:
            price_class = "price-neutral"
            price_arrow = "➖"

        # Card styling
        card_class = "hot-stock" if stock['sentiment'] == 'bullish' else "bearish-stock" if stock['sentiment'] == 'bearish' else "neutral-stock"
        price_display = f"${stock['current_price']:.2f}" if stock['current_price'] > 0 else "Price N/A"

        # Basic card
        st.markdown(f"""
        <div class="{card_class}">
            <h3>{sentiment_emoji} {stock['ticker']} - Score: {stock['opportunity_score']:.0f}/100</h3>
            <p><span class="{price_class}">{price_display} {price_arrow} {stock['change_percent']:.1f}%</span></p>
            <p><strong>Sentiment:</strong> {stock['sentiment'].title()} ({stock['confidence']:.2f} confidence)</p>
            <p><strong>Mentions:</strong> {stock['mention_count_24h']} (24h) | {stock['mention_count_1h']} (1h)</p>
            <p><strong>Basic Play:</strong> {'CALLS' if stock['sentiment'] == 'bullish' else 'PUTS' if stock['sentiment'] == 'bearish' else 'NEUTRAL'}</p>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced analysis (if available)
        enhanced_analysis = stock.get('enhanced_analysis', {})

        if enhanced_analysis and 'error' not in enhanced_analysis:
            with st.expander(f"🤖 Advanced AI + ML Analysis for {stock['ticker']}"):

                # AI Analysis section
                ai_analysis = enhanced_analysis.get('ai_analysis', {})
                if ai_analysis and 'error' not in ai_analysis:

                    st.markdown("""
                    <div class="ai-analysis">
                    <h4>🤖 AI Trading Analysis</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**AI Recommendation:**")
                        overall_rating = ai_analysis.get('overall_rating', 'HOLD')
                        confidence_score = ai_analysis.get('confidence_score', 0)
                        target_price = ai_analysis.get('target_price', 0)

                        st.markdown(f"• **Rating:** {overall_rating}")
                        st.markdown(f"• **Confidence:** {confidence_score:.1%}")
                        st.markdown(f"• **Target Price:** ${target_price:.2f}")

                    with col2:
                        # Key insights
                        catalysts = ai_analysis.get('key_catalysts', [])
                        if catalysts:
                            st.markdown("**Key Catalysts:**")
                            for catalyst in catalysts[:3]:
                                st.markdown(f"• {catalyst}")

                # ML Forecast section
                ml_forecast = enhanced_analysis.get('ml_forecast', {})
                if ml_forecast and 'error' not in ml_forecast:

                    st.markdown("""
                    <div class="ml-forecast">
                    <h4>📊 ML Price Forecast</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        price_change_pct = ml_forecast.get('price_change_pct', 0)
                        predicted_price = ml_forecast.get('predicted_price', 0)
                        st.metric(
                            "ML Predicted Price",
                            f"${predicted_price:.2f}",
                            delta=f"{price_change_pct:+.2f}%"
                        )

                    with col2:
                        direction = ml_forecast.get('direction', 'neutral')
                        magnitude = ml_forecast.get('magnitude', 'low')
                        st.metric(
                            "Direction & Magnitude",
                            direction.upper(),
                            delta=f"{magnitude} magnitude"
                        )

                    with col3:
                        ml_confidence = ml_forecast.get('confidence', 0)
                        st.metric(
                            "ML Confidence",
                            f"{ml_confidence:.1%}",
                            delta="Ensemble model"
                        )

    def render_hot_opportunities(self, data):
        """Render top trading opportunities"""
        st.markdown("## 🔥 Hot Trading Opportunities")

        hot_stocks = data['hot_stocks'][:6]

        if not hot_stocks:
            st.info("No high-scoring opportunities found. Showing top opportunities...")
            hot_stocks = data['opportunities'][:6]

        if not hot_stocks:
            st.info("No trading opportunities available. Check if sentiment analysis is running.")
            return

        # Create 2 columns for hot stocks
        col1, col2 = st.columns(2)

        for i, stock in enumerate(hot_stocks):
            col = col1 if i % 2 == 0 else col2

            with col:
                self.render_opportunity_card(stock)

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

        # Show what analytics are available
        if data.get('advanced_enabled'):
            st.markdown("### 🤖 Analytics Status")
            col1, col2, col3 = st.columns(3)

            with col1:
                ai_status = "✅ Available" if data.get('ai_available') else "❌ Unavailable"
                st.metric("AI Analyst", ai_status)

            with col2:
                ml_status = "✅ Available" if data.get('ml_available') else "❌ Unavailable"
                st.metric("ML Forecaster", ml_status)

            with col3:
                total_advanced = len([op for op in data.get('opportunities', [])
                                     if op.get('enhanced_analysis')])
                st.metric("Advanced Analyses", total_advanced)

    def render_sidebar(self, data):
        """Render sidebar with controls"""
        st.sidebar.markdown("## 💰 Trading Control Panel")

        # Advanced Analytics Toggle
        if ADVANCED_ANALYTICS_AVAILABLE:
            st.sidebar.markdown("### 🚀 Advanced Analytics")

            enable_advanced = st.sidebar.checkbox(
                "🤖 Enable AI + ML Analysis",
                value=False,
                help="Provides detailed AI recommendations and ML price forecasts"
            )

            if enable_advanced:
                max_tickers = st.sidebar.slider(
                    "Max Tickers to Analyze",
                    1, 5, 3,
                    help="More tickers = longer analysis time"
                )
                st.session_state.max_tickers = max_tickers

            # Show what's available
            if AI_ANALYST_AVAILABLE:
                st.sidebar.success("✅ AI Analyst Ready")
            if ML_FORECASTER_AVAILABLE:
                st.sidebar.success("✅ ML Forecaster Ready")

        else:
            enable_advanced = False
            st.sidebar.warning("⚠️ Advanced analytics not available")
            if not AI_ANALYST_AVAILABLE:
                st.sidebar.info("AI Analyst: Import failed")
            if not ML_FORECASTER_AVAILABLE:
                st.sidebar.info("ML Forecaster: Import failed")

        st.session_state.enable_advanced = enable_advanced

        # Manual refresh only (NO AUTO-REFRESH to prevent infinite loops)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Refresh Data"):
                self.firebase_manager.clear_cache()
                st.cache_data.clear()
                st.rerun()

        with col2:
            if st.button("🗑️ Clear Cache"):
                self.firebase_manager.clear_cache()
                st.cache_data.clear()
                self.advanced_cache.clear()
                st.sidebar.success("Caches cleared!")

        st.sidebar.markdown("---")

        # System status
        quota_status = data.get('quota_status', {})
        if quota_status:
            st.sidebar.markdown("### 📊 System Status")
            reads_today = quota_status.get('reads_today', 0)
            daily_limit = quota_status.get('daily_limit', 35000)

            st.sidebar.metric("Firebase Reads Today", f"{reads_today:,}")
            st.sidebar.metric("Daily Limit", f"{daily_limit:,}")

            usage_percent = (reads_today / daily_limit) * 100
            st.sidebar.progress(usage_percent / 100)

        # Market overview
        st.sidebar.markdown("### 📊 Market Overview")
        st.sidebar.metric("Total Opportunities", data['total_tickers'])
        st.sidebar.metric("Hot Stocks", len(data['hot_stocks']))
        st.sidebar.metric("Bullish Sentiment", len(data['bullish_plays']))
        st.sidebar.metric("Bearish Sentiment", len(data['bearish_plays']))

        return False  # NO AUTO-REFRESH

    def run(self):
        """Run the production dashboard"""

        # Header
        st.markdown('<h1 class="main-header">💰 WSB Options Trader</h1>', unsafe_allow_html=True)

        st.markdown("""
        <p style="text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 2rem;">
        🚀 Real-time WSB Sentiment Analysis • Options Trading Intelligence
        </p>
        """, unsafe_allow_html=True)

        # Get trading data
        enable_advanced = st.session_state.get('enable_advanced', False)

        with st.spinner("Loading trading opportunities..."):
            data = self.get_trading_opportunities(enable_advanced)

        # Render sidebar
        auto_refresh = self.render_sidebar(data)

        # Main content
        if 'error' in data:
            st.error(f"Error loading data: {data['error']}")
            return

        # Show success message if we have data
        if data['total_tickers'] > 0:
            st.success(f"✅ Loaded {data['total_tickers']} trading opportunities from Firebase")

        # Trading metrics
        self.render_trading_metrics(data)
        st.markdown("---")

        # Hot opportunities
        self.render_hot_opportunities(data)
        st.markdown("---")

        # Footer
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        💰 WSB Options Trading Intelligence • Manual refresh to prevent loops<br>
        ⚠️ Not financial advice. For educational purposes only.
        </div>
        """, unsafe_allow_html=True)

        # IMPORTANT: NO AUTO-REFRESH - prevents infinite loops


def main():
    """Main function"""
    dashboard = ProductionTradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()