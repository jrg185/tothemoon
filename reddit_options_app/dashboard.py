"""
Enhanced WSB Options Trading Dashboard
Integrates AI Analysis + ML Forecasting with existing functionality
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

# Advanced Analytics (optional imports)
try:
    from ml.enhanced_trading_analyst import EnhancedTradingAnalyst
    from ml.ml_price_forecaster import MLPriceForecaster
    ADVANCED_ANALYTICS_AVAILABLE = True
except ImportError as e:
    ADVANCED_ANALYTICS_AVAILABLE = False
    print(f"⚠️ Advanced analytics not available: {e}")

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
    
    .advanced-toggle {
        background: linear-gradient(135deg, #2a2a5a, #3d3d7d);
        border: 1px solid #5555FF;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class EnhancedTradingDashboard:
    """Enhanced trading dashboard with optional AI analysis and ML forecasting"""

    def __init__(self):
        self.firebase_manager = FirebaseManager()
        self.alpha_vantage_key = FINANCIAL_APIS.get('alpha_vantage')
        self.finnhub_key = FINANCIAL_APIS.get('finnhub')

        # Price API caching
        self.price_cache = {}
        self.price_cache_duration = 300  # 5 minutes
        self.last_price_requests = {}

        # Advanced analytics (optional)
        self.ai_analyst = None
        self.ml_forecaster = None
        self.advanced_cache = {}
        self.advanced_cache_duration = 900  # 15 minutes

        # Initialize advanced analytics if available
        if ADVANCED_ANALYTICS_AVAILABLE:
            try:
                self.ai_analyst = EnhancedTradingAnalyst()
                self.ml_forecaster = MLPriceForecaster()
                logging.info("Advanced analytics initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize advanced analytics: {e}")
                self.ai_analyst = None
                self.ml_forecaster = None

    def get_stock_price_data(self, ticker: str) -> dict:
        """Get current price data with caching (unchanged)"""
        cache_key = ticker.upper()
        current_time = time.time()

        if (cache_key in self.price_cache and
            current_time - self.price_cache[cache_key]['timestamp'] < self.price_cache_duration):
            return self.price_cache[cache_key]['data']

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
                price_data = {
                    'current_price': data.get('c') or 0,
                    'change': data.get('d') or 0,
                    'change_percent': data.get('dp') or 0,
                    'high': data.get('h') or 0,
                    'low': data.get('l') or 0,
                    'open': data.get('o') or 0,
                    'prev_close': data.get('pc') or 0
                }

                self.price_cache[cache_key] = {
                    'data': price_data,
                    'timestamp': current_time
                }
                self.last_price_requests[cache_key] = current_time

                return price_data

        except Exception as e:
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

    def get_enhanced_analysis(self, ticker: str, sentiment_data: Dict, enable_advanced: bool = False) -> Dict:
        """Get enhanced analysis for a ticker (AI + ML if enabled)"""

        if not enable_advanced or not self.ai_analyst:
            return {}

        cache_key = f"enhanced_{ticker}_{int(time.time() / self.advanced_cache_duration)}"

        if cache_key in self.advanced_cache:
            return self.advanced_cache[cache_key]

        try:
            # Get AI analysis
            ai_analysis = self.ai_analyst.analyze_trading_opportunity(
                ticker=ticker,
                sentiment_data=sentiment_data
            )

            # Get ML forecast
            ml_forecast = {}
            if self.ml_forecaster:
                try:
                    # Get historical data
                    historical_data = self.ml_forecaster.get_historical_data(ticker, days=100)

                    if not historical_data.empty:
                        # Train models if needed (lightweight for real-time)
                        if ticker not in self.ml_forecaster.models:
                            train_result = self.ml_forecaster.train_models(ticker)

                        # Make prediction
                        if ticker in self.ml_forecaster.models:
                            ml_forecast = self.ml_forecaster.predict_price_movement(
                                ticker=ticker,
                                current_data=historical_data,
                                sentiment_data=[sentiment_data]
                            )
                except Exception as e:
                    logging.warning(f"ML forecast failed for {ticker}: {e}")
                    ml_forecast = {'error': str(e)}

            result = {
                'ai_analysis': ai_analysis,
                'ml_forecast': ml_forecast,
                'timestamp': datetime.now().isoformat()
            }

            # Cache result
            self.advanced_cache[cache_key] = result
            return result

        except Exception as e:
            logging.error(f"Enhanced analysis failed for {ticker}: {e}")
            return {'error': str(e)}

    @st.cache_data(ttl=300)
    def get_trading_opportunities(_self, enable_advanced: bool = False):
        """Get comprehensive trading data with optional advanced analytics"""
        try:
            fm = _self.firebase_manager

            recent_posts = fm.get_recent_posts(limit=200, hours=24, use_cache=True)
            trending_24h = fm.get_trending_tickers(hours=24, min_mentions=2, use_cache=True)
            trending_1h = fm.get_trending_tickers(hours=1, min_mentions=1, use_cache=True)
            sentiment_overview = fm.get_sentiment_overview(hours=24, use_cache=True)

            enhanced_opportunities = []

            # Limit to fewer tickers if advanced analysis is enabled
            max_items = 10 if not enable_advanced else 5
            top_sentiment_items = sentiment_overview[:max_items]

            for sentiment_item in top_sentiment_items:
                ticker = sentiment_item.get('ticker')
                if ticker:
                    # Get price data
                    price_data = _self.get_stock_price_data(ticker)

                    # Find in trending data
                    trending_info = next((t for t in trending_24h if t.get('ticker') == ticker), {})
                    recent_trending = next((t for t in trending_1h if t.get('ticker') == ticker), {})

                    # Calculate basic opportunity score
                    opportunity_score = _self.calculate_opportunity_score(
                        sentiment_item, price_data, trending_info, recent_trending
                    )

                    # Get enhanced analysis if enabled
                    enhanced_analysis = {}
                    if enable_advanced:
                        enhanced_analysis = _self.get_enhanced_analysis(ticker, sentiment_item, True)

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
                        'price_data': price_data,
                        'enhanced_analysis': enhanced_analysis  # This will be empty if not enabled
                    })

            enhanced_opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)

            return {
                'opportunities': enhanced_opportunities,
                'total_tickers': len(enhanced_opportunities),
                'hot_stocks': [op for op in enhanced_opportunities if op['opportunity_score'] > 50],
                'bullish_plays': [op for op in enhanced_opportunities if op['sentiment'] == 'bullish' and op['confidence'] > 0.5],
                'bearish_plays': [op for op in enhanced_opportunities if op['sentiment'] == 'bearish' and op['confidence'] > 0.5],
                'momentum_plays': [op for op in enhanced_opportunities if op['mention_count_1h'] > 0 or abs(op['change_percent']) > 1],
                'recent_posts': recent_posts[:25],
                'cache_info': fm.get_cache_stats(),
                'quota_status': fm.get_quota_status(),
                'advanced_enabled': enable_advanced
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
                'cache_info': {},
                'quota_status': {},
                'advanced_enabled': False
            }

    def calculate_opportunity_score(self, sentiment_data, price_data, trending_data, recent_trending):
        """Calculate a trading opportunity score (0-100) - unchanged"""
        score = 0

        try:
            confidence = float(sentiment_data.get('confidence', 0))
            sentiment = sentiment_data.get('sentiment', 'neutral')

            if sentiment in ['bullish', 'bearish']:
                score += confidence * 30

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

    def render_enhanced_opportunity_card(self, stock: Dict):
        """Render enhanced opportunity card with optional AI/ML analysis"""

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

        # Card class based on sentiment
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

                        # Options strategy
                        options_strategy = ai_analysis.get('options_strategy', {})
                        if options_strategy:
                            st.markdown("**Options Strategy:**")
                            st.markdown(f"• **Play:** {options_strategy.get('recommended_play', 'N/A')}")
                            st.markdown(f"• **Strike:** {options_strategy.get('strike_selection', 'N/A')}")
                            st.markdown(f"• **Expiration:** {options_strategy.get('expiration', 'N/A')}")

                    with col2:
                        st.markdown("**Key Insights:**")
                        catalysts = ai_analysis.get('key_catalysts', [])
                        for catalyst in catalysts[:3]:
                            st.markdown(f"• {catalyst}")

                        st.markdown("**Risk Factors:**")
                        risks = ai_analysis.get('risk_factors', [])
                        for risk in risks[:3]:
                            st.markdown(f"• {risk}")

                    # Executive summary
                    summary = ai_analysis.get('executive_summary', '')
                    if summary:
                        st.markdown(f"**Executive Summary:** {summary}")

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

                    # Individual predictions
                    individual_preds = ml_forecast.get('individual_predictions', {})
                    if individual_preds:
                        st.markdown("**Individual Model Predictions:**")
                        pred_cols = st.columns(len(individual_preds))
                        for i, (model, pred) in enumerate(individual_preds.items()):
                            with pred_cols[i]:
                                st.metric(model.replace('_', ' ').title(), f"{pred:+.2%}")

        # Show message if advanced analysis was requested but not available
        elif enhanced_analysis and 'error' in enhanced_analysis:
            st.warning(f"⚠️ Advanced analysis unavailable for {stock['ticker']}: {enhanced_analysis['error']}")

    def render_hot_opportunities(self, data):
        """Render top trading opportunities with enhanced cards"""
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
                self.render_enhanced_opportunity_card(stock)

    def render_trading_metrics(self, data):
        """Render key trading metrics with advanced indicators"""
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

        # Advanced metrics if enabled
        if data.get('advanced_enabled'):
            st.markdown("### 🤖 Advanced Analytics Summary")

            col1, col2, col3 = st.columns(3)

            opportunities = data.get('opportunities', [])

            # Count AI recommendations
            ai_strong_buys = len([op for op in opportunities
                                if op.get('enhanced_analysis', {}).get('ai_analysis', {}).get('overall_rating') in ['STRONG_BUY', 'BUY']])

            # Count ML positive predictions
            ml_positive = len([op for op in opportunities
                             if op.get('enhanced_analysis', {}).get('ml_forecast', {}).get('predicted_return', 0) > 0])

            # Count high confidence plays
            high_confidence = len([op for op in opportunities
                                 if op.get('enhanced_analysis', {}).get('ai_analysis', {}).get('confidence_score', 0) > 0.7])

            with col1:
                st.metric("🤖 AI Buy Signals", ai_strong_buys)

            with col2:
                st.metric("📊 ML Positive Forecasts", ml_positive)

            with col3:
                st.metric("🎯 High Confidence Plays", high_confidence)

    def render_options_opportunities(self, data):
        """Render specific options trading opportunities - unchanged"""
        st.markdown("## 💰 Options Trading Opportunities")

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
        """Render recent posts with tickers - unchanged"""
        st.markdown("## 📋 Recent WSB Activity")

        recent_posts = data.get('recent_posts', [])

        if recent_posts:
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
                    permalink = post.get('permalink', '')
                    reddit_url = f"https://reddit.com{permalink}" if permalink else "#"
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
        """Render enhanced sidebar with advanced analytics toggle"""
        st.sidebar.markdown("## 💰 Trading Control Panel")

        # Advanced Analytics Toggle
        if ADVANCED_ANALYTICS_AVAILABLE:
            st.sidebar.markdown("""
            <div class="advanced-toggle">
            <h4>🚀 Advanced Analytics</h4>
            <p>Enable AI Analysis + ML Forecasting</p>
            </div>
            """, unsafe_allow_html=True)

            enable_advanced = st.sidebar.checkbox(
                "🤖 Enable AI + ML Analysis",
                value=False,
                help="Provides detailed AI recommendations and ML price forecasts (slower)"
            )

            if enable_advanced:
                max_tickers = st.sidebar.slider(
                    "Max Tickers to Analyze",
                    1, 5, 3,
                    help="More tickers = longer analysis time"
                )
                st.session_state.max_tickers = max_tickers
            else:
                st.session_state.max_tickers = 10
        else:
            enable_advanced = False
            st.sidebar.warning("⚠️ Advanced analytics not available. Install ML dependencies to enable.")

        st.session_state.enable_advanced = enable_advanced

        # Auto-refresh options
        refresh_options = {
            "Disabled": 0,
            "Every 5 minutes": 300,
            "Every 15 minutes": 900,
            "Every 30 minutes": 1800,
            "Every 1 hour": 3600
        }

        selected_refresh = st.sidebar.selectbox(
            "🔄 Auto-refresh interval",
            list(refresh_options.keys()),
            index=0
        )

        auto_refresh_seconds = refresh_options[selected_refresh]

        # Refresh controls
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
                st.sidebar.success("All caches cleared!")

        st.sidebar.markdown("---")

        # System status
        quota_status = data.get('quota_status', {})
        if quota_status:
            reads_today = quota_status.get('reads_today', 0)
            daily_limit = quota_status.get('daily_limit', 35000)

            st.sidebar.markdown("### 📊 System Status")
            st.sidebar.metric("Firebase Reads Today", f"{reads_today:,}")
            st.sidebar.metric("Daily Limit", f"{daily_limit:,}")

            usage_percent = (reads_today / daily_limit) * 100
            st.sidebar.progress(usage_percent / 100)

        st.sidebar.markdown("---")

        # Market overview
        st.sidebar.markdown("### 📊 Market Overview")
        st.sidebar.metric("Total Opportunities", data['total_tickers'])
        st.sidebar.metric("Hot Stocks", len(data['hot_stocks']))
        st.sidebar.metric("Bullish Sentiment", len(data['bullish_plays']))
        st.sidebar.metric("Bearish Sentiment", len(data['bearish_plays']))

        # Advanced status
        if enable_advanced:
            st.sidebar.markdown("### 🤖 Advanced Status")
            ai_status = "✅ Ready" if self.ai_analyst else "❌ Failed"
            ml_status = "✅ Ready" if self.ml_forecaster else "❌ Failed"
            st.sidebar.metric("AI Analyst", ai_status)
            st.sidebar.metric("ML Forecaster", ml_status)

        return auto_refresh_seconds

    # Add this method to your existing dashboard.py class (EnhancedTradingDashboard)

    def get_automated_analysis_results(self) -> List[Dict]:
        """Get latest automated analysis results from Firebase"""

        try:
            # Get recent automated analysis results
            recent_analyses = self.firebase_manager.query_documents(
                collection_name='automated_analysis',
                order_by='analysis_timestamp',
                limit=20,  # Get recent results
                desc=True,
                use_cache=True
            )

            # Group by ticker to get latest result for each
            latest_by_ticker = {}
            for analysis in recent_analyses:
                ticker = analysis.get('ticker')
                if ticker and ticker not in latest_by_ticker:
                    latest_by_ticker[ticker] = analysis

            # Convert to list and sort by analysis timestamp
            results = list(latest_by_ticker.values())
            results.sort(key=lambda x: x.get('analysis_timestamp', ''), reverse=True)

            return results[:5]  # Return top 5 most recent

        except Exception as e:
            logger.error(f"Error getting automated analysis results: {e}")
            return []

    def render_automated_analysis_section(self):
        """Render automated analysis results section"""

        st.markdown("## 🤖 Automated AI + ML Analysis")
        st.markdown("*Analysis runs automatically every 15 minutes on top trending tickers*")

        # Get automated results
        automated_results = self.get_automated_analysis_results()

        if not automated_results:
            st.info("🔄 No automated analysis results yet. Analysis runs after each scraping cycle (every 15 minutes).")
            return

        # Show last analysis time
        latest_analysis = automated_results[0] if automated_results else None
        if latest_analysis:
            analysis_time = latest_analysis.get('analysis_timestamp', '')
            if analysis_time:
                try:
                    from datetime import datetime
                    analysis_dt = datetime.fromisoformat(analysis_time.replace('Z', '+00:00'))
                    time_ago = datetime.now(timezone.utc) - analysis_dt

                    if time_ago.total_seconds() < 3600:
                        time_str = f"{int(time_ago.total_seconds() / 60)} minutes ago"
                    else:
                        time_str = f"{int(time_ago.total_seconds() / 3600)} hours ago"

                    st.markdown(f"**🕐 Last automated analysis:** {time_str}")
                except:
                    st.markdown(f"**🕐 Last automated analysis:** {analysis_time}")

        # Display automated analysis cards
        for i, result in enumerate(automated_results):
            ticker = result.get('ticker', 'UNKNOWN')
            ai_analysis = result.get('ai_analysis', {})
            ml_forecast = result.get('ml_forecast', {})
            trending_data = result.get('trending_data', {})

            # Skip if no meaningful analysis
            if not ai_analysis or 'error' in ai_analysis:
                continue

            with st.expander(f"🎯 {ticker} - Automated Analysis", expanded=(i == 0)):

                # Create columns for layout
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.markdown("### 🤖 AI Analysis")

                    # AI Rating
                    overall_rating = ai_analysis.get('overall_rating', 'HOLD')
                    confidence_score = ai_analysis.get('confidence_score', 0)
                    target_price = ai_analysis.get('target_price', 0)

                    # Color coding
                    if overall_rating in ['STRONG_BUY', 'BUY']:
                        rating_color = '🟢'
                    elif overall_rating == 'HOLD':
                        rating_color = '🟡'
                    else:
                        rating_color = '🔴'

                    st.markdown(f"**Rating:** {rating_color} {overall_rating}")
                    st.markdown(f"**Confidence:** {confidence_score:.1%}")
                    st.markdown(f"**Target Price:** ${target_price:.2f}")

                    # Options strategy
                    options_strategy = ai_analysis.get('options_strategy', {})
                    if options_strategy:
                        play = options_strategy.get('recommended_play', 'N/A')
                        st.markdown(f"**Options Play:** {play.upper()}")

                with col2:
                    st.markdown("### 📊 ML Forecast")

                    if ml_forecast and 'error' not in ml_forecast:
                        predicted_change = ml_forecast.get('price_change_pct', 0)
                        ml_confidence = ml_forecast.get('confidence', 0)
                        direction = ml_forecast.get('direction', 'neutral')
                        magnitude = ml_forecast.get('magnitude', 'low')

                        # Direction emoji
                        direction_emoji = '📈' if direction == 'up' else '📉' if direction == 'down' else '➡️'

                        st.markdown(f"**Prediction:** {direction_emoji} {predicted_change:+.2f}%")
                        st.markdown(f"**ML Confidence:** {ml_confidence:.1%}")
                        st.markdown(f"**Magnitude:** {magnitude.title()}")

                        # Current vs predicted price
                        current_price = ml_forecast.get('current_price', 0)
                        predicted_price = ml_forecast.get('predicted_price', 0)
                        if current_price > 0 and predicted_price > 0:
                            st.markdown(f"**Current:** ${current_price:.2f}")
                            st.markdown(f"**Predicted:** ${predicted_price:.2f}")
                    else:
                        st.markdown("*ML forecast unavailable*")

                with col3:
                    st.markdown("### 📈 Trending Data")

                    # Trending information
                    mentions_1h = trending_data.get('mentions_1h', 0)
                    mentions_24h = trending_data.get('mentions_24h', 0)
                    combined_score = trending_data.get('combined_score', 0)

                    st.markdown(f"**Mentions (1h):** {mentions_1h}")
                    st.markdown(f"**Mentions (24h):** {mentions_24h}")
                    st.markdown(f"**Trending Score:** {combined_score:.1f}")

                    # Reddit mentions
                    reddit_mentions = result.get('reddit_mentions_count', 0)
                    st.markdown(f"**Reddit Posts:** {reddit_mentions}")

                    # Analysis duration
                    duration = result.get('analysis_duration_seconds', 0)
                    if duration > 0:
                        st.markdown(f"**Analysis Time:** {duration:.1f}s")

                # Executive Summary (full width)
                summary = ai_analysis.get('executive_summary', '')
                if summary:
                    st.markdown("### 📋 Executive Summary")
                    st.markdown(f"*{summary}*")

                # Key insights
                col1, col2 = st.columns(2)

                with col1:
                    catalysts = ai_analysis.get('key_catalysts', [])
                    if catalysts:
                        st.markdown("**🚀 Key Catalysts:**")
                        for catalyst in catalysts[:3]:
                            st.markdown(f"• {catalyst}")

                with col2:
                    risks = ai_analysis.get('risk_factors', [])
                    if risks:
                        st.markdown("**⚠️ Risk Factors:**")
                        for risk in risks[:3]:
                            st.markdown(f"• {risk}")

    def render_automated_analysis_status(self):
        """Render status of automated analysis system"""

        st.sidebar.markdown("### 🤖 Automated Analysis")

        # Check if automated analysis results exist
        automated_results = self.get_automated_analysis_results()

        if automated_results:
            st.sidebar.metric("Latest Analysis", f"{len(automated_results)} tickers")

            # Show last analysis time
            latest_analysis = automated_results[0]
            analysis_time = latest_analysis.get('analysis_timestamp', '')

            if analysis_time:
                try:
                    from datetime import datetime
                    analysis_dt = datetime.fromisoformat(analysis_time.replace('Z', '+00:00'))
                    time_ago = datetime.now(timezone.utc) - analysis_dt

                    if time_ago.total_seconds() < 3600:
                        time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
                    else:
                        time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"

                    st.sidebar.metric("Last Update", time_str)
                except:
                    st.sidebar.metric("Last Update", "Recently")

            # Show top rated ticker
            top_ratings = []
            for result in automated_results:
                ai_analysis = result.get('ai_analysis', {})
                if ai_analysis and 'error' not in ai_analysis:
                    rating = ai_analysis.get('overall_rating', 'HOLD')
                    confidence = ai_analysis.get('confidence_score', 0)
                    ticker = result.get('ticker', '')

                    if rating in ['STRONG_BUY', 'BUY'] and confidence > 0.7:
                        top_ratings.append((ticker, rating, confidence))

            if top_ratings:
                top_ticker = max(top_ratings, key=lambda x: x[2])
                st.sidebar.metric("Top AI Pick", f"{top_ticker[0]} ({top_ticker[1]})")
        else:
            st.sidebar.info("⏳ Waiting for automated analysis...")
            st.sidebar.markdown("*Analysis runs every 15 minutes*")

    # Updated run method to include automated analysis section
    def run_enhanced_with_automation(self):
        """Enhanced run method that includes automated analysis display"""

        # Header
        st.markdown('<h1 class="main-header">💰 WSB Options Trader</h1>', unsafe_allow_html=True)

        # Subtitle
        st.markdown("""
        <p style="text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 2rem;">
        🤖 Automated AI + ML Analysis Every 15 Minutes • Enhanced with Real-time Intelligence
        </p>
        """, unsafe_allow_html=True)

        # Get trading data
        enable_advanced = st.session_state.get('enable_advanced', False)

        with st.spinner(
                "Loading trading opportunities..." if not enable_advanced else "Loading with advanced analysis..."):
            data = self.get_trading_opportunities(enable_advanced)

        # Render sidebar (with automated analysis status)
        auto_refresh_seconds = self.render_sidebar(data)
        self.render_automated_analysis_status()  # Add automated analysis status

        # Main content

        # 1. Automated Analysis Section (NEW - TOP PRIORITY)
        self.render_automated_analysis_section()
        st.markdown("---")

        # 2. Trading Metrics
        self.render_trading_metrics(data)
        st.markdown("---")

        # 3. Hot Opportunities
        self.render_hot_opportunities(data)
        st.markdown("---")

        # 4. Options Opportunities
        self.render_options_opportunities(data)
        st.markdown("---")

        # 5. Recent Posts
        self.render_recent_posts(data)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        🤖 WSB Options Trading Intelligence • 🚀 Automated Analysis Every 15 Minutes<br>
        ⚠️ Not financial advice. Advanced analytics for educational purposes.
        </div>
        """, unsafe_allow_html=True)

        # Auto-refresh
        if auto_refresh_seconds > 0:
            time.sleep(auto_refresh_seconds)
            st.rerun()

    def run(self):
        """Enhanced run method with automated analysis"""
        # Header
        st.markdown('<h1 class="main-header">💰 WSB Options Trader</h1>', unsafe_allow_html=True)

        # Enhanced subtitle
        st.markdown("""
        <p style="text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 2rem;">
        🤖 Automated AI + ML Analysis Every 15 Minutes • Enhanced with Real-time Intelligence
        </p>
        """, unsafe_allow_html=True)

        # Get trading data
        enable_advanced = st.session_state.get('enable_advanced', False)

        with st.spinner("Loading trading opportunities..."):
            data = self.get_trading_opportunities(enable_advanced)

        # Render sidebar with automated analysis status
        auto_refresh_seconds = self.render_sidebar(data)
        self.render_automated_analysis_status()

        # 🟢 NEW: Automated Analysis Section (TOP PRIORITY)
        self.render_automated_analysis_section()
        st.markdown("---")

        # Existing content
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
        🤖 WSB Options Trading Intelligence • 🚀 Automated Analysis Every 15 Minutes<br>
        ⚠️ Not financial advice. Advanced analytics for educational purposes.
        </div>
        """, unsafe_allow_html=True)

        # Auto-refresh
        if auto_refresh_seconds > 0:
            time.sleep(auto_refresh_seconds)
            st.rerun()


def main():
    """Main function"""
    dashboard = EnhancedTradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()