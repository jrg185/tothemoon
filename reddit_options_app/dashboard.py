"""
CLEAN WSB Options Trading Dashboard
Simplified, trading-focused layout with intuitive advanced analytics
"""

import sys
from pathlib import Path
import json
import os

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

# Advanced Analytics availability check
ADVANCED_ANALYTICS_AVAILABLE = False
AI_ANALYST_AVAILABLE = False
ML_FORECASTER_AVAILABLE = False

try:
    from ml.enhanced_trading_analyst import EnhancedTradingAnalyst
    AI_ANALYST_AVAILABLE = True
except ImportError:
    pass

try:
    from ml.ml_price_forecaster import MLPriceForecaster
    ML_FORECASTER_AVAILABLE = True
except ImportError:
    pass

ADVANCED_ANALYTICS_AVAILABLE = AI_ANALYST_AVAILABLE or ML_FORECASTER_AVAILABLE

# Configure Streamlit
st.set_page_config(
    page_title="WSB Options Trader",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for cleaner look
)

# Clean, trading-focused CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00FF88;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .trading-card {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        border: 2px solid #00FF88;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.1);
    }
    
    .bearish-card {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        border: 2px solid #FF4444;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.1);
    }
    
    .neutral-card {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        border: 2px solid #888888;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(136, 136, 136, 0.1);
    }
    
    .price-up { color: #00FF88; font-weight: bold; font-size: 1.1em; }
    .price-down { color: #FF4444; font-weight: bold; font-size: 1.1em; }
    .price-neutral { color: #FFFFFF; font-size: 1.1em; }
    
    .ai-analysis-section {
        background: linear-gradient(135deg, #1a1a4a, #2d2d6d);
        border: 2px solid #4488FF;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .analytics-button {
        background: linear-gradient(135deg, #4488FF, #6699FF);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-size: 1.1em;
        font-weight: bold;
        margin: 10px 5px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(68, 136, 255, 0.3);
    }
    
    .tab-button {
        padding: 10px 20px;
        margin: 5px;
        border-radius: 20px;
        border: 2px solid #00FF88;
        background: transparent;
        color: #00FF88;
        font-weight: bold;
        cursor: pointer;
    }
    
    .tab-button-active {
        background: #00FF88;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)


class AccurateReadCounter:
    """Simple read counter (same as before)"""

    def __init__(self):
        self.counter_file = Path("logs/firebase_read_counter.json")
        self.counter_file.parent.mkdir(exist_ok=True)
        self.load_counter()

    def load_counter(self):
        try:
            if self.counter_file.exists():
                with open(self.counter_file, 'r') as f:
                    data = json.load(f)
                self.daily_reads = data.get('daily_reads', 0)
                self.hourly_reads = data.get('hourly_reads', 0)
                self.last_daily_reset = data.get('last_daily_reset', time.time())
                self.last_hourly_reset = data.get('last_hourly_reset', time.time())
            else:
                self.reset_counters()
        except Exception:
            self.reset_counters()

    def reset_counters(self):
        self.daily_reads = 0
        self.hourly_reads = 0
        self.last_daily_reset = time.time()
        self.last_hourly_reset = time.time()
        self.save_counter()

    def save_counter(self):
        try:
            data = {
                'daily_reads': self.daily_reads,
                'hourly_reads': self.hourly_reads,
                'last_daily_reset': self.last_daily_reset,
                'last_hourly_reset': self.last_hourly_reset,
                'last_updated': time.time()
            }
            with open(self.counter_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def check_resets(self):
        current_time = time.time()
        if current_time - self.last_daily_reset > 86400:
            self.daily_reads = 0
            self.last_daily_reset = current_time
        if current_time - self.last_hourly_reset > 3600:
            self.hourly_reads = 0
            self.last_hourly_reset = current_time
        self.save_counter()

    def increment_read(self, is_cache_hit: bool = False):
        self.check_resets()
        if not is_cache_hit:
            self.daily_reads += 1
            self.hourly_reads += 1
        self.save_counter()

    def get_status(self) -> Dict:
        self.check_resets()
        return {
            'daily_reads': self.daily_reads,
            'hourly_reads': self.hourly_reads,
            'daily_remaining': max(0, 35000 - self.daily_reads),
            'quota_healthy': self.daily_reads < 30000
        }


class CleanTradingDashboard:
    """Clean, trading-focused dashboard"""

    def __init__(self):
        self.firebase_manager = FirebaseManager()
        self.finnhub_key = FINANCIAL_APIS.get('finnhub')
        self.read_counter = AccurateReadCounter()

        # Price API caching
        self.price_cache = {}
        self.price_cache_duration = 300

        # Advanced analytics
        self.ai_analyst = None
        self.ml_forecaster = None
        self.advanced_cache = {}

        if AI_ANALYST_AVAILABLE:
            try:
                self.ai_analyst = EnhancedTradingAnalyst()
            except Exception:
                self.ai_analyst = None

        if ML_FORECASTER_AVAILABLE:
            try:
                self.ml_forecaster = MLPriceForecaster()
            except Exception:
                self.ml_forecaster = None

    def get_stock_price_data(self, ticker: str) -> dict:
        """Get price data with caching"""
        cache_key = ticker.upper()
        current_time = time.time()

        if (cache_key in self.price_cache and
            current_time - self.price_cache[cache_key]['timestamp'] < self.price_cache_duration):
            return self.price_cache[cache_key]['data']

        try:
            if not self.finnhub_key:
                return self._get_default_price_data()

            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
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
                    self.price_cache[cache_key] = {'data': price_data, 'timestamp': current_time}
                    return price_data
        except Exception:
            pass

        return self._get_default_price_data()

    def _get_default_price_data(self):
        return {
            'current_price': 0, 'change': 0, 'change_percent': 0,
            'high': 0, 'low': 0, 'open': 0, 'prev_close': 0
        }

    def get_enhanced_analysis(self, ticker: str, sentiment_data: Dict) -> Dict:
        """Get enhanced analysis"""
        cache_key = f"enhanced_{ticker}_{int(time.time() / 900)}"
        if cache_key in self.advanced_cache:
            return self.advanced_cache[cache_key]

        result = {}

        if self.ai_analyst:
            try:
                ai_analysis = self.ai_analyst.analyze_trading_opportunity(ticker=ticker, sentiment_data=sentiment_data)
                result['ai_analysis'] = ai_analysis
            except Exception as e:
                result['ai_analysis'] = {'error': str(e)}

        if self.ml_forecaster:
            try:
                historical_data = self.ml_forecaster.get_historical_data(ticker, days=100)
                if not historical_data.empty:
                    ml_forecast = self.ml_forecaster.predict_price_movement(
                        ticker=ticker, current_data=historical_data, sentiment_data=[sentiment_data]
                    )
                    result['ml_forecast'] = ml_forecast
                else:
                    result['ml_forecast'] = {'error': 'No historical data'}
            except Exception as e:
                result['ml_forecast'] = {'error': str(e)}

        result['timestamp'] = datetime.now().isoformat()
        self.advanced_cache[cache_key] = result
        return result

    def get_trading_opportunities(self, max_tickers: int = 10):
        """Get trading opportunities with clean loading"""
        try:
            fm = self.firebase_manager

            # Simple, clean data loading
            recent_posts = fm.get_recent_posts(limit=100, hours=24, use_cache=True)
            trending_24h = fm.get_trending_tickers(hours=24, min_mentions=2, use_cache=True)
            trending_1h = fm.get_trending_tickers(hours=1, min_mentions=1, use_cache=True)
            sentiment_overview = fm.get_sentiment_overview(hours=24, use_cache=True)

            # Track reads
            for _ in range(4):  # 4 main queries
                self.read_counter.increment_read(is_cache_hit=False)

            opportunities = []
            top_sentiment_items = sentiment_overview[:max_tickers] if sentiment_overview else []

            for sentiment_item in top_sentiment_items:
                ticker = sentiment_item.get('ticker')
                if ticker:
                    price_data = self.get_stock_price_data(ticker)
                    trending_info = next((t for t in trending_24h if t.get('ticker') == ticker), {})
                    recent_trending = next((t for t in trending_1h if t.get('ticker') == ticker), {})

                    opportunity_score = self.calculate_opportunity_score(
                        sentiment_item, price_data, trending_info, recent_trending
                    )

                    opportunity = {
                        'ticker': ticker,
                        'sentiment': sentiment_item.get('sentiment', 'neutral'),
                        'confidence': float(sentiment_item.get('confidence', 0)),
                        'numerical_score': float(sentiment_item.get('numerical_score', 0)),
                        'mention_count_24h': int(trending_info.get('mention_count', 0)),
                        'mention_count_1h': int(recent_trending.get('mention_count', 0)) if recent_trending else 0,
                        'current_price': float(price_data['current_price']),
                        'change_percent': float(price_data['change_percent']),
                        'opportunity_score': opportunity_score,
                        'price_data': price_data,
                    }
                    opportunities.append(opportunity)

            opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)

            return {
                'opportunities': opportunities,
                'total_tickers': len(opportunities),
                'hot_stocks': [op for op in opportunities if op['opportunity_score'] > 50],
                'bullish_plays': [op for op in opportunities if op['sentiment'] == 'bullish' and op['confidence'] > 0.5],
                'bearish_plays': [op for op in opportunities if op['sentiment'] == 'bearish' and op['confidence'] > 0.5],
                'momentum_plays': [op for op in opportunities if op['mention_count_1h'] > 0 or abs(op['change_percent']) > 1],
                'read_status': self.read_counter.get_status(),
                'max_tickers_used': len(top_sentiment_items)
            }

        except Exception as e:
            return {
                'opportunities': [], 'total_tickers': 0, 'hot_stocks': [], 'bullish_plays': [],
                'bearish_plays': [], 'momentum_plays': [], 'error': str(e),
                'read_status': self.read_counter.get_status()
            }

    def calculate_opportunity_score(self, sentiment_data, price_data, trending_data, recent_trending):
        """Calculate opportunity score"""
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

        except (ValueError, TypeError):
            pass

        return min(score, 100)

    def render_clean_opportunity_card(self, stock: Dict, show_enhanced: bool = False):
        """Render clean, focused opportunity card with FIXED HTML rendering"""

        # Determine overall sentiment from AI + ML + Reddit
        ai_rating = stock.get('enhanced_analysis', {}).get('ai_analysis', {}).get('overall_rating', 'HOLD')
        ml_direction = stock.get('enhanced_analysis', {}).get('ml_forecast', {}).get('direction', stock['sentiment'])
        reddit_sentiment = stock['sentiment']

        # Smart recommendation logic - prioritize AI > ML > Reddit
        if ai_rating in ['STRONG_BUY', 'BUY']:
            final_recommendation = 'CALLS'
            final_sentiment = 'bullish'
            card_color = '#1B4332'  # Dark green
            border_color = '#2D6A4F'
            accent_color = '#52B788'
        elif ai_rating in ['STRONG_SELL', 'SELL']:
            final_recommendation = 'PUTS'
            final_sentiment = 'bearish'
            card_color = '#4D1F1F'  # Dark red
            border_color = '#8B3A3A'
            accent_color = '#E74C3C'
        elif ml_direction == 'up':
            final_recommendation = 'CALLS'
            final_sentiment = 'bullish'
            card_color = '#1B4332'
            border_color = '#2D6A4F'
            accent_color = '#52B788'
        elif ml_direction == 'down':
            final_recommendation = 'PUTS'
            final_sentiment = 'bearish'
            card_color = '#4D1F1F'
            border_color = '#8B3A3A'
            accent_color = '#E74C3C'
        elif reddit_sentiment == 'bullish':
            final_recommendation = 'CALLS'
            final_sentiment = 'bullish'
            card_color = '#1B4332'
            border_color = '#2D6A4F'
            accent_color = '#52B788'
        elif reddit_sentiment == 'bearish':
            final_recommendation = 'PUTS'
            final_sentiment = 'bearish'
            card_color = '#4D1F1F'
            border_color = '#8B3A3A'
            accent_color = '#E74C3C'
        else:
            final_recommendation = 'HOLD'
            final_sentiment = 'neutral'
            card_color = '#2C2C2C'
            border_color = '#666666'
            accent_color = '#999999'

        # Price styling
        price_change = stock['change_percent']
        if price_change > 0:
            price_color = "#00FF88"
            price_arrow = "↗"
        elif price_change < 0:
            price_color = "#FF4444"
            price_arrow = "↘"
        else:
            price_color = "#FFFFFF"
            price_arrow = "→"

        # Emoji mapping
        sentiment_emoji = {'bullish': '🚀', 'bearish': '📉', 'neutral': '😐'}[final_sentiment]

        # Confidence calculation (average of available confidences)
        confidences = []
        if stock.get('confidence'):
            confidences.append(stock['confidence'])
        if show_enhanced:
            ai_analysis = stock.get('enhanced_analysis', {}).get('ai_analysis', {})
            ml_forecast = stock.get('enhanced_analysis', {}).get('ml_forecast', {})
            if ai_analysis.get('confidence_score'):
                confidences.append(ai_analysis['confidence_score'])
            if ml_forecast.get('confidence'):
                confidences.append(ml_forecast['confidence'])

        avg_confidence = sum(confidences) / len(confidences) if confidences else stock.get('confidence', 0)

        price_display = f"${stock['current_price']:.2f}" if stock['current_price'] > 0 else "Price N/A"

        # FIXED: Use Streamlit columns for better layout instead of complex HTML

        # Card container with simple styling
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {card_color}, {card_color}DD); border: 2px solid {border_color}; border-radius: 12px; padding: 16px; margin: 8px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        """, unsafe_allow_html=True)

        # Header row using Streamlit columns
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
            <h3 style="margin: 0; color: {accent_color}; font-size: 1.4em;">
                {sentiment_emoji} {stock['ticker']}
            </h3>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="text-align: right;">
                <div style="font-size: 0.8em; color: #888;">Score: {stock['opportunity_score']:.0f}/100</div>
                <div style="font-size: 1.1em; color: {price_color}; font-weight: bold;">
                    {price_display} {price_arrow} {price_change:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Stats row using Streamlit columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div style="font-size: 0.9em;">
                <span style="color: #AAA;">Sentiment:</span> 
                <span style="color: {accent_color};">{reddit_sentiment.title()}</span> 
                <span style="color: #888;">({avg_confidence:.2f})</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            mentions_text = f"{stock['mention_count_24h']} (24h)"
            if stock['mention_count_1h'] > 0:
                mentions_text += f" | {stock['mention_count_1h']} (1h)"

            st.markdown(f"""
            <div style="font-size: 0.9em; text-align: right;">
                <span style="color: #AAA;">Mentions:</span> 
                <span style="color: white;">{mentions_text}</span>
            </div>
            """, unsafe_allow_html=True)

        # Recommendation row
        recommendation_text = (
            'Strong bullish signals aligned' if final_sentiment == 'bullish'
            else 'Strong bearish signals aligned' if final_sentiment == 'bearish'
            else 'Mixed signals - consider waiting'
        )

        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; border-left: 4px solid {accent_color}; text-align: center; font-weight: bold; color: {accent_color}; margin-top: 12px;">
            🎯 {final_recommendation}: {recommendation_text}
        </div>
        """, unsafe_allow_html=True)

        # Close card container
        st.markdown("</div>", unsafe_allow_html=True)

        # Enhanced analysis section (if requested) - COMPREHENSIVE VERSION
        if show_enhanced:
            enhanced_analysis = stock.get('enhanced_analysis', {})
            if enhanced_analysis and 'error' not in enhanced_analysis:

                # AI Analysis
                ai_analysis = enhanced_analysis.get('ai_analysis', {})
                ml_forecast = enhanced_analysis.get('ml_forecast', {})

                if (ai_analysis and 'error' not in ai_analysis) or (ml_forecast and 'error' not in ml_forecast):

                    # AI + ML header
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #1a1a3a, #2d2d4d); border: 1px solid #4488FF; border-radius: 8px; padding: 12px; margin: 8px 0;">
                    <div style="color: #4488FF; font-weight: bold; margin-bottom: 8px; font-size: 0.9em;">🤖 AI + ML Analysis</div>
                    """, unsafe_allow_html=True)

                    # Create metrics using Streamlit columns
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if ai_analysis and 'error' not in ai_analysis:
                            ai_rating = ai_analysis.get('overall_rating', 'HOLD')
                            ai_confidence = ai_analysis.get('confidence_score', 0)
                            rating_color = "#00FF88" if ai_rating in ['STRONG_BUY',
                                                                      'BUY'] else "#FF4444" if ai_rating in [
                                'STRONG_SELL', 'SELL'] else "#888888"

                            st.markdown(f"""
                            <div style="text-align: center;">
                                <div style="font-size: 0.7em; color: #888;">AI Rating</div>
                                <div style="font-size: 1.0em; color: {rating_color}; font-weight: bold;">{ai_rating}</div>
                                <div style="font-size: 0.7em; color: #AAA;">{ai_confidence:.0%} conf</div>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        if ml_forecast and 'error' not in ml_forecast:
                            ml_direction = ml_forecast.get('direction', 'neutral')
                            ml_confidence = ml_forecast.get('confidence', 0)
                            ml_change = ml_forecast.get('price_change_pct', 0)
                            ml_color = "#00FF88" if ml_direction == 'up' else "#FF4444" if ml_direction == 'down' else "#888888"
                            ml_arrow = "↗" if ml_direction == 'up' else "↘" if ml_direction == 'down' else "→"

                            st.markdown(f"""
                            <div style="text-align: center;">
                                <div style="font-size: 0.7em; color: #888;">ML Forecast</div>
                                <div style="font-size: 1.0em; color: {ml_color}; font-weight: bold;">{ml_change:+.1f}% {ml_arrow}</div>
                                <div style="font-size: 0.7em; color: #AAA;">{ml_confidence:.0%} conf</div>
                            </div>
                            """, unsafe_allow_html=True)

                    with col3:
                        if ai_analysis and 'error' not in ai_analysis:
                            options_strategy = ai_analysis.get('options_strategy', {})
                            recommended_play = options_strategy.get('recommended_play', 'hold')
                            expiration = options_strategy.get('expiration', 'weekly')
                            play_color = "#00FF88" if recommended_play == 'calls' else "#FF4444" if recommended_play == 'puts' else "#888888"

                            st.markdown(f"""
                            <div style="text-align: center;">
                                <div style="font-size: 0.7em; color: #888;">Options Play</div>
                                <div style="font-size: 1.0em; color: {play_color}; font-weight: bold;">{recommended_play.upper()}</div>
                                <div style="font-size: 0.7em; color: #AAA;">{expiration}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Entry/Exit Price Recommendations
                    if ai_analysis and 'error' not in ai_analysis:
                        target_price = ai_analysis.get('target_price', 0)
                        stop_loss = ai_analysis.get('stop_loss', 0)
                        current_price = stock['current_price']

                        if target_price > 0 and stop_loss > 0:
                            upside = ((target_price - current_price) / current_price * 100) if current_price > 0 else 0
                            downside = ((current_price - stop_loss) / current_price * 100) if current_price > 0 else 0

                            st.markdown(f"""
                            <div style="margin-top: 12px; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                                <div style="font-size: 0.8em; color: #4488FF; font-weight: bold; margin-bottom: 4px;">💰 Entry/Exit Targets</div>
                                <div style="display: flex; justify-content: space-between; font-size: 0.8em;">
                                    <span style="color: #00FF88;">🎯 Target: ${target_price:.2f} (+{upside:.1f}%)</span>
                                    <span style="color: #FF4444;">🛑 Stop: ${stop_loss:.2f} (-{downside:.1f}%)</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Key Catalysts and Risk Factors
                    if ai_analysis and 'error' not in ai_analysis:
                        catalysts = ai_analysis.get('key_catalysts', [])
                        risks = ai_analysis.get('risk_factors', [])

                        if catalysts or risks:
                            cat_col, risk_col = st.columns(2)

                            with cat_col:
                                if catalysts:
                                    catalysts_text = " • ".join(catalysts[:2])  # Show first 2
                                    st.markdown(f"""
                                    <div style="margin-top: 8px; font-size: 0.7em;">
                                        <div style="color: #00FF88; font-weight: bold;">📈 Catalysts:</div>
                                        <div style="color: #AAA;">{catalysts_text}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                            with risk_col:
                                if risks:
                                    risks_text = " • ".join(risks[:2])  # Show first 2
                                    st.markdown(f"""
                                    <div style="margin-top: 8px; font-size: 0.7em;">
                                        <div style="color: #FF4444; font-weight: bold;">⚠️ Risks:</div>
                                        <div style="color: #AAA;">{risks_text}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                    # Executive Summary
                    if ai_analysis and 'error' not in ai_analysis:
                        summary = ai_analysis.get('executive_summary', '')
                        if summary:
                            st.markdown(f"""
                            <div style="margin-top: 8px; padding: 6px; background: rgba(68,136,255,0.1); border-radius: 4px; font-size: 0.75em; color: #CCCCCC; font-style: italic;">
                                💡 {summary}
                            </div>
                            """, unsafe_allow_html=True)

                    # Close AI + ML section
                    st.markdown("</div>", unsafe_allow_html=True)

        # Sentiment Recap Section (always show if sentiment data exists)
        sentiment_distribution = stock.get('sentiment_distribution', {})
        if sentiment_distribution:
            st.markdown(f"""
            <div style="background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); border-radius: 6px; padding: 8px; margin: 8px 0;">
                <div style="color: #00FF88; font-weight: bold; font-size: 0.8em; margin-bottom: 4px;">📊 Sentiment Breakdown</div>
                <div style="font-size: 0.7em; display: flex; justify-content: space-between;">
                    <span style="color: #00FF88;">🐂 Bullish: {sentiment_distribution.get('bullish', 0)}</span>
                    <span style="color: #888888;">😐 Neutral: {sentiment_distribution.get('neutral', 0)}</span>
                    <span style="color: #FF4444;">🐻 Bearish: {sentiment_distribution.get('bearish', 0)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Reddit Posts Links Section - REAL LINKS
        reddit_posts = stock.get('reddit_posts', [])
        if reddit_posts:
            st.markdown(f"""
            <div style="background: rgba(255,69,0,0.1); border: 1px solid rgba(255,69,0,0.3); border-radius: 6px; padding: 8px; margin: 8px 0;">
                <div style="color: #FF4500; font-weight: bold; font-size: 0.8em; margin-bottom: 6px;">🔗 Recent Reddit Posts</div>
            """, unsafe_allow_html=True)

            for post in reddit_posts[:2]:  # Show top 2 posts
                if post.get('permalink'):
                    st.markdown(f"""
                    <div style="margin: 4px 0;">
                        <a href="{post['permalink']}" target="_blank" style="color: #FF6B35; font-size: 0.7em; text-decoration: none;">
                            📝 {post['title']} (⬆️ {post['score']})
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; margin-top: 8px;">
                <span style="color: #888888; font-size: 0.7em;">
                    🔍 No recent Reddit posts found for {stock['ticker']}
                </span>
            </div>
            """, unsafe_allow_html=True)

    def render_main_interface(self):
        """Render the main trading interface - UPDATED"""

        # COMPACT header
        st.markdown('<h1 style="text-align: center; color: #00FF88; margin-bottom: 0.5rem;">💰 WSB Options Trader</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #888; margin-bottom: 1rem;">Real-time Reddit sentiment • AI + ML trading intelligence</p>', unsafe_allow_html=True)

        # COMPACT action controls
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

        with col1:
            max_tickers = st.selectbox("📊 Analyze", [5, 10, 15, 20], index=1)

        with col2:
            analysis_mode = st.selectbox("🎯 Mode", ["Basic", "Advanced AI + ML"], index=0)

        with col3:
            if st.button("🔄", use_container_width=True, help="Refresh Data"):
                self.firebase_manager.clear_cache()
                st.cache_data.clear()
                st.rerun()

        with col4:
            # Compact quota indicator
            read_status = self.read_counter.get_status()
            quota_pct = (read_status['daily_reads'] / 35000) * 100
            if quota_pct > 80:
                st.error(f"⚠️{quota_pct:.0f}%")
            elif quota_pct > 50:
                st.warning(f"📊{quota_pct:.0f}%")
            else:
                st.success(f"✅{quota_pct:.0f}%")

        # Get data
        enable_advanced = (analysis_mode == "Advanced AI + ML")

        with st.spinner("🔍 Loading opportunities..."):
            data = self.get_trading_opportunities(max_tickers)

        if 'error' in data:
            st.error(f"❌ Error loading data: {data['error']}")
            return

        # COMPACT metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔥 Hot", len(data['hot_stocks']))
        with col2:
            st.metric("🚀 Bulls", len(data['bullish_plays']))
        with col3:
            st.metric("📉 Bears", len(data['bearish_plays']))
        with col4:
            st.metric("⚡ Momentum", len(data['momentum_plays']))

        # Enhanced analysis for advanced mode
        opportunities = data['opportunities']
        if enable_advanced and ADVANCED_ANALYTICS_AVAILABLE:
            with st.spinner(f"🤖 Running AI + ML analysis on top {min(5, len(opportunities))} opportunities..."):
                for i, opp in enumerate(opportunities[:5]):
                    enhanced_analysis = self.get_enhanced_analysis(opp['ticker'], {
                        'sentiment': opp['sentiment'],
                        'confidence': opp['confidence'],
                        'numerical_score': opp['numerical_score']
                    })
                    opportunities[i]['enhanced_analysis'] = enhanced_analysis

        # Display opportunities in a more compact grid
        st.markdown("## 🎯 Top Trading Opportunities")

        # Show opportunities in 2 columns for better space usage
        display_count = min(6, len(opportunities))  # Show fewer for better display
        for i in range(0, display_count, 2):
            col1, col2 = st.columns(2)

            with col1:
                if i < len(opportunities):
                    self.render_clean_opportunity_card(opportunities[i], enable_advanced)

            with col2:
                if i + 1 < len(opportunities):
                    self.render_clean_opportunity_card(opportunities[i + 1], enable_advanced)

        # COMPACT technical details in collapsible section
        with st.expander("🔧 Technical Details", expanded=False):
            read_status = data.get('read_status', {})
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("FB Reads", f"{read_status.get('daily_reads', 0):,}", help="Firebase reads today")
            with col2:
                st.metric("Analyzed", data['max_tickers_used'], help="Tickers analyzed")
            with col3:
                health = "🟢 Live" if read_status.get('quota_healthy', True) else "🟡 Cached"
                st.metric("Data", health, help="Data freshness status")

    def run(self):
        """Run the clean dashboard"""
        self.render_main_interface()


def main():
    """Main function"""
    dashboard = CleanTradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()