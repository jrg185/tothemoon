"""
Enhanced AI Trading Analyst - UPDATED with ML Integration
Provides deep analysis of trading opportunities using LLM intelligence + ML forecasting
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import openai
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
import requests
import time
import numpy as np

from config.settings import LLM_CONFIG, FINANCIAL_APIS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTradingAnalyst:
    """Advanced AI analyst for trading opportunities with ML integration"""

    def __init__(self):
        """Initialize the enhanced trading analyst"""
        # Initialize LLM
        self._initialize_llm()

        # Financial APIs
        self.finnhub_key = FINANCIAL_APIS.get('finnhub')
        self.alpha_vantage_key = FINANCIAL_APIS.get('alpha_vantage')

        # Analysis cache to avoid repeated API calls
        self.analysis_cache = {}
        self.cache_duration = 1800  # 30 minutes

        # Initialize ML forecaster if available
        self.ml_forecaster = None
        self.ml_available = False
        self._initialize_ml_forecaster()

        logger.info("Enhanced Trading Analyst initialized")

    def _initialize_llm(self):
        """Initialize LLM for enhanced analysis"""
        provider = LLM_CONFIG['default_provider']

        if provider == 'xai':
            api_key = LLM_CONFIG['xai']['api_key']
            self.llm_client = openai.OpenAI(
                api_key=api_key,
                base_url=LLM_CONFIG['xai']['base_url']
            )
            self.llm_model = LLM_CONFIG['xai']['model']
        else:
            api_key = LLM_CONFIG['openai']['api_key']
            self.llm_client = openai.OpenAI(api_key=api_key)
            self.llm_model = LLM_CONFIG['openai']['model']

    def _initialize_ml_forecaster(self):
        """Initialize ML forecaster if available"""
        try:
            from ml.ml_price_forecaster import MLPriceForecaster
            self.ml_forecaster = MLPriceForecaster()
            self.ml_available = True
            logger.info("✅ ML Price Forecaster integrated successfully")
        except Exception as e:
            logger.warning(f"⚠️ ML Price Forecaster not available: {e}")
            self.ml_forecaster = None
            self.ml_available = False

    def get_enhanced_market_data(self, ticker: str) -> Dict:
        """Get comprehensive market data for a ticker"""
        try:
            market_data = {}

            # Get current quote data
            if self.finnhub_key:
                quote_url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}"
                quote_response = requests.get(quote_url, timeout=10)
                if quote_response.status_code == 200:
                    quote_data = quote_response.json()
                    market_data.update({
                        'current_price': quote_data.get('c', 0),
                        'change': quote_data.get('d', 0),
                        'change_percent': quote_data.get('dp', 0),
                        'high': quote_data.get('h', 0),
                        'low': quote_data.get('l', 0),
                        'open': quote_data.get('o', 0),
                        'prev_close': quote_data.get('pc', 0)
                    })

                # Get company profile
                profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={self.finnhub_key}"
                profile_response = requests.get(profile_url, timeout=10)
                if profile_response.status_code == 200:
                    profile_data = profile_response.json()
                    market_data.update({
                        'company_name': profile_data.get('name', ticker),
                        'industry': profile_data.get('finnhubIndustry', 'Unknown'),
                        'market_cap': profile_data.get('marketCapitalization', 0),
                        'shares_outstanding': profile_data.get('shareOutstanding', 0)
                    })

                # Get basic financials
                metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={self.finnhub_key}"
                metrics_response = requests.get(metrics_url, timeout=10)
                if metrics_response.status_code == 200:
                    metrics_data = metrics_response.json()
                    if 'metric' in metrics_data:
                        metric = metrics_data['metric']
                        market_data.update({
                            'pe_ratio': metric.get('peBasicExclExtraTTM', 0),
                            'pb_ratio': metric.get('pbQuarterly', 0),
                            'roe': metric.get('roeRfy', 0),
                            'debt_to_equity': metric.get('totalDebt/totalEquityQuarterly', 0),
                            '52_week_high': metric.get('52WeekHigh', 0),
                            '52_week_low': metric.get('52WeekLow', 0)
                        })

                # Get recent news
                news_url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')}&to={datetime.now().strftime('%Y-%m-%d')}&token={self.finnhub_key}"
                news_response = requests.get(news_url, timeout=10)
                if news_response.status_code == 200:
                    news_data = news_response.json()
                    market_data['recent_news'] = [
                        {
                            'headline': article.get('headline', ''),
                            'summary': article.get('summary', ''),
                            'datetime': article.get('datetime', 0),
                            'source': article.get('source', '')
                        }
                        for article in news_data[:5]  # Top 5 recent news
                    ]

                time.sleep(0.2)  # Rate limiting

            return market_data

        except Exception as e:
            logger.error(f"Error getting market data for {ticker}: {e}")
            return {}

    def calculate_technical_indicators(self, ticker: str) -> Dict:
        """Calculate technical indicators for analysis"""
        try:
            if not self.alpha_vantage_key:
                return {}

            # Get daily price data
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={self.alpha_vantage_key}"
            response = requests.get(url, timeout=15)

            if response.status_code != 200:
                return {}

            data = response.json()

            if 'Time Series (Daily)' not in data:
                return {}

            time_series = data['Time Series (Daily)']
            dates = sorted(time_series.keys(), reverse=True)[:50]  # Last 50 days

            # Extract price data
            closes = [float(time_series[date]['4. close']) for date in dates]
            highs = [float(time_series[date]['2. high']) for date in dates]
            lows = [float(time_series[date]['3. low']) for date in dates]
            volumes = [float(time_series[date]['5. volume']) for date in dates]

            # Calculate technical indicators
            indicators = {}

            # Simple Moving Averages
            if len(closes) >= 20:
                indicators['sma_5'] = np.mean(closes[:5])
                indicators['sma_10'] = np.mean(closes[:10])
                indicators['sma_20'] = np.mean(closes[:20])

                # Price relative to SMAs
                current_price = closes[0]
                indicators['price_vs_sma20'] = ((current_price - indicators['sma_20']) / indicators['sma_20']) * 100

            # Volatility (20-day)
            if len(closes) >= 20:
                returns = [(closes[i] - closes[i + 1]) / closes[i + 1] for i in range(19)]
                indicators['volatility_20d'] = np.std(returns) * np.sqrt(252) * 100  # Annualized

            # Volume trend
            if len(volumes) >= 10:
                recent_volume = np.mean(volumes[:5])
                avg_volume = np.mean(volumes[5:])
                indicators['volume_trend'] = ((recent_volume - avg_volume) / avg_volume) * 100

            # Support/Resistance levels
            if len(highs) >= 20 and len(lows) >= 20:
                indicators['resistance_level'] = np.max(highs[:20])
                indicators['support_level'] = np.min(lows[:20])

            return indicators

        except Exception as e:
            logger.error(f"Error calculating technical indicators for {ticker}: {e}")
            return {}

    def get_ml_forecast(self, ticker: str, sentiment_data: Dict) -> Dict:
        """Get ML forecast for the ticker"""
        if not self.ml_available or not self.ml_forecaster:
            return {'error': 'ML forecaster not available'}

        try:
            logger.info(f"🤖 Getting ML forecast for {ticker}...")

            # Make prediction with sentiment integration
            prediction_result = self.ml_forecaster.predict_price_movement(
                ticker=ticker,
                current_data=None,  # Will fetch automatically
                sentiment_data=[sentiment_data]
            )

            if 'error' not in prediction_result:
                ml_forecast = {
                    'direction': prediction_result['direction'],
                    'confidence': prediction_result['confidence'],
                    'price_change_pct': prediction_result.get('price_change_pct', 0),
                    'predicted_price': prediction_result.get('predicted_price', 0),
                    'predicted_return': prediction_result.get('predicted_return', 0),
                    'signals': prediction_result.get('signals', []),
                    'prediction_id': prediction_result.get('prediction_id'),
                    'model_version': 'v2.0_improved',
                    'saved_to_database': prediction_result.get('saved_to_database', False)
                }
                logger.info(f"✅ ML forecast: {prediction_result['direction']} ({prediction_result['confidence']:.2%})")
                return ml_forecast
            else:
                logger.warning(f"⚠️ ML forecast failed: {prediction_result['error']}")
                return {'error': prediction_result['error']}

        except Exception as e:
            logger.warning(f"⚠️ ML forecast exception: {e}")
            return {'error': str(e)}

        def generate_enhanced_analysis(self,
                                       ticker: str,
                                       sentiment_data: Dict,
                                       market_data: Dict,
                                       technical_indicators: Dict,
                                       reddit_mentions: List[Dict],
                                       ml_forecast: Dict = None) -> Dict:
            """Generate comprehensive AI analysis of trading opportunity with ML integration"""

            cache_key = f"analysis_{ticker}_{int(time.time() / self.cache_duration)}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]

            try:
                # Prepare context for LLM analysis - FIXED: Handle None values properly
                current_price = market_data.get('current_price', 0) or 0
                change_percent = market_data.get('change_percent', 0) or 0
                sentiment = sentiment_data.get('sentiment', 'neutral') or 'neutral'
                confidence = sentiment_data.get('confidence', 0) or 0
                mention_count = sentiment_data.get('mention_count', 0) or 0

                # Recent news headlines
                news_context = ""
                if 'recent_news' in market_data:
                    headlines = [news['headline'] for news in market_data['recent_news'][:3] if news.get('headline')]
                    news_context = "Recent news: " + "; ".join(headlines) if headlines else "No recent news"

                # Technical context - FIXED: Handle None values
                tech_context = ""
                if technical_indicators:
                    tech_items = []
                    price_vs_sma20 = technical_indicators.get('price_vs_sma20')
                    if price_vs_sma20 is not None:
                        tech_items.append(f"Price vs 20-day SMA: {price_vs_sma20:.1f}%")

                    volatility_20d = technical_indicators.get('volatility_20d')
                    if volatility_20d is not None:
                        tech_items.append(f"20-day volatility: {volatility_20d:.1f}%")

                    volume_trend = technical_indicators.get('volume_trend')
                    if volume_trend is not None:
                        tech_items.append(f"Volume trend: {volume_trend:.1f}%")

                    tech_context = "; ".join(tech_items) if tech_items else "No technical data available"

                # Reddit sentiment context
                reddit_context = f"WSB mentions: {mention_count}, sentiment: {sentiment} (confidence: {confidence:.2f})"

                # ML forecast context - FIXED: Handle None values
                ml_context = ""
                if ml_forecast and 'error' not in ml_forecast:
                    ml_direction = ml_forecast.get('direction', 'unknown') or 'unknown'
                    ml_confidence = ml_forecast.get('confidence', 0) or 0
                    ml_change = ml_forecast.get('price_change_pct', 0) or 0
                    ml_context = f"ML Model Prediction: {ml_direction} ({ml_confidence:.1%} confidence, {ml_change:+.1f}% expected change)"
                elif ml_forecast:
                    ml_context = f"ML Model: {ml_forecast.get('error', 'unavailable')}"
                else:
                    ml_context = "ML Model: not available"

                # Sample of recent mentions for context
                mention_samples = []
                for mention in reddit_mentions[:3]:
                    text = mention.get('text', '')[:100] if mention.get('text') else ''
                    score = mention.get('score', 0) or 0
                    if text:  # Only add if we have text
                        mention_samples.append(f"'{text}' (score: {score})")

                # FIXED: Use safe formatting for company info
                company_name = market_data.get('company_name') or ticker
                industry = market_data.get('industry') or 'Unknown'
                market_cap = market_data.get('market_cap', 0) or 0
                pe_ratio = market_data.get('pe_ratio', 0) or 0

                prompt = f"""
    Analyze this trading opportunity for {ticker}:

    CURRENT MARKET DATA:
    - Price: ${current_price:.2f} ({change_percent:+.2f}%)
    - Company: {company_name}
    - Industry: {industry}
    - Market Cap: ${market_cap:.1f}M
    - P/E Ratio: {pe_ratio:.1f}

    TECHNICAL ANALYSIS:
    {tech_context}

    REDDIT SENTIMENT:
    {reddit_context}

    ML FORECAST:
    {ml_context}

    RECENT NEWS:
    {news_context}

    SAMPLE REDDIT MENTIONS:
    {'; '.join(mention_samples) if mention_samples else 'No mentions available'}

    Provide a comprehensive trading analysis in this EXACT JSON format:
    {{
        "overall_rating": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
        "confidence_score": 0.0-1.0,
        "target_price": number,
        "stop_loss": number,
        "time_horizon": "1-7 days|1-2 weeks|1 month|3+ months",
        "key_catalysts": ["catalyst1", "catalyst2", "catalyst3"],
        "risk_factors": ["risk1", "risk2", "risk3"],
        "options_strategy": {{
            "recommended_play": "calls|puts|straddle|iron_condor|cash",
            "strike_selection": "ATM|OTM|ITM|multiple_strikes",
            "expiration": "weekly|monthly|quarterly",
            "reasoning": "brief explanation"
        }},
        "technical_outlook": {{
            "trend": "bullish|bearish|sideways",
            "support_level": number,
            "resistance_level": number,
            "momentum": "strong|moderate|weak"
        }},
        "sentiment_analysis": {{
            "wsb_sentiment": "very_bullish|bullish|neutral|bearish|very_bearish",
            "sentiment_quality": "high|medium|low",
            "contrarian_signal": true/false
        }},
        "ml_integration": {{
            "ml_direction_agreement": "agrees|disagrees|neutral",
            "combined_confidence": 0.0-1.0,
            "ml_sentiment_alignment": "aligned|conflicted|neutral"
        }},
        "executive_summary": "2-3 sentence summary of the play including ML insights"
    }}

    IMPORTANT: Return ONLY valid JSON. No markdown, no explanations outside the JSON.
    """

                # Make LLM call with appropriate parameters for provider
                if LLM_CONFIG['default_provider'] == 'xai':
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system",
                             "content": "You are an expert quantitative trader and financial analyst with access to ML models. Provide detailed, actionable trading analysis that integrates ML predictions with fundamental and technical analysis. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=4000,
                        timeout=45
                    )
                else:
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system",
                             "content": "You are an expert quantitative trader and financial analyst with access to ML models. Provide detailed, actionable trading analysis that integrates ML predictions with fundamental and technical analysis. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=2000,
                        timeout=30
                    )

                # Parse response
                raw_content = response.choices[0].message.content.strip()

                # Clean JSON response
                if raw_content.startswith('```json'):
                    raw_content = raw_content[7:]
                elif raw_content.startswith('```'):
                    raw_content = raw_content[3:]
                if raw_content.endswith('```'):
                    raw_content = raw_content[:-3]

                raw_content = raw_content.strip()

                # Find JSON boundaries
                start_idx = raw_content.find('{')
                end_idx = raw_content.rfind('}')

                if start_idx != -1 and end_idx != -1:
                    json_content = raw_content[start_idx:end_idx + 1]
                    analysis = json.loads(json_content)

                    # Add metadata
                    analysis['ticker'] = ticker
                    analysis['analysis_timestamp'] = datetime.now(timezone.utc).isoformat()
                    analysis['data_sources'] = ['reddit_sentiment', 'market_data', 'technical_analysis', 'recent_news',
                                                'ml_forecast']
                    analysis['ml_forecast_available'] = ml_forecast is not None and 'error' not in ml_forecast

                    # Cache the result
                    self.analysis_cache[cache_key] = analysis

                    return analysis
                else:
                    raise ValueError("No valid JSON found in response")

            except Exception as e:
                logger.error(f"Error generating enhanced analysis for {ticker}: {e}")
                # Return basic analysis as fallback - FIXED: Handle None values safely
                current_price = market_data.get('current_price', 0) or 0
                return {
                    'ticker': ticker,
                    'overall_rating': 'HOLD',
                    'confidence_score': 0.5,
                    'target_price': current_price,
                    'stop_loss': current_price * 0.95,
                    'time_horizon': '1-2 weeks',
                    'key_catalysts': ['Reddit momentum'],
                    'risk_factors': ['Market volatility'],
                    'options_strategy': {
                        'recommended_play': 'calls' if sentiment == 'bullish' else 'puts',
                        'strike_selection': 'ATM',
                        'expiration': 'weekly',
                        'reasoning': 'Basic sentiment-based play'
                    },
                    'technical_outlook': {
                        'trend': sentiment,
                        'support_level': current_price * 0.95,
                        'resistance_level': current_price * 1.05,
                        'momentum': 'moderate'
                    },
                    'sentiment_analysis': {
                        'wsb_sentiment': sentiment,
                        'sentiment_quality': 'medium',
                        'contrarian_signal': False
                    },
                    'ml_integration': {
                        'ml_direction_agreement': 'neutral',
                        'combined_confidence': 0.5,
                        'ml_sentiment_alignment': 'neutral'
                    },
                    'executive_summary': f"Basic analysis for {ticker} based on {sentiment} sentiment",
                    'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                    'error': str(e)
                }

    def analyze_trading_opportunity(self,
                                    ticker: str,
                                    sentiment_data: Dict,
                                    reddit_mentions: List[Dict] = None) -> Dict:
        """Complete analysis of a trading opportunity with ML integration"""

        logger.info(f"Starting enhanced analysis for {ticker}")

        try:
            # Get comprehensive market data
            market_data = self.get_enhanced_market_data(ticker)

            # Calculate technical indicators
            technical_indicators = self.calculate_technical_indicators(ticker)

            # Get ML forecast
            ml_forecast = self.get_ml_forecast(ticker, sentiment_data)

            # Generate AI analysis with ML integration
            if reddit_mentions is None:
                reddit_mentions = []

            enhanced_analysis = self.generate_enhanced_analysis(
                ticker=ticker,
                sentiment_data=sentiment_data,
                market_data=market_data,
                technical_indicators=technical_indicators,
                reddit_mentions=reddit_mentions,
                ml_forecast=ml_forecast
            )

            # Add ML forecast to the analysis output
            enhanced_analysis['ml_forecast'] = ml_forecast

            # Add raw data for reference
            enhanced_analysis['raw_data'] = {
                'market_data': market_data,
                'technical_indicators': technical_indicators,
                'sentiment_data': sentiment_data
            }

            logger.info(
                f"Enhanced analysis complete for {ticker}: {enhanced_analysis.get('overall_rating', 'UNKNOWN')}")

            return enhanced_analysis

        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }

    def batch_analyze_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Analyze multiple trading opportunities with ML integration"""

        enhanced_opportunities = []

        for i, opp in enumerate(opportunities[:10]):  # Limit to top 10 to avoid API limits
            ticker = opp.get('ticker')
            if not ticker:
                continue

            logger.info(f"Analyzing {i + 1}/{min(len(opportunities), 10)}: {ticker}")

            try:
                # Convert opportunity data to sentiment format
                sentiment_data = {
                    'sentiment': opp.get('sentiment', 'neutral'),
                    'confidence': opp.get('confidence', 0),
                    'mention_count': opp.get('mention_count_24h', 0),
                    'numerical_score': opp.get('numerical_score', 0)
                }

                # Get enhanced analysis with ML
                analysis = self.analyze_trading_opportunity(
                    ticker=ticker,
                    sentiment_data=sentiment_data
                )

                # Merge with original opportunity data
                enhanced_opp = {**opp, 'enhanced_analysis': analysis}
                enhanced_opportunities.append(enhanced_opp)

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
                enhanced_opportunities.append({**opp, 'enhanced_analysis': {'error': str(e)}})

        return enhanced_opportunities


def main():
    """Test the enhanced trading analyst with ML integration"""
    analyst = EnhancedTradingAnalyst()

    # Test analysis
    test_sentiment = {
        'sentiment': 'bullish',
        'confidence': 0.75,
        'mention_count': 15,
        'numerical_score': 0.6
    }

    test_mentions = [
        {'text': 'TSLA calls looking good for earnings', 'score': 150},
        {'text': 'Tesla delivery numbers beat expectations', 'score': 89}
    ]

    print("🤖 Testing Enhanced Trading Analyst with ML Integration...")
    analysis = analyst.analyze_trading_opportunity('TSLA', test_sentiment, test_mentions)

    print(f"\n📊 Analysis Results:")
    print(f"Rating: {analysis.get('overall_rating', 'N/A')}")
    print(f"Confidence: {analysis.get('confidence_score', 0):.2f}")
    print(f"Target: ${analysis.get('target_price', 0):.2f}")
    print(f"Strategy: {analysis.get('options_strategy', {}).get('recommended_play', 'N/A')}")
    print(f"ML Available: {analysis.get('ml_forecast_available', False)}")

    if 'ml_forecast' in analysis and 'error' not in analysis['ml_forecast']:
        ml_forecast = analysis['ml_forecast']
        print(f"ML Direction: {ml_forecast.get('direction', 'N/A')}")
        print(f"ML Confidence: {ml_forecast.get('confidence', 0):.2%}")
        print(f"Prediction ID: {ml_forecast.get('prediction_id', 'N/A')}")

    print(f"Summary: {analysis.get('executive_summary', 'N/A')}")


if __name__ == "__main__":
    main()