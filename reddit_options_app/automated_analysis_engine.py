"""
Automated Analysis Engine
Runs advanced AI + ML analysis on top trending tickers after each scraping cycle
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.firebase_manager import FirebaseManager

# Import ML components with proper availability checking
try:
    from ml.enhanced_trading_analyst import EnhancedTradingAnalyst
    from ml.ml_price_forecaster import MLPriceForecaster
    from ml import ML_COMPONENTS_AVAILABLE, ML_DATABASE_AVAILABLE

    if ML_COMPONENTS_AVAILABLE and ML_DATABASE_AVAILABLE:
        AUTOMATED_ANALYSIS_AVAILABLE = True
        print("✅ All ML components available for automated analysis")
    else:
        AUTOMATED_ANALYSIS_AVAILABLE = True  # Can still work with limited functionality
        print("⚠️ ML database not fully available - automated analysis will work with limited functionality")

except ImportError as e:
    AUTOMATED_ANALYSIS_AVAILABLE = False
    EnhancedTradingAnalyst = None
    MLPriceForecaster = None
    ML_COMPONENTS_AVAILABLE = False
    ML_DATABASE_AVAILABLE = False
    logging.warning(f"Advanced ML components not available: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomatedAnalysisEngine:
    """Automated analysis engine that runs after each scraping cycle"""

    def __init__(self):
        """Initialize the automated analysis engine"""

        self.firebase_manager = FirebaseManager()

        # Initialize advanced analytics
        try:
            self.ai_analyst = EnhancedTradingAnalyst()
            self.ml_forecaster = MLPriceForecaster()
            self.analytics_available = True
            logger.info("✅ Advanced analytics initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize analytics: {e}")
            self.analytics_available = False
            self.ai_analyst = None
            self.ml_forecaster = None

        # Configuration
        self.top_tickers_count = 5
        self.min_mentions_threshold = 2
        self.max_analysis_time = 600  # 10 minutes max
        self.parallel_analysis = True  # Run analyses in parallel

        # Results storage
        self.last_analysis_time = None
        self.analysis_results = {}

        logger.info(f"🤖 Automated Analysis Engine initialized - analyzing top {self.top_tickers_count} tickers")

    def get_top_trending_tickers(self, hours: int = 1) -> List[Dict]:
        """Get the top trending tickers from recent data"""

        try:
            # Get trending tickers from the last hour and 24 hours
            trending_1h = self.firebase_manager.get_trending_tickers(
                hours=1,
                min_mentions=1,
                use_cache=False  # Fresh data for analysis
            )

            trending_24h = self.firebase_manager.get_trending_tickers(
                hours=24,
                min_mentions=self.min_mentions_threshold,
                use_cache=False
            )

            # Combine and score tickers
            ticker_scores = {}

            # Score based on recent momentum (1h) and overall popularity (24h)
            for ticker_data in trending_1h:
                ticker = ticker_data['ticker']
                mentions_1h = ticker_data['mention_count']
                avg_score_1h = ticker_data.get('avg_score', 0)

                # Recent momentum score (higher weight for recency)
                momentum_score = mentions_1h * 2.0 + (avg_score_1h / 100)
                ticker_scores[ticker] = {
                    'ticker': ticker,
                    'momentum_score': momentum_score,
                    'mentions_1h': mentions_1h,
                    'mentions_24h': 0,
                    'avg_score_1h': avg_score_1h,
                    'avg_score_24h': 0,
                    'combined_score': momentum_score
                }

            # Add 24h data for context
            for ticker_data in trending_24h:
                ticker = ticker_data['ticker']
                mentions_24h = ticker_data['mention_count']
                avg_score_24h = ticker_data.get('avg_score', 0)

                if ticker in ticker_scores:
                    ticker_scores[ticker]['mentions_24h'] = mentions_24h
                    ticker_scores[ticker]['avg_score_24h'] = avg_score_24h
                    # Boost score if also trending over 24h
                    ticker_scores[ticker]['combined_score'] += mentions_24h * 0.5
                else:
                    # Add tickers that are only trending over 24h (with lower priority)
                    ticker_scores[ticker] = {
                        'ticker': ticker,
                        'momentum_score': 0,
                        'mentions_1h': 0,
                        'mentions_24h': mentions_24h,
                        'avg_score_1h': 0,
                        'avg_score_24h': avg_score_24h,
                        'combined_score': mentions_24h * 0.3  # Lower score for 24h only
                    }

            # Sort by combined score and return top tickers
            sorted_tickers = sorted(
                ticker_scores.values(),
                key=lambda x: x['combined_score'],
                reverse=True
            )

            top_tickers = sorted_tickers[:self.top_tickers_count]

            logger.info(f"📊 Selected top {len(top_tickers)} trending tickers for analysis:")
            for i, ticker_data in enumerate(top_tickers, 1):
                logger.info(f"  {i}. {ticker_data['ticker']} - Score: {ticker_data['combined_score']:.1f} "
                            f"(1h: {ticker_data['mentions_1h']}, 24h: {ticker_data['mentions_24h']})")

            return top_tickers

        except Exception as e:
            logger.error(f"❌ Error getting top trending tickers: {e}")
            return []

    def get_sentiment_data_for_ticker(self, ticker: str) -> Dict:
        """Get comprehensive sentiment data for a specific ticker"""

        try:
            # Get recent sentiment analysis for this ticker
            sentiment_overview = self.firebase_manager.get_sentiment_overview(hours=24, use_cache=False)

            # Find sentiment data for this ticker
            ticker_sentiment = None
            for sentiment_item in sentiment_overview:
                if sentiment_item.get('ticker', '').upper() == ticker.upper():
                    ticker_sentiment = sentiment_item
                    break

            if not ticker_sentiment:
                # Create default sentiment data if not found
                ticker_sentiment = {
                    'ticker': ticker,
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'numerical_score': 0.0,
                    'mention_count': 1
                }

            return ticker_sentiment

        except Exception as e:
            logger.error(f"❌ Error getting sentiment data for {ticker}: {e}")
            return {
                'ticker': ticker,
                'sentiment': 'neutral',
                'confidence': 0.5,
                'numerical_score': 0.0,
                'mention_count': 1
            }

    def get_reddit_mentions_for_ticker(self, ticker: str) -> List[Dict]:
        """Get recent Reddit mentions for a ticker"""

        try:
            # Get recent posts mentioning this ticker
            recent_posts = self.firebase_manager.get_posts_by_ticker(
                ticker=ticker,
                limit=10,
                use_cache=False
            )

            reddit_mentions = []
            for post in recent_posts:
                reddit_mentions.append({
                    'text': f"{post.get('title', '')} {post.get('selftext', '')}".strip(),
                    'score': post.get('score', 0),
                    'created_utc': post.get('created_utc', 0),
                    'permalink': post.get('permalink', '')
                })

            return reddit_mentions

        except Exception as e:
            logger.error(f"❌ Error getting Reddit mentions for {ticker}: {e}")
            return []

    def analyze_single_ticker(self, ticker_data: Dict) -> Dict:
        """Run complete analysis on a single ticker"""

        ticker = ticker_data['ticker']
        start_time = time.time()

        logger.info(f"🔍 Starting analysis for {ticker}...")

        try:
            # Get sentiment data
            sentiment_data = self.get_sentiment_data_for_ticker(ticker)

            # Get Reddit mentions
            reddit_mentions = self.get_reddit_mentions_for_ticker(ticker)

            # Initialize results
            analysis_result = {
                'ticker': ticker,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'trending_data': ticker_data,
                'sentiment_data': sentiment_data,
                'reddit_mentions_count': len(reddit_mentions),
                'ai_analysis': {},
                'ml_forecast': {},
                'analysis_duration_seconds': 0,
                'status': 'pending'
            }

            # Run AI Analysis
            if self.ai_analyst:
                try:
                    logger.info(f"🤖 Running AI analysis for {ticker}...")
                    ai_analysis = self.ai_analyst.analyze_trading_opportunity(
                        ticker=ticker,
                        sentiment_data=sentiment_data,
                        reddit_mentions=reddit_mentions
                    )
                    analysis_result['ai_analysis'] = ai_analysis
                    logger.info(f"✅ AI analysis complete for {ticker}: {ai_analysis.get('overall_rating', 'UNKNOWN')}")
                except Exception as e:
                    logger.error(f"❌ AI analysis failed for {ticker}: {e}")
                    analysis_result['ai_analysis'] = {'error': str(e)}

            # Run ML Forecast
            if self.ml_forecaster:
                try:
                    logger.info(f"📊 Running ML forecast for {ticker}...")

                    # Get historical data
                    historical_data = self.ml_forecaster.get_historical_data(ticker, days=100)

                    if not historical_data.empty:
                        # Train models if needed
                        if ticker not in self.ml_forecaster.models:
                            train_result = self.ml_forecaster.train_models(ticker)
                            if train_result.get('success'):
                                logger.info(f"📈 Trained {train_result.get('models_trained', 0)} models for {ticker}")

                        # Make prediction
                        if ticker in self.ml_forecaster.models:
                            ml_forecast = self.ml_forecaster.predict_price_movement(
                                ticker=ticker,
                                current_data=historical_data,
                                sentiment_data=[sentiment_data]
                            )
                            analysis_result['ml_forecast'] = ml_forecast

                            predicted_change = ml_forecast.get('price_change_pct', 0)
                            confidence = ml_forecast.get('confidence', 0)
                            logger.info(
                                f"✅ ML forecast complete for {ticker}: {predicted_change:+.2f}% (confidence: {confidence:.2f})")
                        else:
                            analysis_result['ml_forecast'] = {'error': 'Model training failed'}
                    else:
                        analysis_result['ml_forecast'] = {'error': 'No historical data available'}

                except Exception as e:
                    logger.error(f"❌ ML forecast failed for {ticker}: {e}")
                    analysis_result['ml_forecast'] = {'error': str(e)}

            # Calculate analysis duration
            analysis_duration = time.time() - start_time
            analysis_result['analysis_duration_seconds'] = round(analysis_duration, 2)
            analysis_result['status'] = 'completed'

            logger.info(f"🎯 Analysis complete for {ticker} in {analysis_duration:.1f}s")

            return analysis_result

        except Exception as e:
            analysis_duration = time.time() - start_time
            logger.error(f"❌ Complete analysis failed for {ticker}: {e}")

            return {
                'ticker': ticker,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'trending_data': ticker_data,
                'analysis_duration_seconds': round(analysis_duration, 2),
                'status': 'failed',
                'error': str(e)
            }

    def run_parallel_analysis(self, top_tickers: List[Dict]) -> List[Dict]:
        """Run analysis on multiple tickers in parallel"""

        if not self.parallel_analysis:
            # Sequential analysis
            results = []
            for ticker_data in top_tickers:
                result = self.analyze_single_ticker(ticker_data)
                results.append(result)
            return results

        # Parallel analysis
        logger.info(f"🚀 Starting parallel analysis of {len(top_tickers)} tickers...")

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:  # Limit concurrent analyses
            # Submit all analysis tasks
            future_to_ticker = {
                executor.submit(self.analyze_single_ticker, ticker_data): ticker_data['ticker']
                for ticker_data in top_tickers
            }

            # Collect results as they complete
            for future in as_completed(future_to_ticker, timeout=self.max_analysis_time):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=120)  # 2 minutes per ticker max
                    results.append(result)
                    logger.info(f"✅ Completed analysis for {ticker}")
                except Exception as e:
                    logger.error(f"❌ Analysis failed for {ticker}: {e}")
                    results.append({
                        'ticker': ticker,
                        'status': 'failed',
                        'error': str(e),
                        'analysis_timestamp': datetime.now(timezone.utc).isoformat()
                    })

        return results

    def save_analysis_results(self, analysis_results: List[Dict]):
        """Save analysis results to Firebase"""

        try:
            # Prepare data for storage
            storage_data = []

            for result in analysis_results:
                storage_record = {
                    'ticker': result['ticker'],
                    'analysis_timestamp': result['analysis_timestamp'],
                    'analysis_type': 'automated_trending',
                    'trending_data': result.get('trending_data', {}),
                    'sentiment_data': result.get('sentiment_data', {}),
                    'ai_analysis': result.get('ai_analysis', {}),
                    'ml_forecast': result.get('ml_forecast', {}),
                    'reddit_mentions_count': result.get('reddit_mentions_count', 0),
                    'analysis_duration_seconds': result.get('analysis_duration_seconds', 0),
                    'status': result.get('status', 'unknown'),
                    'automation_version': '1.0'
                }

                if 'error' in result:
                    storage_record['error'] = result['error']

                storage_data.append(storage_record)

            # Save to Firebase with batch operation
            saved_count = self.firebase_manager.batch_save(
                'automated_analysis',
                storage_data,
                'ticker'  # Use ticker as ID field for deduplication
            )

            logger.info(f"💾 Saved {saved_count} automated analysis results to Firebase")

        except Exception as e:
            logger.error(f"❌ Failed to save analysis results: {e}")

    def run_automated_analysis(self) -> Dict:
        """Run the complete automated analysis workflow"""

        if not self.analytics_available:
            logger.warning("⚠️ Analytics not available, skipping automated analysis")
            return {'status': 'skipped', 'reason': 'Analytics not available'}

        start_time = time.time()
        self.last_analysis_time = datetime.now(timezone.utc)

        logger.info("🚀 Starting automated analysis of top trending tickers...")

        try:
            # Step 1: Get top trending tickers
            top_tickers = self.get_top_trending_tickers()

            if not top_tickers:
                logger.warning("⚠️ No trending tickers found for analysis")
                return {'status': 'completed', 'tickers_analyzed': 0, 'reason': 'No trending tickers'}

            # Step 2: Run analysis on all tickers
            analysis_results = self.run_parallel_analysis(top_tickers)

            # Step 3: Save results
            self.save_analysis_results(analysis_results)

            # Step 4: Update in-memory cache for dashboard
            self.analysis_results = {
                result['ticker']: result for result in analysis_results
            }

            # Calculate summary stats
            total_time = time.time() - start_time
            successful_analyses = len([r for r in analysis_results if r.get('status') == 'completed'])

            summary = {
                'status': 'completed',
                'analysis_timestamp': self.last_analysis_time.isoformat(),
                'tickers_analyzed': len(analysis_results),
                'successful_analyses': successful_analyses,
                'total_duration_seconds': round(total_time, 2),
                'average_time_per_ticker': round(total_time / len(analysis_results), 2) if analysis_results else 0,
                'tickers': [r['ticker'] for r in analysis_results],
                'results': analysis_results
            }

            logger.info(f"🎉 Automated analysis complete!")
            logger.info(f"   📊 Analyzed {len(analysis_results)} tickers in {total_time:.1f}s")
            logger.info(f"   ✅ {successful_analyses} successful analyses")
            logger.info(f"   📈 Average time per ticker: {total_time / len(analysis_results):.1f}s")

            return summary

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Automated analysis failed after {total_time:.1f}s: {e}")

            return {
                'status': 'failed',
                'error': str(e),
                'total_duration_seconds': round(total_time, 2),
                'analysis_timestamp': self.last_analysis_time.isoformat() if self.last_analysis_time else None
            }

    def get_latest_results(self) -> Dict:
        """Get the latest automated analysis results"""

        return {
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'results_count': len(self.analysis_results),
            'results': self.analysis_results,
            'analytics_available': self.analytics_available
        }

    def should_run_analysis(self, force: bool = False) -> bool:
        """Check if analysis should run based on timing"""

        if force:
            return True

        if not self.last_analysis_time:
            return True

        # Don't run more than once every 15 minutes
        time_since_last = datetime.now(timezone.utc) - self.last_analysis_time
        return time_since_last.total_seconds() > 900  # 15 minutes


def main():
    """Test the automated analysis engine"""

    print("🧪 Testing Automated Analysis Engine")
    print("=" * 50)

    engine = AutomatedAnalysisEngine()

    if not engine.analytics_available:
        print("❌ Analytics not available, exiting")
        return

    # Run analysis
    result = engine.run_automated_analysis()

    print(f"\n📊 Analysis Result:")
    print(f"Status: {result['status']}")
    print(f"Tickers analyzed: {result.get('tickers_analyzed', 0)}")
    print(f"Duration: {result.get('total_duration_seconds', 0):.1f}s")

    if result['status'] == 'completed':
        print(f"Successful analyses: {result.get('successful_analyses', 0)}")
        print(f"Tickers: {', '.join(result.get('tickers', []))}")


if __name__ == "__main__":
    main()