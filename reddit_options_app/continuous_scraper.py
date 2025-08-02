"""
Enhanced Continuous Reddit Scraper with Automated Analysis
Runs advanced analysis on top trending tickers after each scraping cycle
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import schedule
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List
from data import RedditScraper
from data.firebase_manager import FirebaseManager
from processing.sentiment_analyzer import FinancialSentimentAnalyzer
from config.settings import APP_CONFIG

# Import automated analysis engine
try:
    from automated_analysis_engine import AutomatedAnalysisEngine
    AUTOMATED_ANALYSIS_AVAILABLE = True
except ImportError as e:
    AUTOMATED_ANALYSIS_AVAILABLE = False
    logging.warning(f"Automated analysis not available: {e}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedContinuousRedditScraper:
    """Enhanced continuous scraper with automated analysis"""

    def __init__(self):
        """Initialize enhanced continuous scraper"""

        # Core scraping components
        self.firebase_manager = FirebaseManager()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.scraper = RedditScraper()
        self.scraper.firebase_manager = self.firebase_manager

        # Automated analysis engine
        self.analysis_engine = None
        if AUTOMATED_ANALYSIS_AVAILABLE:
            try:
                self.analysis_engine = AutomatedAnalysisEngine()
                logger.info("✅ Automated analysis engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize analysis engine: {e}")
                self.analysis_engine = None
        else:
            logger.warning("⚠️ Automated analysis engine not available")

        # Statistics
        self.total_posts_scraped = 0
        self.total_comments_scraped = 0
        self.total_tickers_found = set()
        self.total_analyses_run = 0
        self.start_time = datetime.now(timezone.utc)

        # Deduplication cache
        self.processed_posts = set()
        self.processed_comments = set()

        # Cycle timing
        self.last_scrape_time = 0
        self.last_analysis_time = 0
        self.min_cycle_interval = 900  # 15 minutes between cycles

        logger.info("🚀 Enhanced Continuous Reddit Scraper initialized")

    def should_run_cycle(self) -> bool:
        """Check if enough time has passed since last cycle"""
        current_time = time.time()
        time_since_last = current_time - self.last_scrape_time

        if time_since_last < self.min_cycle_interval:
            remaining = self.min_cycle_interval - time_since_last
            logger.info(f"⏳ Next cycle in {remaining/60:.1f} minutes")
            return False

        # Check Firebase quota
        quota_status = self.firebase_manager.get_quota_status()
        reads_today = quota_status.get('reads_today', 0)
        daily_limit = quota_status.get('daily_limit', 35000)

        if reads_today >= daily_limit * 0.9:  # 90% of limit
            logger.warning(f"🚨 Firebase quota at 90% ({reads_today}/{daily_limit}) - skipping cycle")
            return False

        return True

    def scrape_cycle(self) -> Dict:
        """Execute one scraping cycle"""
        if not self.should_run_cycle():
            return {'skipped': 'Too soon or approaching quota limits'}

        cycle_start = time.time()
        logger.info("🔄 Starting scraping cycle...")

        try:
            # Phase 1: Scrape Reddit data
            all_posts_data = []
            all_comments_data = []

            # Get posts from multiple sources
            hot_posts = self.scraper.scrape_posts(sort_type='hot', limit=25)
            all_posts_data.extend(hot_posts)
            logger.info(f"📈 Hot posts: {len(hot_posts)}")

            new_posts = self.scraper.scrape_posts(sort_type='new', limit=15)
            all_posts_data.extend(new_posts)
            logger.info(f"🆕 New posts: {len(new_posts)}")

            rising_posts = self.scraper.scrape_posts(sort_type='rising', limit=15)
            all_posts_data.extend(rising_posts)
            logger.info(f"📊 Rising posts: {len(rising_posts)}")

            # Smart deduplication
            unique_posts = {}
            new_posts_count = 0

            for post in all_posts_data:
                post_id = post['id']
                if (post_id not in unique_posts and
                    post_id not in self.processed_posts and
                    (len(post.get('tickers', [])) > 0 or post.get('score', 0) > 100)):
                    unique_posts[post_id] = post
                    new_posts_count += 1

            final_posts = list(unique_posts.values())
            logger.info(f"📝 New posts to process: {new_posts_count}")

            # Phase 2: Comment scraping
            high_value_posts = [
                post for post in final_posts
                if (len(post.get('tickers', [])) >= 1 and
                    post.get('num_comments', 0) > 5 and
                    post.get('score', 0) > 50)
            ]

            high_value_posts.sort(
                key=lambda x: (len(x.get('tickers', [])), x.get('num_comments', 0), x.get('score', 0)),
                reverse=True
            )

            comments_target_posts = high_value_posts[:8]

            for post in comments_target_posts:
                try:
                    post_comments = self.scraper.scrape_comments(post['id'], limit=25)
                    new_comments = [
                        comment for comment in post_comments
                        if (comment['id'] not in self.processed_comments and
                            (len(comment.get('tickers', [])) > 0 or comment.get('score', 0) > 10))
                    ]
                    all_comments_data.extend(new_comments)
                    logger.info(f"💬 Post {post['id']}: {len(new_comments)} new quality comments")
                    time.sleep(2)
                except Exception as e:
                    logger.warning(f"Failed to scrape comments for {post['id']}: {e}")
                    continue

            # Phase 3: Sentiment Analysis
            sentiment_data = []
            ticker_sentiment_summary = {}

            if final_posts or all_comments_data:
                logger.info("🧠 Starting sentiment analysis...")
                all_content = final_posts + all_comments_data

                ticker_counts = {}
                for item in all_content:
                    for ticker in item.get('tickers', []):
                        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

                significant_tickers = [
                    ticker for ticker, count in ticker_counts.items()
                    if count >= 2
                ][:20]

                logger.info(f"🎯 Analyzing sentiment for {len(significant_tickers)} significant tickers")

                for ticker in significant_tickers:
                    try:
                        ticker_sentiment = self.sentiment_analyzer.analyze_ticker_sentiment(
                            all_content, ticker
                        )
                        ticker_sentiment_summary[ticker] = ticker_sentiment

                        sentiment_record = {
                            'ticker': ticker,
                            'sentiment': ticker_sentiment['sentiment'],
                            'confidence': ticker_sentiment['confidence'],
                            'numerical_score': ticker_sentiment['numerical_score'],
                            'mention_count': ticker_sentiment['mention_count'],
                            'sentiment_distribution': ticker_sentiment['sentiment_distribution'],
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'cycle_id': f"cycle_{int(time.time())}"
                        }
                        sentiment_data.append(sentiment_record)

                        logger.info(f"💭 {ticker}: {ticker_sentiment['sentiment']} (confidence: {ticker_sentiment['confidence']:.2f})")

                    except Exception as e:
                        logger.warning(f"Sentiment analysis failed for {ticker}: {e}")
                        continue

            # Phase 4: Save to Firebase
            if final_posts or all_comments_data:
                if final_posts:
                    saved_posts = self.firebase_manager.batch_save('reddit_posts', final_posts, 'id')
                    logger.info(f"💾 Saved {saved_posts} posts to Firebase")

                if all_comments_data:
                    saved_comments = self.firebase_manager.batch_save('reddit_posts_comments', all_comments_data, 'id')
                    logger.info(f"💾 Saved {saved_comments} comments to Firebase")

                if sentiment_data:
                    saved_sentiment = self.firebase_manager.save_sentiment_analysis(sentiment_data)
                    logger.info(f"💾 Saved sentiment analysis for {saved_sentiment} tickers")

                # Update processed caches
                for post in final_posts:
                    self.processed_posts.add(post['id'])
                for comment in all_comments_data:
                    self.processed_comments.add(comment['id'])

                # Keep cache sizes reasonable
                if len(self.processed_posts) > 15000:
                    self.processed_posts = set(list(self.processed_posts)[-7500:])
                if len(self.processed_comments) > 30000:
                    self.processed_comments = set(list(self.processed_comments)[-15000:])

                # Update statistics
                self.total_posts_scraped += len(final_posts)
                self.total_comments_scraped += len(all_comments_data)

                # Collect cycle tickers
                cycle_tickers = set()
                for post in final_posts:
                    cycle_tickers.update(post.get('tickers', []))
                for comment in all_comments_data:
                    cycle_tickers.update(comment.get('tickers', []))

                self.total_tickers_found.update(cycle_tickers)

                # Calculate sentiment summary
                bullish_tickers = [t for t, s in ticker_sentiment_summary.items() if s['sentiment'] == 'bullish']
                bearish_tickers = [t for t, s in ticker_sentiment_summary.items() if s['sentiment'] == 'bearish']

                # Update last scrape time
                self.last_scrape_time = time.time()

                # Cycle statistics
                cycle_time = time.time() - cycle_start

                stats = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'posts_scraped': len(final_posts),
                    'comments_scraped': len(all_comments_data),
                    'new_posts_processed': new_posts_count,
                    'unique_tickers_this_cycle': len(cycle_tickers),
                    'tickers_this_cycle': list(cycle_tickers),
                    'sentiment_summary': {
                        'bullish_tickers': bullish_tickers,
                        'bearish_tickers': bearish_tickers,
                        'neutral_tickers': [t for t in cycle_tickers if t not in bullish_tickers and t not in bearish_tickers],
                        'total_analyzed': len(ticker_sentiment_summary)
                    },
                    'cycle_time_seconds': round(cycle_time, 2),
                    'total_posts_scraped': self.total_posts_scraped,
                    'total_comments_scraped': self.total_comments_scraped,
                    'total_unique_tickers': len(self.total_tickers_found),
                    'quota_status': self.firebase_manager.get_quota_status()
                }

                logger.info(f"✅ Scraping cycle complete: {len(final_posts)} posts, {len(all_comments_data)} comments")
                logger.info(f"🎯 Tickers found: {list(cycle_tickers)}")
                logger.info(f"📊 Sentiment: 🐂{len(bullish_tickers)} bullish, 🐻{len(bearish_tickers)} bearish")

                # Phase 5: AUTOMATED ANALYSIS TRIGGER
                if self.analysis_engine:
                    logger.info("🚀 Triggering automated analysis of top trending tickers...")

                    try:
                        # Wait a moment for data to settle
                        time.sleep(5)

                        # Run automated analysis
                        analysis_result = self.analysis_engine.run_automated_analysis()

                        if analysis_result['status'] == 'completed':
                            self.total_analyses_run += 1
                            self.last_analysis_time = time.time()

                            stats['automated_analysis'] = {
                                'status': 'completed',
                                'tickers_analyzed': analysis_result.get('tickers_analyzed', 0),
                                'successful_analyses': analysis_result.get('successful_analyses', 0),
                                'duration_seconds': analysis_result.get('total_duration_seconds', 0),
                                'tickers': analysis_result.get('tickers', [])
                            }

                            logger.info(f"🤖 Automated analysis complete!")
                            logger.info(f"   📊 Analyzed {analysis_result.get('tickers_analyzed', 0)} tickers")
                            logger.info(f"   ✅ {analysis_result.get('successful_analyses', 0)} successful")
                            logger.info(f"   ⏱️ Duration: {analysis_result.get('total_duration_seconds', 0):.1f}s")
                            logger.info(f"   🎯 Tickers: {', '.join(analysis_result.get('tickers', []))}")
                        else:
                            logger.warning(f"⚠️ Automated analysis failed: {analysis_result.get('error', 'Unknown error')}")
                            stats['automated_analysis'] = {
                                'status': 'failed',
                                'error': analysis_result.get('error', 'Unknown error')
                            }

                    except Exception as e:
                        logger.error(f"❌ Automated analysis exception: {e}")
                        stats['automated_analysis'] = {
                            'status': 'error',
                            'error': str(e)
                        }
                else:
                    logger.info("⚠️ Automated analysis engine not available")
                    stats['automated_analysis'] = {
                        'status': 'unavailable',
                        'reason': 'Analysis engine not initialized'
                    }

                return stats

            else:
                logger.info("No new valuable posts to process in this cycle")
                return {'info': 'No new valuable posts processed'}

        except Exception as e:
            logger.error(f"❌ Scraping cycle failed: {e}")
            return {'error': str(e)}

    def cleanup_old_data(self):
        """Clean up old data and analysis results"""
        try:
            # Delete old posts, comments, sentiment data
            deleted_posts = self.firebase_manager.delete_old_data('reddit_posts', days=7)
            deleted_comments = self.firebase_manager.delete_old_data('reddit_posts_comments', days=7)
            deleted_sentiment = self.firebase_manager.delete_old_data('sentiment_analysis', days=7)

            # Delete old automated analysis results
            deleted_analysis = self.firebase_manager.delete_old_data('automated_analysis', days=3)

            logger.info(f"🗑️ Cleanup: {deleted_posts} posts, {deleted_comments} comments, "
                       f"{deleted_sentiment} sentiment, {deleted_analysis} analysis records deleted")

            # Clear local caches
            self.processed_posts.clear()
            self.processed_comments.clear()
            self.firebase_manager.clear_cache()

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

    def get_status(self) -> Dict:
        """Get current scraper status"""
        uptime = datetime.now(timezone.utc) - self.start_time

        return {
            'status': 'running',
            'version': 'enhanced_with_automated_analysis',
            'uptime_hours': round(uptime.total_seconds() / 3600, 2),
            'total_posts_scraped': self.total_posts_scraped,
            'total_comments_scraped': self.total_comments_scraped,
            'total_unique_tickers': len(self.total_tickers_found),
            'total_analyses_run': self.total_analyses_run,
            'processed_posts_cache_size': len(self.processed_posts),
            'processed_comments_cache_size': len(self.processed_comments),
            'min_cycle_interval_minutes': self.min_cycle_interval / 15,
            'minutes_since_last_scrape': (time.time() - self.last_scrape_time) / 15,
            'minutes_since_last_analysis': (time.time() - self.last_analysis_time) / 15 if self.last_analysis_time > 0 else None,
            'all_tickers_found': sorted(list(self.total_tickers_found)),
            'firebase_quota_status': self.firebase_manager.get_quota_status(),
            'automated_analysis_available': self.analysis_engine is not None,
            'last_check': datetime.now(timezone.utc).isoformat()
        }

    def run_single_cycle(self):
        """Run a single scraping + analysis cycle for testing"""
        logger.info("🧪 Running single enhanced cycle with automated analysis...")
        stats = self.scrape_cycle()

        print("\n📊 ENHANCED CYCLE RESULTS:")
        print(f"Posts: {stats.get('posts_scraped', 0)}")
        print(f"Comments: {stats.get('comments_scraped', 0)}")
        print(f"New posts: {stats.get('new_posts_processed', 0)}")
        print(f"Tickers: {stats.get('tickers_this_cycle', [])}")
        print(f"Scraping time: {stats.get('cycle_time_seconds', 0)}s")

        # Automated analysis results
        analysis_info = stats.get('automated_analysis', {})
        if analysis_info:
            print(f"\n🤖 AUTOMATED ANALYSIS:")
            print(f"Status: {analysis_info.get('status', 'unknown')}")
            if analysis_info.get('status') == 'completed':
                print(f"Tickers analyzed: {analysis_info.get('tickers_analyzed', 0)}")
                print(f"Successful analyses: {analysis_info.get('successful_analyses', 0)}")
                print(f"Analysis time: {analysis_info.get('duration_seconds', 0):.1f}s")
                print(f"Analyzed tickers: {', '.join(analysis_info.get('tickers', []))}")
            elif 'error' in analysis_info:
                print(f"Error: {analysis_info['error']}")

        return stats

    def start_continuous_scraping(self):
        """Start continuous scraping with automated analysis"""
        logger.info("🔄 Starting enhanced continuous scraping every 15 minutes...")
        logger.info("🤖 Automated analysis will run after each scraping cycle")
        logger.info("📊 Top 5 trending tickers will be analyzed with AI + ML")

        # Schedule every 15 minutes
        schedule.every(15).minutes.do(self.scrape_cycle)

        # Schedule weekly cleanup
        schedule.every().sunday.at("02:00").do(self.cleanup_old_data)

        # Run initial cycle
        logger.info("🚀 Running initial enhanced scraping + analysis cycle...")
        self.scrape_cycle()

        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("⏹️ Enhanced continuous scraping stopped by user")
        except Exception as e:
            logger.error(f"❌ Enhanced continuous scraping error: {e}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Reddit Continuous Scraper with Automated Analysis')
    parser.add_argument('--test', action='store_true', help='Run single cycle test')
    parser.add_argument('--continuous', action='store_true', help='Start continuous scraping')
    parser.add_argument('--status', action='store_true', help='Show current status')

    args = parser.parse_args()

    scraper = EnhancedContinuousRedditScraper()

    if args.test:
        scraper.run_single_cycle()
    elif args.continuous:
        scraper.start_continuous_scraping()
    elif args.status:
        status = scraper.get_status()
        print(f"📊 Enhanced Status: {status}")
    else:
        # Default: run single test cycle
        print("🧪 Running enhanced test cycle (use --continuous for continuous mode)")
        scraper.run_single_cycle()


if __name__ == "__main__":
    main()