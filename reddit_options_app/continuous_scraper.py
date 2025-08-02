"""
FIXED Enhanced Continuous Reddit Scraper with Automated Analysis
Simplified timing mechanism for reliable 15-minute intervals
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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

# Set up logging with better configuration
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_scraper.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FixedContinuousRedditScraper:
    """Fixed continuous scraper with simplified, reliable timing"""

    def __init__(self):
        """Initialize fixed continuous scraper"""

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
        self.cycle_count = 0

        # Deduplication cache
        self.processed_posts = set()
        self.processed_comments = set()

        # SIMPLIFIED TIMING - just use interval
        self.cycle_interval_minutes = 15
        self.last_cycle_start = 0

        logger.info("🚀 Fixed Continuous Reddit Scraper initialized")
        logger.info(f"⏰ Will run every {self.cycle_interval_minutes} minutes")

    def should_run_cycle(self) -> tuple[bool, str]:
        """Simplified cycle check - returns (should_run, reason)"""
        current_time = time.time()

        # Check if enough time has passed (simplified)
        if self.last_cycle_start > 0:
            time_since_last = current_time - self.last_cycle_start
            time_needed = self.cycle_interval_minutes * 60

            if time_since_last < time_needed:
                remaining_minutes = (time_needed - time_since_last) / 60
                return False, f"Next cycle in {remaining_minutes:.1f} minutes"

        # Check Firebase quota (but be less restrictive)
        try:
            quota_status = self.firebase_manager.get_quota_status()
            reads_today = quota_status.get('reads_today', 0)
            daily_limit = quota_status.get('daily_limit', 35000)

            # Only stop if we're at 95% of limit (less restrictive)
            if reads_today >= daily_limit * 0.95:
                return False, f"Firebase quota at 95% ({reads_today}/{daily_limit})"

            logger.info(f"📊 Firebase quota: {reads_today}/{daily_limit} ({reads_today/daily_limit*100:.1f}%)")

        except Exception as e:
            logger.warning(f"⚠️ Could not check quota: {e} - continuing anyway")

        return True, "Ready to run"

    def scrape_cycle(self) -> Dict:
        """Execute one scraping cycle with improved error handling"""

        # Mark cycle start immediately
        self.last_cycle_start = time.time()
        self.cycle_count += 1

        cycle_start_time = datetime.now(timezone.utc)
        logger.info(f"🔄 Starting scraping cycle #{self.cycle_count} at {cycle_start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        try:
            # Phase 1: Scrape Reddit data
            all_posts_data = []
            all_comments_data = []

            # Get posts from multiple sources
            logger.info("📈 Scraping hot posts...")
            hot_posts = self.scraper.scrape_posts(sort_type='hot', limit=25)
            all_posts_data.extend(hot_posts)
            logger.info(f"   Hot posts: {len(hot_posts)}")

            logger.info("🆕 Scraping new posts...")
            new_posts = self.scraper.scrape_posts(sort_type='new', limit=15)
            all_posts_data.extend(new_posts)
            logger.info(f"   New posts: {len(new_posts)}")

            logger.info("📊 Scraping rising posts...")
            rising_posts = self.scraper.scrape_posts(sort_type='rising', limit=15)
            all_posts_data.extend(rising_posts)
            logger.info(f"   Rising posts: {len(rising_posts)}")

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

            # Phase 2: Comment scraping (only for high-value posts)
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

            comments_target_posts = high_value_posts[:8]  # Limit to top 8

            for post in comments_target_posts:
                try:
                    logger.info(f"💬 Scraping comments for post {post['id'][:8]}...")
                    post_comments = self.scraper.scrape_comments(post['id'], limit=25)
                    new_comments = [
                        comment for comment in post_comments
                        if (comment['id'] not in self.processed_comments and
                            (len(comment.get('tickers', [])) > 0 or comment.get('score', 0) > 10))
                    ]
                    all_comments_data.extend(new_comments)
                    logger.info(f"   Added {len(new_comments)} new quality comments")
                    time.sleep(2)  # Be respectful to Reddit
                except Exception as e:
                    logger.warning(f"Failed to scrape comments for {post['id']}: {e}")
                    continue

            # Phase 3: Sentiment Analysis
            sentiment_data = []
            ticker_sentiment_summary = {}

            if final_posts or all_comments_data:
                logger.info("🧠 Starting sentiment analysis...")
                all_content = final_posts + all_comments_data

                # Count ticker mentions
                ticker_counts = {}
                for item in all_content:
                    for ticker in item.get('tickers', []):
                        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

                # Analyze tickers with at least 2 mentions
                significant_tickers = [
                    ticker for ticker, count in ticker_counts.items()
                    if count >= 2
                ][:20]  # Limit to top 20

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
                            'cycle_id': f"cycle_{self.cycle_count}_{int(time.time())}"
                        }
                        sentiment_data.append(sentiment_record)

                        logger.info(f"💭 {ticker}: {ticker_sentiment['sentiment']} (confidence: {ticker_sentiment['confidence']:.2f})")

                    except Exception as e:
                        logger.warning(f"Sentiment analysis failed for {ticker}: {e}")
                        continue

            # Phase 4: Save to Firebase
            saved_posts = 0
            saved_comments = 0
            saved_sentiment = 0

            if final_posts:
                try:
                    saved_posts = self.firebase_manager.batch_save('reddit_posts', final_posts, 'id')
                    logger.info(f"💾 Saved {saved_posts} posts to Firebase")
                except Exception as e:
                    logger.error(f"❌ Failed to save posts: {e}")

            if all_comments_data:
                try:
                    saved_comments = self.firebase_manager.batch_save('reddit_posts_comments', all_comments_data, 'id')
                    logger.info(f"💾 Saved {saved_comments} comments to Firebase")
                except Exception as e:
                    logger.error(f"❌ Failed to save comments: {e}")

            if sentiment_data:
                try:
                    saved_sentiment = self.firebase_manager.save_sentiment_analysis(sentiment_data)
                    logger.info(f"💾 Saved sentiment analysis for {saved_sentiment} tickers")
                except Exception as e:
                    logger.error(f"❌ Failed to save sentiment: {e}")

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

            # Cycle statistics
            cycle_time = time.time() - self.last_cycle_start

            stats = {
                'cycle_number': self.cycle_count,
                'timestamp': cycle_start_time.isoformat(),
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
                'firebase_saves': {
                    'posts': saved_posts,
                    'comments': saved_comments,
                    'sentiment': saved_sentiment
                }
            }

            logger.info(f"✅ Scraping cycle #{self.cycle_count} complete!")
            logger.info(f"   📊 {len(final_posts)} posts, {len(all_comments_data)} comments in {cycle_time:.1f}s")
            logger.info(f"   🎯 Tickers: {list(cycle_tickers)}")
            logger.info(f"   📈 Sentiment: 🐂{len(bullish_tickers)} bullish, 🐻{len(bearish_tickers)} bearish")

            # Phase 5: AUTOMATED ANALYSIS TRIGGER
            if self.analysis_engine and cycle_tickers:
                logger.info("🚀 Triggering automated analysis of top trending tickers...")

                try:
                    # Wait a moment for data to settle
                    time.sleep(5)

                    # Run automated analysis
                    analysis_result = self.analysis_engine.run_automated_analysis()

                    if analysis_result['status'] == 'completed':
                        self.total_analyses_run += 1

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
                if not self.analysis_engine:
                    logger.info("⚠️ Automated analysis engine not available")
                else:
                    logger.info("ℹ️ No tickers found - skipping automated analysis")

                stats['automated_analysis'] = {
                    'status': 'skipped',
                    'reason': 'No analysis engine or no tickers' if not self.analysis_engine else 'No tickers found'
                }

            return stats

        except Exception as e:
            cycle_time = time.time() - self.last_cycle_start
            logger.error(f"❌ Scraping cycle #{self.cycle_count} failed after {cycle_time:.1f}s: {e}")
            return {
                'cycle_number': self.cycle_count,
                'error': str(e),
                'timestamp': cycle_start_time.isoformat(),
                'cycle_time_seconds': round(cycle_time, 2)
            }

    def start_continuous_scraping(self):
        """Start continuous scraping with SIMPLIFIED timing"""
        logger.info("🔄 Starting FIXED continuous scraping...")
        logger.info(f"⏰ Will run every {self.cycle_interval_minutes} minutes")
        logger.info("🤖 Automated analysis will run after each scraping cycle")

        # Run initial cycle
        logger.info("🚀 Running initial scraping + analysis cycle...")
        initial_result = self.scrape_cycle()
        if 'error' in initial_result:
            logger.error(f"❌ Initial cycle failed: {initial_result['error']}")
        else:
            logger.info("✅ Initial cycle completed successfully")

        # SIMPLIFIED main loop - no external scheduling library
        logger.info("⏰ Starting main timing loop...")

        try:
            while True:
                # Check if it's time to run
                should_run, reason = self.should_run_cycle()

                if should_run:
                    logger.info("🎯 Time for next cycle!")
                    result = self.scrape_cycle()

                    if 'error' in result:
                        logger.error(f"❌ Cycle failed: {result['error']}")
                    else:
                        logger.info(f"✅ Cycle #{result.get('cycle_number')} completed")
                else:
                    # Log waiting status every 5 minutes
                    current_time = time.time()
                    if not hasattr(self, '_last_waiting_log') or current_time - self._last_waiting_log > 300:
                        logger.info(f"⏳ Waiting: {reason}")
                        self._last_waiting_log = current_time

                # Check every minute
                time.sleep(60)

        except KeyboardInterrupt:
            logger.info("⏹️ Continuous scraping stopped by user")
        except Exception as e:
            logger.error(f"❌ Continuous scraping error: {e}")

    def get_status(self) -> Dict:
        """Get current scraper status"""
        uptime = datetime.now(timezone.utc) - self.start_time

        # Calculate time until next cycle
        time_until_next = "Unknown"
        if self.last_cycle_start > 0:
            time_since_last = time.time() - self.last_cycle_start
            time_needed = self.cycle_interval_minutes * 60
            if time_since_last < time_needed:
                time_until_next = f"{(time_needed - time_since_last) / 60:.1f} minutes"
            else:
                time_until_next = "Ready to run"

        return {
            'status': 'running',
            'version': 'fixed_continuous_with_automated_analysis',
            'cycle_count': self.cycle_count,
            'uptime_hours': round(uptime.total_seconds() / 3600, 2),
            'cycle_interval_minutes': self.cycle_interval_minutes,
            'time_until_next_cycle': time_until_next,
            'total_posts_scraped': self.total_posts_scraped,
            'total_comments_scraped': self.total_comments_scraped,
            'total_unique_tickers': len(self.total_tickers_found),
            'total_analyses_run': self.total_analyses_run,
            'last_cycle_time': datetime.fromtimestamp(self.last_cycle_start).isoformat() if self.last_cycle_start > 0 else None,
            'all_tickers_found': sorted(list(self.total_tickers_found)),
            'automated_analysis_available': self.analysis_engine is not None,
            'last_check': datetime.now(timezone.utc).isoformat()
        }

    def run_single_cycle(self):
        """Run a single cycle for testing"""
        logger.info("🧪 Running single test cycle...")
        stats = self.scrape_cycle()

        print("\n📊 CYCLE RESULTS:")
        print(f"Cycle: #{stats.get('cycle_number', 'N/A')}")
        print(f"Posts: {stats.get('posts_scraped', 0)}")
        print(f"Comments: {stats.get('comments_scraped', 0)}")
        print(f"Tickers: {stats.get('tickers_this_cycle', [])}")
        print(f"Time: {stats.get('cycle_time_seconds', 0)}s")

        if 'automated_analysis' in stats:
            analysis_info = stats['automated_analysis']
            print(f"\n🤖 AUTOMATED ANALYSIS:")
            print(f"Status: {analysis_info.get('status', 'unknown')}")
            if analysis_info.get('status') == 'completed':
                print(f"Tickers analyzed: {analysis_info.get('tickers_analyzed', 0)}")
                print(f"Successful: {analysis_info.get('successful_analyses', 0)}")

        return stats


def main():
    """Main function with clearer options"""
    import argparse

    parser = argparse.ArgumentParser(description='FIXED Enhanced Reddit Continuous Scraper')
    parser.add_argument('--test', action='store_true', help='Run single cycle test')
    parser.add_argument('--continuous', action='store_true', help='Start continuous scraping (fixed timing)')
    parser.add_argument('--status', action='store_true', help='Show current status')

    args = parser.parse_args()

    scraper = FixedContinuousRedditScraper()

    if args.test:
        print("🧪 Running single test cycle...")
        scraper.run_single_cycle()
    elif args.continuous:
        print("🔄 Starting FIXED continuous scraping mode...")
        print("   - Simplified timing mechanism")
        print("   - Reliable 15-minute intervals")
        print("   - Better error handling")
        print("   - Enhanced logging")
        scraper.start_continuous_scraping()
    elif args.status:
        status = scraper.get_status()
        print(f"📊 Status: {status}")
    else:
        print("🧪 No arguments provided - running test cycle")
        print("Use --continuous for continuous mode")
        scraper.run_single_cycle()


if __name__ == "__main__":
    main()