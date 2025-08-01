"""
Ultra-Optimized Continuous Reddit Scraper
Reduces Firebase writes from excessive levels to sustainable amounts
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
from data.firebase_manager import FirebaseManager  # Uses ultra-optimized manager
from processing.sentiment_analyzer import FinancialSentimentAnalyzer
from config.settings import APP_CONFIG

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


class UltraOptimizedContinuousRedditScraper:
    """Ultra-optimized continuous scraper - dramatically reduced Firebase usage"""

    def __init__(self):
        """Initialize ultra-optimized continuous scraper"""
        # Use singleton Firebase manager to avoid repeated connections
        self.firebase_manager = FirebaseManager()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()

        # Initialize Reddit scraper (will reuse our Firebase manager)
        self.scraper = RedditScraper()
        self.scraper.firebase_manager = self.firebase_manager  # Reuse connection

        # Statistics
        self.total_posts_scraped = 0
        self.total_comments_scraped = 0
        self.total_tickers_found = set()
        self.start_time = datetime.now(timezone.utc)

        # ULTRA-AGGRESSIVE deduplication cache to avoid processing same posts
        self.processed_posts = set()
        self.processed_comments = set()

        # Last scrape tracking to avoid frequent cycles
        self.last_scrape_time = 0
        self.min_cycle_interval = 3600  # MINIMUM 1 hour between cycles (was 15 minutes)

        logger.info("🚀 ULTRA-Optimized Continuous Reddit Scraper initialized")

    def should_run_cycle(self) -> bool:
        """Check if enough time has passed since last cycle"""
        current_time = time.time()
        time_since_last = current_time - self.last_scrape_time

        if time_since_last < self.min_cycle_interval:
            remaining = self.min_cycle_interval - time_since_last
            logger.info(f"⏳ Skipping cycle - {remaining/60:.1f} minutes remaining until next allowed cycle")
            return False

        # Check Firebase quota before proceeding
        quota_status = self.firebase_manager.get_quota_status()
        if not quota_status.get('quota_healthy', True):
            logger.warning("🚨 Firebase quota near limit - skipping scrape cycle")
            return False

        return True

    def scrape_cycle(self) -> Dict:
        """Execute one ULTRA-CONSERVATIVE scraping cycle"""
        if not self.should_run_cycle():
            return {'skipped': 'Too soon or quota limits'}

        cycle_start = time.time()
        logger.info("🔄 Starting ULTRA-CONSERVATIVE scraping cycle...")

        try:
            # Phase 1: DRAMATICALLY reduced scraping limits
            all_posts_data = []
            all_comments_data = []

            # ULTRA-REDUCED limits to minimize Reddit API calls
            hot_posts = self.scraper.scrape_posts(sort_type='hot', limit=8)    # REDUCED from 15 to 8
            all_posts_data.extend(hot_posts)
            logger.info(f"📈 Hot posts: {len(hot_posts)}")

            new_posts = self.scraper.scrape_posts(sort_type='new', limit=5)    # REDUCED from 10 to 5
            all_posts_data.extend(new_posts)
            logger.info(f"🆕 New posts: {len(new_posts)}")

            # Skip rising posts to reduce API calls further
            # rising_posts = self.scraper.scrape_posts(sort_type='rising', limit=8)
            # all_posts_data.extend(rising_posts)
            # logger.info(f"📊 Rising posts: {len(rising_posts)}")

            # AGGRESSIVE deduplication and filtering
            unique_posts = {}
            new_posts_count = 0

            for post in all_posts_data:
                post_id = post['id']
                # Only process posts we haven't seen AND that have tickers
                if (post_id not in unique_posts and
                    post_id not in self.processed_posts and
                    len(post.get('tickers', [])) > 0):  # ONLY posts with tickers
                    unique_posts[post_id] = post
                    new_posts_count += 1

            final_posts = list(unique_posts.values())
            logger.info(f"📝 New posts with tickers to process: {new_posts_count}")

            # Phase 2: ULTRA-CONSERVATIVE comment scraping
            # Only scrape comments from posts with multiple tickers (high value)
            high_value_posts = [
                post for post in final_posts
                if len(post.get('tickers', [])) >= 2 and post.get('num_comments', 0) > 10
            ]

            # Sort by engagement and limit to top posts
            high_value_posts.sort(
                key=lambda x: (len(x.get('tickers', [])), x.get('num_comments', 0)),
                reverse=True
            )

            # DRASTICALLY reduce comment scraping to save quota
            comments_target_posts = high_value_posts[:3]  # REDUCED from 5 to 3

            for post in comments_target_posts:
                try:
                    post_comments = self.scraper.scrape_comments(
                        post['id'],
                        limit=15  # REDUCED from 20 to 15
                    )

                    # Filter out already processed comments AND require tickers
                    new_comments = [
                        comment for comment in post_comments
                        if (comment['id'] not in self.processed_comments and
                            len(comment.get('tickers', [])) > 0)  # ONLY comments with tickers
                    ]

                    all_comments_data.extend(new_comments)
                    logger.info(f"💬 Post {post['id']}: {len(new_comments)} new comments with tickers")

                    # Longer pause to be respectful
                    time.sleep(3)  # INCREASED from 1 to 3 seconds

                except Exception as e:
                    logger.warning(f"Failed to scrape comments for {post['id']}: {e}")
                    continue

            # Phase 3: ULTRA-SELECTIVE Sentiment Analysis
            sentiment_data = []
            ticker_sentiment_summary = {}

            if final_posts or all_comments_data:
                logger.info("🧠 Starting ULTRA-SELECTIVE sentiment analysis...")

                # Combine content for analysis
                all_content = final_posts + all_comments_data

                # Get unique tickers, but ONLY the most significant ones
                ticker_counts = {}
                for item in all_content:
                    for ticker in item.get('tickers', []):
                        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

                # ULTRA-SELECTIVE: Only analyze tickers with 3+ mentions (was 2+)
                significant_tickers = [
                    ticker for ticker, count in ticker_counts.items()
                    if count >= 3
                ][:10]  # Limit to top 10 tickers (was 15)

                logger.info(f"🎯 Analyzing sentiment for {len(significant_tickers)} HIGHLY significant tickers")

                # Analyze sentiment for only the most significant tickers
                for ticker in significant_tickers:
                    try:
                        ticker_sentiment = self.sentiment_analyzer.analyze_ticker_sentiment(
                            all_content, ticker
                        )
                        ticker_sentiment_summary[ticker] = ticker_sentiment

                        # Create sentiment record
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

            # Phase 4: CONSERVATIVE Firebase saving
            if final_posts or all_comments_data:
                # Save using optimized batch saving with aggressive deduplication
                if final_posts:
                    saved_posts = self.firebase_manager.batch_save(
                        'reddit_posts',
                        final_posts,
                        'id'
                    )
                    logger.info(f"💾 Saved {saved_posts} posts to Firebase")

                if all_comments_data:
                    saved_comments = self.firebase_manager.batch_save(
                        'reddit_posts_comments',
                        all_comments_data,
                        'id'
                    )
                    logger.info(f"💾 Saved {saved_comments} comments to Firebase")

                # Save sentiment analysis
                if sentiment_data:
                    saved_sentiment = self.firebase_manager.save_sentiment_analysis(sentiment_data)
                    logger.info(f"💾 Saved sentiment analysis for {saved_sentiment} tickers")

                # Update processed caches with LARGER limits
                for post in final_posts:
                    self.processed_posts.add(post['id'])
                for comment in all_comments_data:
                    self.processed_comments.add(comment['id'])

                # Keep cache sizes manageable but larger
                if len(self.processed_posts) > 10000:  # INCREASED from 5000
                    self.processed_posts = set(list(self.processed_posts)[-5000:])
                if len(self.processed_comments) > 20000:  # INCREASED from 10000
                    self.processed_comments = set(list(self.processed_comments)[-10000:])

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

                logger.info(f"✅ ULTRA-CONSERVATIVE cycle complete: {len(final_posts)} posts, {len(all_comments_data)} comments")
                logger.info(f"🎯 Tickers found: {list(cycle_tickers)}")
                logger.info(f"📊 Sentiment: 🐂{len(bullish_tickers)} bullish, 🐻{len(bearish_tickers)} bearish")
                logger.info(f"🔥 Quota status: {stats['quota_status']}")

                return stats

            else:
                logger.warning("No new high-value posts to process in this cycle")
                return {'info': 'No new high-value posts processed'}

        except Exception as e:
            logger.error(f"❌ ULTRA-CONSERVATIVE scraping cycle failed: {e}")
            return {'error': str(e)}

    def cleanup_old_data(self):
        """Clean up old data with CONSERVATIVE deletion"""
        try:
            # Delete data older than 5 days (was 7 days) with smaller batches
            deleted_posts = self.firebase_manager.delete_old_data('reddit_posts', days=5)
            deleted_comments = self.firebase_manager.delete_old_data('reddit_posts_comments', days=5)
            deleted_sentiment = self.firebase_manager.delete_old_data('sentiment_analysis', days=5)

            logger.info(f"🗑️ CONSERVATIVE cleanup: {deleted_posts} posts, {deleted_comments} comments, {deleted_sentiment} sentiment records deleted")

            # Clear local caches
            self.processed_posts.clear()
            self.processed_comments.clear()
            self.firebase_manager.clear_cache()

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

    def get_status(self) -> Dict:
        """Get current scraper status with quota information"""
        uptime = datetime.now(timezone.utc) - self.start_time

        return {
            'status': 'running',
            'optimization_level': 'ULTRA_CONSERVATIVE',
            'uptime_hours': round(uptime.total_seconds() / 3600, 2),
            'total_posts_scraped': self.total_posts_scraped,
            'total_comments_scraped': self.total_comments_scraped,
            'total_unique_tickers': len(self.total_tickers_found),
            'processed_posts_cache_size': len(self.processed_posts),
            'processed_comments_cache_size': len(self.processed_comments),
            'min_cycle_interval_minutes': self.min_cycle_interval / 60,
            'minutes_since_last_scrape': (time.time() - self.last_scrape_time) / 60,
            'all_tickers_found': sorted(list(self.total_tickers_found)),
            'firebase_quota_status': self.firebase_manager.get_quota_status(),
            'last_check': datetime.now(timezone.utc).isoformat()
        }

    def run_single_cycle(self):
        """Run a single ULTRA-CONSERVATIVE scraping cycle (for testing)"""
        logger.info("🧪 Running single ULTRA-CONSERVATIVE test cycle...")
        stats = self.scrape_cycle()
        print("\n📊 ULTRA-CONSERVATIVE CYCLE RESULTS:")
        print(f"Posts: {stats.get('posts_scraped', 0)}")
        print(f"Comments: {stats.get('comments_scraped', 0)}")
        print(f"New posts: {stats.get('new_posts_processed', 0)}")
        print(f"Tickers: {stats.get('tickers_this_cycle', [])}")
        print(f"Time: {stats.get('cycle_time_seconds', 0)}s")
        print(f"Quota status: {stats.get('quota_status', {})}")
        return stats

    def start_continuous_scraping(self):
        """Start the ULTRA-CONSERVATIVE continuous scraping schedule"""
        logger.info("🔄 Starting ULTRA-CONSERVATIVE continuous scraping every 90 minutes...")
        logger.info("⚡ Using singleton Firebase manager with 1-hour caching for maximum efficiency")
        logger.info("🎯 ULTRA-SELECTIVE: Only processing posts/comments with tickers")

        # MUCH longer interval to reduce Firebase load dramatically
        schedule.every(90).minutes.do(self.scrape_cycle)  # INCREASED from 20 to 90 minutes

        # Schedule weekly cleanup (unchanged)
        schedule.every().sunday.at("02:00").do(self.cleanup_old_data)

        # Run initial cycle immediately
        logger.info("🚀 Running initial ULTRA-CONSERVATIVE scraping cycle...")
        self.scrape_cycle()

        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(300)  # Check every 5 minutes (was 1 minute)

        except KeyboardInterrupt:
            logger.info("⏹️ ULTRA-CONSERVATIVE continuous scraping stopped by user")
        except Exception as e:
            logger.error(f"❌ ULTRA-CONSERVATIVE continuous scraping error: {e}")


def main():
    """Main function for testing or running"""
    import argparse

    parser = argparse.ArgumentParser(description='ULTRA-Optimized Reddit Continuous Scraper')
    parser.add_argument('--test', action='store_true', help='Run single cycle test')
    parser.add_argument('--continuous', action='store_true', help='Start continuous scraping')
    parser.add_argument('--status', action='store_true', help='Show current status')

    args = parser.parse_args()

    scraper = UltraOptimizedContinuousRedditScraper()

    if args.test:
        scraper.run_single_cycle()
    elif args.continuous:
        scraper.start_continuous_scraping()
    elif args.status:
        status = scraper.get_status()
        print(f"📊 ULTRA-CONSERVATIVE Status: {status}")
    else:
        # Default: run single test cycle
        print("🧪 Running ULTRA-CONSERVATIVE test cycle (use --continuous for continuous mode)")
        scraper.run_single_cycle()


if __name__ == "__main__":
    main()