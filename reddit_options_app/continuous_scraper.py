"""
Continuous Reddit Scraper with Realistic Resource Usage
Uses reasonable Firebase limits while maintaining efficiency
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


class ContinuousRedditScraper:
    """Continuous scraper with realistic resource usage"""

    def __init__(self):
        """Initialize continuous scraper"""
        # Use singleton Firebase manager to avoid repeated connections
        self.firebase_manager = FirebaseManager()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()

        # Initialize Reddit scraper
        self.scraper = RedditScraper()
        self.scraper.firebase_manager = self.firebase_manager

        # Statistics
        self.total_posts_scraped = 0
        self.total_comments_scraped = 0
        self.total_tickers_found = set()
        self.start_time = datetime.now(timezone.utc)

        # Reasonable deduplication cache
        self.processed_posts = set()
        self.processed_comments = set()

        # Cycle timing - reasonable intervals
        self.last_scrape_time = 0
        self.min_cycle_interval = 1800  # 30 minutes between cycles (reasonable)

        logger.info("🚀 Continuous Reddit Scraper initialized with realistic limits")

    def should_run_cycle(self) -> bool:
        """Check if enough time has passed since last cycle"""
        current_time = time.time()
        time_since_last = current_time - self.last_scrape_time

        if time_since_last < self.min_cycle_interval:
            remaining = self.min_cycle_interval - time_since_last
            logger.info(f"⏳ Next cycle in {remaining/60:.1f} minutes")
            return False

        # Check Firebase quota (but with realistic limits)
        quota_status = self.firebase_manager.get_quota_status()
        reads_today = quota_status.get('reads_today', 0)
        daily_limit = quota_status.get('daily_limit', 35000)

        # Only skip if we're actually near the limit (not ultra-conservative)
        if reads_today >= daily_limit * 0.9:  # 90% of limit
            logger.warning(f"🚨 Firebase quota at 90% ({reads_today}/{daily_limit}) - skipping cycle")
            return False

        return True

    def scrape_cycle(self) -> Dict:
        """Execute one scraping cycle with reasonable limits"""
        if not self.should_run_cycle():
            return {'skipped': 'Too soon or approaching quota limits'}

        cycle_start = time.time()
        logger.info("🔄 Starting scraping cycle...")

        try:
            # Phase 1: Reasonable scraping limits
            all_posts_data = []
            all_comments_data = []

            # Get posts from multiple sources with reasonable limits
            hot_posts = self.scraper.scrape_posts(sort_type='hot', limit=25)
            all_posts_data.extend(hot_posts)
            logger.info(f"📈 Hot posts: {len(hot_posts)}")

            new_posts = self.scraper.scrape_posts(sort_type='new', limit=15)
            all_posts_data.extend(new_posts)
            logger.info(f"🆕 New posts: {len(new_posts)}")

            rising_posts = self.scraper.scrape_posts(sort_type='rising', limit=10)
            all_posts_data.extend(rising_posts)
            logger.info(f"📊 Rising posts: {len(rising_posts)}")

            # Smart deduplication - keep posts we haven't seen with tickers
            unique_posts = {}
            new_posts_count = 0

            for post in all_posts_data:
                post_id = post['id']
                # Process posts we haven't seen that have tickers OR high scores
                if (post_id not in unique_posts and
                    post_id not in self.processed_posts and
                    (len(post.get('tickers', [])) > 0 or post.get('score', 0) > 100)):
                    unique_posts[post_id] = post
                    new_posts_count += 1

            final_posts = list(unique_posts.values())
            logger.info(f"📝 New posts to process: {new_posts_count}")

            # Phase 2: Reasonable comment scraping
            # Focus on posts with multiple tickers or high engagement
            high_value_posts = [
                post for post in final_posts
                if (len(post.get('tickers', [])) >= 1 and
                    post.get('num_comments', 0) > 5 and
                    post.get('score', 0) > 50)
            ]

            # Sort by engagement and limit reasonably
            high_value_posts.sort(
                key=lambda x: (len(x.get('tickers', [])), x.get('num_comments', 0), x.get('score', 0)),
                reverse=True
            )

            # Get comments from top posts
            comments_target_posts = high_value_posts[:8]  # Reasonable number

            for post in comments_target_posts:
                try:
                    post_comments = self.scraper.scrape_comments(
                        post['id'],
                        limit=25  # Reasonable comment limit
                    )

                    # Filter for new comments with tickers or high scores
                    new_comments = [
                        comment for comment in post_comments
                        if (comment['id'] not in self.processed_comments and
                            (len(comment.get('tickers', [])) > 0 or comment.get('score', 0) > 10))
                    ]

                    all_comments_data.extend(new_comments)
                    logger.info(f"💬 Post {post['id']}: {len(new_comments)} new quality comments")

                    # Reasonable pause
                    time.sleep(2)

                except Exception as e:
                    logger.warning(f"Failed to scrape comments for {post['id']}: {e}")
                    continue

            # Phase 3: Smart Sentiment Analysis
            sentiment_data = []
            ticker_sentiment_summary = {}

            if final_posts or all_comments_data:
                logger.info("🧠 Starting sentiment analysis...")

                # Combine content for analysis
                all_content = final_posts + all_comments_data

                # Get significant tickers (reasonable threshold)
                ticker_counts = {}
                for item in all_content:
                    for ticker in item.get('tickers', []):
                        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

                # Focus on tickers with 2+ mentions (reasonable threshold)
                significant_tickers = [
                    ticker for ticker, count in ticker_counts.items()
                    if count >= 2
                ][:20]  # Reasonable limit

                logger.info(f"🎯 Analyzing sentiment for {len(significant_tickers)} significant tickers")

                # Analyze sentiment for significant tickers
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

            # Phase 4: Save to Firebase
            if final_posts or all_comments_data:
                # Save using batch operations
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

                # Update processed caches with reasonable limits
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

                logger.info(f"✅ Cycle complete: {len(final_posts)} posts, {len(all_comments_data)} comments")
                logger.info(f"🎯 Tickers found: {list(cycle_tickers)}")
                logger.info(f"📊 Sentiment: 🐂{len(bullish_tickers)} bullish, 🐻{len(bearish_tickers)} bearish")

                return stats

            else:
                logger.info("No new valuable posts to process in this cycle")
                return {'info': 'No new valuable posts processed'}

        except Exception as e:
            logger.error(f"❌ Scraping cycle failed: {e}")
            return {'error': str(e)}

    def cleanup_old_data(self):
        """Clean up old data reasonably"""
        try:
            # Delete data older than 7 days (reasonable retention)
            deleted_posts = self.firebase_manager.delete_old_data('reddit_posts', days=7)
            deleted_comments = self.firebase_manager.delete_old_data('reddit_posts_comments', days=7)
            deleted_sentiment = self.firebase_manager.delete_old_data('sentiment_analysis', days=7)

            logger.info(f"🗑️ Cleanup: {deleted_posts} posts, {deleted_comments} comments, {deleted_sentiment} sentiment records deleted")

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
            'optimization_level': 'REALISTIC',
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
        """Run a single scraping cycle for testing"""
        logger.info("🧪 Running single test cycle...")
        stats = self.scrape_cycle()
        print("\n📊 CYCLE RESULTS:")
        print(f"Posts: {stats.get('posts_scraped', 0)}")
        print(f"Comments: {stats.get('comments_scraped', 0)}")
        print(f"New posts: {stats.get('new_posts_processed', 0)}")
        print(f"Tickers: {stats.get('tickers_this_cycle', [])}")
        print(f"Time: {stats.get('cycle_time_seconds', 0)}s")
        print(f"Quota status: {stats.get('quota_status', {})}")
        return stats

    def start_continuous_scraping(self):
        """Start continuous scraping with reasonable intervals"""
        logger.info("🔄 Starting continuous scraping every 30 minutes...")
        logger.info("📊 Using realistic Firebase limits with smart caching")
        logger.info("🎯 Focusing on posts/comments with tickers or high engagement")

        # Reasonable interval - every 30 minutes
        schedule.every(30).minutes.do(self.scrape_cycle)

        # Schedule weekly cleanup
        schedule.every().sunday.at("02:00").do(self.cleanup_old_data)

        # Run initial cycle
        logger.info("🚀 Running initial scraping cycle...")
        self.scrape_cycle()

        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("⏹️ Continuous scraping stopped by user")
        except Exception as e:
            logger.error(f"❌ Continuous scraping error: {e}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Reddit Continuous Scraper with Realistic Limits')
    parser.add_argument('--test', action='store_true', help='Run single cycle test')
    parser.add_argument('--continuous', action='store_true', help='Start continuous scraping')
    parser.add_argument('--status', action='store_true', help='Show current status')

    args = parser.parse_args()

    scraper = ContinuousRedditScraper()

    if args.test:
        scraper.run_single_cycle()
    elif args.continuous:
        scraper.start_continuous_scraping()
    elif args.status:
        status = scraper.get_status()
        print(f"📊 Status: {status}")
    else:
        # Default: run single test cycle
        print("🧪 Running test cycle (use --continuous for continuous mode)")
        scraper.run_single_cycle()


if __name__ == "__main__":
    main()