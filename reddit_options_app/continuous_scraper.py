"""
Optimized Continuous Reddit Scraper
Reduced Firebase API calls and improved efficiency
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
from data.firebase_manager import OptimizedFirebaseManager as FirebaseManager
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


class OptimizedContinuousRedditScraper:
    """Optimized continuous scraper with reduced Firebase calls"""

    def __init__(self):
        """Initialize optimized continuous scraper (singleton Firebase manager)"""
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

        # Deduplication cache to avoid processing same posts
        self.processed_posts = set()
        self.processed_comments = set()

        logger.info("🚀 Optimized Continuous Reddit Scraper initialized")

    def scrape_cycle(self) -> Dict:
        """Execute one optimized scraping cycle"""
        cycle_start = time.time()
        logger.info("🔄 Starting optimized scraping cycle...")

        try:
            # Phase 1: Scrape posts with reduced limits
            all_posts_data = []
            all_comments_data = []

            # Reduced limits to save API quota
            hot_posts = self.scraper.scrape_posts(sort_type='hot', limit=15)  # Reduced from 20
            all_posts_data.extend(hot_posts)
            logger.info(f"📈 Hot posts: {len(hot_posts)}")

            new_posts = self.scraper.scrape_posts(sort_type='new', limit=10)  # Reduced from 15
            all_posts_data.extend(new_posts)
            logger.info(f"🆕 New posts: {len(new_posts)}")

            rising_posts = self.scraper.scrape_posts(sort_type='rising', limit=8)  # Reduced from 10
            all_posts_data.extend(rising_posts)
            logger.info(f"📊 Rising posts: {len(rising_posts)}")

            # Remove duplicates and filter already processed posts
            unique_posts = {}
            new_posts_count = 0

            for post in all_posts_data:
                post_id = post['id']
                if post_id not in unique_posts and post_id not in self.processed_posts:
                    unique_posts[post_id] = post
                    new_posts_count += 1

            final_posts = list(unique_posts.values())
            logger.info(f"📝 New posts to process: {new_posts_count} (after deduplication)")

            # Phase 2: Optimized comment scraping
            ticker_rich_posts = [
                post for post in final_posts
                if len(post.get('tickers', [])) > 0 and post.get('num_comments', 0) > 5
            ]

            # Sort by engagement and limit to top posts
            ticker_rich_posts.sort(
                key=lambda x: (len(x.get('tickers', [])), x.get('num_comments', 0)),
                reverse=True
            )

            # Reduce comment scraping to save quota
            comments_target_posts = ticker_rich_posts[:5]  # Reduced from 8

            for post in comments_target_posts:
                try:
                    post_comments = self.scraper.scrape_comments(
                        post['id'],
                        limit=20  # Reduced from 30
                    )

                    # Filter out already processed comments
                    new_comments = [
                        comment for comment in post_comments
                        if comment['id'] not in self.processed_comments
                    ]

                    all_comments_data.extend(new_comments)
                    logger.info(f"💬 Post {post['id']}: {len(new_comments)} new comments")

                    # Brief pause
                    time.sleep(1)  # Reduced from 2

                except Exception as e:
                    logger.warning(f"Failed to scrape comments for {post['id']}: {e}")
                    continue

            # Phase 3: Optimized Sentiment Analysis
            sentiment_data = []
            ticker_sentiment_summary = {}

            if final_posts or all_comments_data:
                logger.info("🧠 Starting optimized sentiment analysis...")

                # Combine content for analysis
                all_content = final_posts + all_comments_data

                # Get unique tickers, but limit to most mentioned ones
                ticker_counts = {}
                for item in all_content:
                    for ticker in item.get('tickers', []):
                        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

                # Only analyze tickers with 2+ mentions to save LLM API calls
                significant_tickers = [
                    ticker for ticker, count in ticker_counts.items()
                    if count >= 2
                ][:15]  # Limit to top 15 tickers

                logger.info(f"🎯 Analyzing sentiment for {len(significant_tickers)} significant tickers")

                # Analyze sentiment for significant tickers only
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

            # Phase 4: Optimized Firebase saving
            if final_posts or all_comments_data:
                # Save posts and comments (uses optimized batch saving with deduplication)
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

                # Update processed posts/comments cache
                for post in final_posts:
                    self.processed_posts.add(post['id'])
                for comment in all_comments_data:
                    self.processed_comments.add(comment['id'])

                # Limit cache size to prevent memory issues
                if len(self.processed_posts) > 5000:
                    self.processed_posts = set(list(self.processed_posts)[-2500:])
                if len(self.processed_comments) > 10000:
                    self.processed_comments = set(list(self.processed_comments)[-5000:])

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
                    'cache_info': self.firebase_manager.get_cache_stats()
                }

                logger.info(f"✅ Optimized cycle complete: {len(final_posts)} posts, {len(all_comments_data)} comments")
                logger.info(f"🎯 Tickers found: {list(cycle_tickers)}")
                logger.info(f"📊 Sentiment: 🐂{len(bullish_tickers)} bullish, 🐻{len(bearish_tickers)} bearish")
                logger.info(f"⚡ Cache stats: {stats['cache_info']}")

                return stats

            else:
                logger.warning("No new posts to process in this cycle")
                return {'info': 'No new posts processed'}

        except Exception as e:
            logger.error(f"❌ Optimized scraping cycle failed: {e}")
            return {'error': str(e)}

    def cleanup_old_data(self):
        """Clean up old data with optimized deletion"""
        try:
            # Delete data older than 7 days with smaller batches
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
            'uptime_hours': round(uptime.total_seconds() / 3600, 2),
            'total_posts_scraped': self.total_posts_scraped,
            'total_comments_scraped': self.total_comments_scraped,
            'total_unique_tickers': len(self.total_tickers_found),
            'processed_posts_cache_size': len(self.processed_posts),
            'processed_comments_cache_size': len(self.processed_comments),
            'all_tickers_found': sorted(list(self.total_tickers_found)),
            'firebase_cache_info': self.firebase_manager.get_cache_stats(),
            'last_check': datetime.now(timezone.utc).isoformat()
        }

    def run_single_cycle(self):
        """Run a single optimized scraping cycle (for testing)"""
        logger.info("🧪 Running single optimized test cycle...")
        stats = self.scrape_cycle()
        print("\n📊 OPTIMIZED CYCLE RESULTS:")
        print(f"Posts: {stats.get('posts_scraped', 0)}")
        print(f"Comments: {stats.get('comments_scraped', 0)}")
        print(f"New posts: {stats.get('new_posts_processed', 0)}")
        print(f"Tickers: {stats.get('tickers_this_cycle', [])}")
        print(f"Time: {stats.get('cycle_time_seconds', 0)}s")
        print(f"Cache info: {stats.get('cache_info', {})}")
        return stats

    def start_continuous_scraping(self):
        """Start the optimized continuous scraping schedule"""
        logger.info("🔄 Starting optimized continuous scraping every 20 minutes...")
        logger.info("⚡ Using singleton Firebase manager and caching for efficiency")

        # Increase interval to 20 minutes to reduce API load
        schedule.every(20).minutes.do(self.scrape_cycle)

        # Schedule weekly cleanup
        schedule.every().sunday.at("02:00").do(self.cleanup_old_data)

        # Run initial cycle immediately
        logger.info("🚀 Running initial optimized scraping cycle...")
        self.scrape_cycle()

        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute for scheduled tasks

        except KeyboardInterrupt:
            logger.info("⏹️ Optimized continuous scraping stopped by user")
        except Exception as e:
            logger.error(f"❌ Optimized continuous scraping error: {e}")


def main():
    """Main function for testing or running"""
    import argparse

    parser = argparse.ArgumentParser(description='Optimized Reddit Continuous Scraper')
    parser.add_argument('--test', action='store_true', help='Run single cycle test')
    parser.add_argument('--continuous', action='store_true', help='Start continuous scraping')
    parser.add_argument('--status', action='store_true', help='Show current status')

    args = parser.parse_args()

    scraper = OptimizedContinuousRedditScraper()

    if args.test:
        scraper.run_single_cycle()
    elif args.continuous:
        scraper.start_continuous_scraping()
    elif args.status:
        status = scraper.get_status()
        print(f"📊 Optimized Status: {status}")
    else:
        # Default: run single test cycle
        print("🧪 Running optimized test cycle (use --continuous for continuous mode)")
        scraper.run_single_cycle()


if __name__ == "__main__":
    main()