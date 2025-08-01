"""
Continuous Reddit Scraper
Runs every 15 minutes to collect WSB posts and comments
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
from data import RedditScraper, FirebaseManager
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
    """Automated Reddit scraper that runs continuously"""

    def __init__(self):
        """Initialize continuous scraper"""
        self.scraper = RedditScraper()
        self.firebase_manager = FirebaseManager()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.total_posts_scraped = 0
        self.total_comments_scraped = 0
        self.total_tickers_found = set()
        self.start_time = datetime.now(timezone.utc)

        logger.info("🚀 Continuous Reddit Scraper initialized with sentiment analysis")

    def scrape_cycle(self) -> Dict:
        """Execute one complete scraping cycle"""
        cycle_start = time.time()
        logger.info("🔄 Starting scraping cycle...")

        try:
            # Phase 1: Scrape posts from different sources
            all_posts_data = []
            all_comments_data = []

            # Hot posts (trending discussions)
            hot_posts = self.scraper.scrape_posts(sort_type='hot', limit=20)
            all_posts_data.extend(hot_posts)
            logger.info(f"📈 Hot posts: {len(hot_posts)}")

            # New posts (catch early discussions)
            new_posts = self.scraper.scrape_posts(sort_type='new', limit=15)
            all_posts_data.extend(new_posts)
            logger.info(f"🆕 New posts: {len(new_posts)}")

            # Rising posts (gaining momentum)
            rising_posts = self.scraper.scrape_posts(sort_type='rising', limit=10)
            all_posts_data.extend(rising_posts)
            logger.info(f"📊 Rising posts: {len(rising_posts)}")

            # Remove duplicates based on post ID
            unique_posts = {post['id']: post for post in all_posts_data}
            final_posts = list(unique_posts.values())

            # Phase 2: Scrape comments from ticker-heavy posts
            ticker_rich_posts = [
                post for post in final_posts
                if len(post.get('tickers', [])) > 0 and post.get('num_comments', 0) > 5
            ]

            # Sort by comment count and engagement
            ticker_rich_posts.sort(
                key=lambda x: (len(x.get('tickers', [])), x.get('num_comments', 0)),
                reverse=True
            )

            # Scrape comments from top posts (limit to avoid rate limiting)
            comments_target_posts = ticker_rich_posts[:8]  # Top 8 ticker-rich posts

            for post in comments_target_posts:
                try:
                    post_comments = self.scraper.scrape_comments(
                        post['id'],
                        limit=30  # 30 comments per post
                    )
                    all_comments_data.extend(post_comments)
                    logger.info(f"💬 Post {post['id']}: {len(post_comments)} comments")

                    # Brief pause to be respectful
                    time.sleep(2)

                except Exception as e:
                    logger.warning(f"Failed to scrape comments for {post['id']}: {e}")
                    continue

            # Phase 3: Perform Sentiment Analysis
            sentiment_data = []
            ticker_sentiment_summary = {}

            if final_posts or all_comments_data:
                logger.info("🧠 Starting sentiment analysis...")

                # Combine posts and comments for analysis
                all_content = final_posts + all_comments_data

                # Get all unique tickers mentioned
                all_mentioned_tickers = set()
                for item in all_content:
                    all_mentioned_tickers.update(item.get('tickers', []))

                # Analyze sentiment for each ticker
                for ticker in all_mentioned_tickers:
                    try:
                        ticker_sentiment = self.sentiment_analyzer.analyze_ticker_sentiment(
                            all_content, ticker
                        )
                        ticker_sentiment_summary[ticker] = ticker_sentiment

                        # Save individual sentiment analysis
                        sentiment_data.append({
                            'ticker': ticker,
                            'sentiment': ticker_sentiment['sentiment'],
                            'confidence': ticker_sentiment['confidence'],
                            'numerical_score': ticker_sentiment['numerical_score'],
                            'mention_count': ticker_sentiment['mention_count'],
                            'sentiment_distribution': ticker_sentiment['sentiment_distribution'],
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'cycle_id': f"cycle_{int(time.time())}"
                        })

                        logger.info(f"💭 {ticker}: {ticker_sentiment['sentiment']} (confidence: {ticker_sentiment['confidence']:.2f})")

                    except Exception as e:
                        logger.warning(f"Sentiment analysis failed for {ticker}: {e}")
                        continue

            # Phase 4: Save all data to Firebase
            # Phase 4: Save all data to Firebase
            if final_posts:
                self.scraper.save_to_firebase(final_posts, all_comments_data)

                # Save sentiment analysis results
                if sentiment_data:
                    self.firebase_manager.save_sentiment_analysis(sentiment_data)
                    logger.info(f"💾 Saved sentiment analysis for {len(sentiment_data)} tickers")

                # Update statistics
                self.total_posts_scraped += len(final_posts)
                self.total_comments_scraped += len(all_comments_data)

                # Collect all tickers found
                cycle_tickers = set()
                for post in final_posts:
                    cycle_tickers.update(post.get('tickers', []))
                for comment in all_comments_data:
                    cycle_tickers.update(comment.get('tickers', []))

                self.total_tickers_found.update(cycle_tickers)

                # Cycle statistics
                cycle_time = time.time() - cycle_start

                # Calculate sentiment summary
                bullish_tickers = [t for t, s in ticker_sentiment_summary.items() if s['sentiment'] == 'bullish']
                bearish_tickers = [t for t, s in ticker_sentiment_summary.items() if s['sentiment'] == 'bearish']

                stats = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'posts_scraped': len(final_posts),
                    'comments_scraped': len(all_comments_data),
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
                    'total_unique_tickers': len(self.total_tickers_found)
                }

                logger.info(f"✅ Cycle complete: {len(final_posts)} posts, {len(all_comments_data)} comments")
                logger.info(f"🎯 Tickers found: {list(cycle_tickers)}")
                logger.info(f"📊 Sentiment: 🐂{len(bullish_tickers)} bullish, 🐻{len(bearish_tickers)} bearish")

                return stats

            else:
                logger.warning("No posts scraped in this cycle")
                return {'error': 'No posts scraped'}

        except Exception as e:
            logger.error(f"❌ Scraping cycle failed: {e}")
            return {'error': str(e)}

    def cleanup_old_data(self):
        """Clean up old data weekly"""
        try:
            # Delete data older than 7 days
            deleted_posts = self.firebase_manager.delete_old_data('reddit_posts', days=7)
            deleted_comments = self.firebase_manager.delete_old_data('reddit_posts_comments', days=7)

            logger.info(f"🗑️ Cleanup: {deleted_posts} old posts, {deleted_comments} old comments deleted")

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
            'all_tickers_found': sorted(list(self.total_tickers_found)),
            'last_check': datetime.now(timezone.utc).isoformat()
        }

    def run_single_cycle(self):
        """Run a single scraping cycle (for testing)"""
        logger.info("🧪 Running single test cycle...")
        stats = self.scrape_cycle()
        print("\n📊 CYCLE RESULTS:")
        print(f"Posts: {stats.get('posts_scraped', 0)}")
        print(f"Comments: {stats.get('comments_scraped', 0)}")
        print(f"Tickers: {stats.get('tickers_this_cycle', [])}")
        print(f"Time: {stats.get('cycle_time_seconds', 0)}s")
        return stats

    def start_continuous_scraping(self):
        """Start the continuous scraping schedule"""
        logger.info("🔄 Starting continuous scraping every 15 minutes...")

        # Schedule the main scraping cycle
        schedule.every(15).minutes.do(self.scrape_cycle)

        # Schedule weekly cleanup
        schedule.every().sunday.at("02:00").do(self.cleanup_old_data)

        # Run initial cycle immediately
        logger.info("🚀 Running initial scraping cycle...")
        self.scrape_cycle()

        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute for scheduled tasks

        except KeyboardInterrupt:
            logger.info("⏹️ Continuous scraping stopped by user")
        except Exception as e:
            logger.error(f"❌ Continuous scraping error: {e}")


def main():
    """Main function for testing or running"""
    import argparse

    parser = argparse.ArgumentParser(description='Reddit Continuous Scraper')
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