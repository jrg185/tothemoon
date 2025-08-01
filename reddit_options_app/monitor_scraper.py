"""
Monitor the continuous Reddit scraper
Real-time dashboard for tracking scraping progress
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import time
from datetime import datetime, timezone, timedelta
from data import FirebaseManager
import os


class ScraperMonitor:
    """Monitor continuous scraper progress and health"""

    def __init__(self):
        self.firebase_manager = FirebaseManager()

    def get_scraping_stats(self, hours: int = 24) -> dict:
        """Get comprehensive scraping statistics"""

        # Get recent posts and comments
        recent_posts = self.firebase_manager.get_recent_posts(limit=1000, hours=hours)

        # Calculate statistics
        if not recent_posts:
            return {
                'total_posts': 0,
                'posts_with_tickers': 0,
                'ticker_hit_rate': 0,
                'total_comments': 0,
                'unique_tickers': 0,
                'tickers': [],
                'posts_per_hour': 0,
                'avg_score': 0,
                'last_post_time': None,
                'hours_analyzed': hours
            }

        # Count comments (estimate from Firebase)
        total_comments = 0
        all_tickers = set()
        posts_with_tickers = 0
        total_score = 0

        for post in recent_posts:
            tickers = post.get('tickers', [])
            if tickers:
                posts_with_tickers += 1
                all_tickers.update(tickers)

            total_score += post.get('score', 0)

        # Calculate rates
        posts_per_hour = len(recent_posts) / hours if hours > 0 else 0

        # Latest post time
        latest_post_time = None
        if recent_posts:
            latest_timestamp = max(post.get('created_utc', 0) for post in recent_posts)
            latest_post_time = datetime.fromtimestamp(latest_timestamp, tz=timezone.utc)

        return {
            'total_posts': len(recent_posts),
            'posts_with_tickers': posts_with_tickers,
            'ticker_hit_rate': round(posts_with_tickers / len(recent_posts) * 100, 1) if recent_posts else 0,
            'unique_tickers': len(all_tickers),
            'tickers': sorted(list(all_tickers)),
            'posts_per_hour': round(posts_per_hour, 1),
            'avg_score': round(total_score / len(recent_posts), 1) if recent_posts else 0,
            'last_post_time': latest_post_time,
            'hours_analyzed': hours
        }

    def get_trending_analysis(self) -> dict:
        """Get trending ticker analysis"""

        # Get trending for different time periods
        trending_1h = self.firebase_manager.get_trending_tickers(hours=1, min_mentions=1)
        trending_6h = self.firebase_manager.get_trending_tickers(hours=6, min_mentions=2)
        trending_24h = self.firebase_manager.get_trending_tickers(hours=24, min_mentions=3)

        return {
            'trending_1h': trending_1h[:10],
            'trending_6h': trending_6h[:10],
            'trending_24h': trending_24h[:10]
        }

    def check_scraper_health(self) -> dict:
        """Check if continuous scraper is working properly"""

        # Check if we've received data recently
        recent_posts = self.firebase_manager.get_recent_posts(limit=5, hours=1)

        health_status = {
            'status': 'unknown',
            'last_data_time': None,
            'minutes_since_last_data': None,
            'recent_posts_count': len(recent_posts),
            'health_score': 0
        }

        if recent_posts:
            # Find most recent post
            latest_timestamp = max(post.get('created_utc', 0) for post in recent_posts)
            latest_time = datetime.fromtimestamp(latest_timestamp, tz=timezone.utc)

            minutes_ago = (datetime.now(timezone.utc) - latest_time).total_seconds() / 60

            health_status.update({
                'last_data_time': latest_time,
                'minutes_since_last_data': round(minutes_ago, 1)
            })

            # Determine health status
            if minutes_ago < 20:  # Within 20 minutes
                health_status['status'] = 'healthy'
                health_status['health_score'] = 100
            elif minutes_ago < 60:  # Within 1 hour
                health_status['status'] = 'warning'
                health_status['health_score'] = 70
            else:  # Over 1 hour
                health_status['status'] = 'unhealthy'
                health_status['health_score'] = 30
        else:
            health_status['status'] = 'no_data'
            health_status['health_score'] = 0

        return health_status

    def print_dashboard(self):
        """Print a real-time dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen

        print("🚀 REDDIT SCRAPER MONITOR")
        print("=" * 60)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Health check
        health = self.check_scraper_health()
        status_emoji = {
            'healthy': '🟢',
            'warning': '🟡',
            'unhealthy': '🔴',
            'no_data': '⚫',
            'unknown': '❓'
        }

        print(f"\n{status_emoji.get(health['status'], '❓')} SCRAPER STATUS: {health['status'].upper()}")
        if health['last_data_time']:
            print(f"   Last data: {health['minutes_since_last_data']:.1f} minutes ago")
        print(f"   Recent posts: {health['recent_posts_count']}")

        # 24h statistics
        stats_24h = self.get_scraping_stats(hours=24)
        print(f"\n📊 24-HOUR STATISTICS:")
        print(f"   Posts collected: {stats_24h['total_posts']}")
        print(f"   Posts with tickers: {stats_24h['posts_with_tickers']} ({stats_24h['ticker_hit_rate']}%)")
        print(f"   Unique tickers: {stats_24h['unique_tickers']}")
        print(f"   Collection rate: {stats_24h['posts_per_hour']} posts/hour")
        print(f"   Avg post score: {stats_24h['avg_score']}")

        # Recent trending
        trending = self.get_trending_analysis()
        print(f"\n🔥 TRENDING TICKERS:")

        if trending['trending_1h']:
            print("   Last 1 hour:")
            for ticker in trending['trending_1h'][:5]:
                print(f"     {ticker['ticker']}: {ticker['mention_count']} mentions")

        if trending['trending_24h']:
            print("   Last 24 hours:")
            for ticker in trending['trending_24h'][:5]:
                print(f"     {ticker['ticker']}: {ticker['mention_count']} mentions (avg score: {ticker['avg_score']})")

        if not trending['trending_1h'] and not trending['trending_24h']:
            print("     No trending tickers found")

        # All tickers found
        if stats_24h['tickers']:
            print(f"\n🎯 ALL TICKERS (24h): {', '.join(stats_24h['tickers'][:20])}")
            if len(stats_24h['tickers']) > 20:
                print(f"     ...and {len(stats_24h['tickers']) - 20} more")

        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop monitoring")

    def run_live_monitor(self, update_interval: int = 30):
        """Run live monitoring dashboard"""
        try:
            while True:
                self.print_dashboard()
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("\n👋 Monitor stopped")

    def run_single_report(self):
        """Generate a single comprehensive report"""
        print("📊 REDDIT SCRAPER COMPREHENSIVE REPORT")
        print("=" * 60)

        # Multi-timeframe analysis
        for hours in [1, 6, 24]:
            stats = self.get_scraping_stats(hours=hours)
            print(f"\n⏰ LAST {hours} HOUR{'S' if hours > 1 else ''}:")
            print(f"   Posts: {stats['total_posts']}")
            print(f"   With tickers: {stats['posts_with_tickers']} ({stats['ticker_hit_rate']}%)")
            print(f"   Unique tickers: {stats['unique_tickers']}")
            print(f"   Rate: {stats['posts_per_hour']:.1f} posts/hour")

            if stats['tickers']:
                print(f"   Tickers: {', '.join(stats['tickers'][:10])}")

        # Health check
        health = self.check_scraper_health()
        print(f"\n🏥 HEALTH CHECK:")
        print(f"   Status: {health['status']}")
        print(f"   Score: {health['health_score']}/100")
        if health['minutes_since_last_data']:
            print(f"   Last data: {health['minutes_since_last_data']:.1f} minutes ago")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Reddit Scraper Monitor')
    parser.add_argument('--live', action='store_true', help='Run live monitoring dashboard')
    parser.add_argument('--report', action='store_true', help='Generate single report')
    parser.add_argument('--interval', type=int, default=30, help='Update interval for live mode (seconds)')

    args = parser.parse_args()

    monitor = ScraperMonitor()

    if args.live:
        monitor.run_live_monitor(update_interval=args.interval)
    elif args.report:
        monitor.run_single_report()
    else:
        # Default: single report
        monitor.run_single_report()


if __name__ == "__main__":
    main()