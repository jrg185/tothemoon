"""
Reddit Scraper for wallstreetbets
Collects posts, comments, and metadata from r/wallstreetbets
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import praw
import prawcore
import time
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

# Now import config and other modules (after fixing path)
from config.settings import REDDIT_CONFIG, FIREBASE_CONFIG
from data.firebase_manager import FirebaseManager
from data.llm_ticker_extractor import EnhancedTickerExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Data class for Reddit post information"""
    id: str
    title: str
    selftext: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    permalink: str
    url: str
    subreddit: str
    is_self: bool
    over_18: bool
    spoiler: bool
    stickied: bool
    gilded: int
    total_awards_received: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for Firebase storage"""
        return {
            'id': self.id,
            'title': self.title,
            'selftext': self.selftext,
            'author': self.author if self.author != '[deleted]' else 'deleted_user',
            'score': self.score,
            'upvote_ratio': self.upvote_ratio,
            'num_comments': self.num_comments,
            'created_utc': self.created_utc,
            'created_datetime': datetime.fromtimestamp(self.created_utc, tz=timezone.utc).isoformat(),
            'permalink': self.permalink,
            'url': self.url,
            'subreddit': self.subreddit,
            'is_self': self.is_self,
            'over_18': self.over_18,
            'spoiler': self.spoiler,
            'stickied': self.stickied,
            'gilded': self.gilded,
            'total_awards_received': self.total_awards_received,
            'scraped_at': datetime.now(timezone.utc).isoformat(),
            'text_for_analysis': f"{self.title} {self.selftext}".strip()
        }


@dataclass
class RedditComment:
    """Data class for Reddit comment information"""
    id: str
    body: str
    author: str
    score: int
    created_utc: float
    parent_id: str
    post_id: str
    is_submitter: bool
    gilded: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for Firebase storage"""
        return {
            'id': self.id,
            'body': self.body,
            'author': self.author if self.author != '[deleted]' else 'deleted_user',
            'score': self.score,
            'created_utc': self.created_utc,
            'created_datetime': datetime.fromtimestamp(self.created_utc, tz=timezone.utc).isoformat(),
            'parent_id': self.parent_id,
            'post_id': self.post_id,
            'is_submitter': self.is_submitter,
            'gilded': self.gilded,
            'scraped_at': datetime.now(timezone.utc).isoformat()
        }


class RedditScraper:
    """Main Reddit scraper class"""

    def __init__(self):
        """Initialize Reddit scraper with PRAW"""
        self.reddit = None
        self.firebase_manager = None
        self.ticker_extractor = EnhancedTickerExtractor(use_llm=True)
        self._initialize_reddit()
        self._initialize_firebase()

    def _initialize_reddit(self):
        """Initialize PRAW Reddit instance"""
        try:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CONFIG['client_id'],
                client_secret=REDDIT_CONFIG['client_secret'],
                user_agent=REDDIT_CONFIG['user_agent'],
                ratelimit_seconds=600  # Wait 10 minutes on rate limit
            )

            # Test connection
            logger.info(f"Connected to Reddit as: {self.reddit.user.me() or 'Anonymous'}")
            logger.info(f"Rate limit remaining: {self.reddit.auth.limits}")

        except Exception as e:
            logger.error(f"Failed to initialize Reddit connection: {e}")
            raise

    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            self.firebase_manager = FirebaseManager()
            logger.info("Firebase connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise

    def _extract_post_data(self, submission) -> RedditPost:
        """Extract data from a Reddit submission"""
        try:
            return RedditPost(
                id=submission.id,
                title=submission.title,
                selftext=submission.selftext,
                author=str(submission.author) if submission.author else '[deleted]',
                score=submission.score,
                upvote_ratio=submission.upvote_ratio,
                num_comments=submission.num_comments,
                created_utc=submission.created_utc,
                permalink=submission.permalink,
                url=submission.url,
                subreddit=str(submission.subreddit),
                is_self=submission.is_self,
                over_18=submission.over_18,
                spoiler=submission.spoiler,
                stickied=submission.stickied,
                gilded=submission.gilded,
                total_awards_received=submission.total_awards_received
            )
        except Exception as e:
            logger.error(f"Error extracting post data: {e}")
            return None

    def _extract_comment_data(self, comment, post_id: str) -> Optional[RedditComment]:
        """Extract data from a Reddit comment"""
        try:
            # Skip deleted/removed comments
            if not hasattr(comment, 'body') or comment.body in ['[deleted]', '[removed]']:
                return None

            return RedditComment(
                id=comment.id,
                body=comment.body,
                author=str(comment.author) if comment.author else '[deleted]',
                score=comment.score,
                created_utc=comment.created_utc,
                parent_id=comment.parent_id,
                post_id=post_id,
                is_submitter=comment.is_submitter,
                gilded=comment.gilded
            )
        except Exception as e:
            logger.error(f"Error extracting comment data: {e}")
            return None

    def scrape_posts(self,
                    sort_type: str = 'hot',
                    limit: int = 100,
                    time_filter: str = 'day') -> List[Dict]:
        """
        Scrape posts from r/wallstreetbets

        Args:
            sort_type: 'hot', 'new', 'top', 'rising'
            limit: Number of posts to scrape
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'

        Returns:
            List of post dictionaries
        """
        posts_data = []

        try:
            subreddit = self.reddit.subreddit(REDDIT_CONFIG['subreddit'])
            logger.info(f"Scraping {limit} {sort_type} posts from r/{REDDIT_CONFIG['subreddit']}")

            # Get posts based on sort type
            if sort_type == 'hot':
                submissions = subreddit.hot(limit=limit)
            elif sort_type == 'new':
                submissions = subreddit.new(limit=limit)
            elif sort_type == 'top':
                submissions = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort_type == 'rising':
                submissions = subreddit.rising(limit=limit)
            else:
                raise ValueError(f"Invalid sort_type: {sort_type}")

            for submission in submissions:
                try:
                    post_data = self._extract_post_data(submission)
                    if post_data:
                        post_dict = post_data.to_dict()

                        # Extract tickers from title and text
                        text_to_analyze = f"{post_data.title} {post_data.selftext}"
                        tickers = self.ticker_extractor.extract_tickers(text_to_analyze)
                        post_dict['tickers'] = tickers
                        post_dict['ticker_count'] = len(tickers)

                        posts_data.append(post_dict)
                        logger.info(f"Scraped post: {post_data.id} - {post_data.title[:50]}...")

                        # Be respectful to Reddit's servers
                        time.sleep(0.1)

                except prawcore.exceptions.TooManyRequests:
                    logger.warning("Rate limited by Reddit. Waiting...")
                    time.sleep(60)
                    continue
                except Exception as e:
                    logger.error(f"Error processing submission: {e}")
                    continue

            logger.info(f"Successfully scraped {len(posts_data)} posts")
            return posts_data

        except Exception as e:
            logger.error(f"Error scraping posts: {e}")
            return posts_data

    def scrape_comments(self, post_id: str, limit: int = 50) -> List[Dict]:
        """
        Scrape comments from a specific post with better filtering

        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments to scrape

        Returns:
            List of comment dictionaries
        """
        comments_data = []

        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=3)  # Expand some "more comments"

            comment_count = 0
            all_comments = submission.comments.list()

            # Sort comments by score to get most relevant ones first
            all_comments.sort(key=lambda c: getattr(c, 'score', 0), reverse=True)

            for comment in all_comments:
                if comment_count >= limit:
                    break

                comment_data = self._extract_comment_data(comment, post_id)
                if comment_data and comment_data.body and len(comment_data.body.strip()) > 10:
                    comment_dict = comment_data.to_dict()

                    # Extract tickers from comment
                    tickers = self.ticker_extractor.extract_tickers(comment_data.body)
                    comment_dict['tickers'] = tickers
                    comment_dict['ticker_count'] = len(tickers)

                    # Only include comments with decent content or tickers
                    if len(comment_data.body) > 20 or tickers:
                        comments_data.append(comment_dict)
                        comment_count += 1

                        # Brief pause to be respectful
                        time.sleep(0.1)

            logger.info(f"Scraped {len(comments_data)} quality comments from post {post_id}")
            return comments_data

        except Exception as e:
            logger.error(f"Error scraping comments for post {post_id}: {e}")
            return comments_data

    def save_to_firebase(self, posts_data: List[Dict], comments_data: List[Dict] = None):
        """Save scraped data to Firebase"""
        try:
            # Save posts
            if posts_data:
                collection_name = FIREBASE_CONFIG['collections']['reddit_posts']
                saved_posts = self.firebase_manager.batch_save(collection_name, posts_data, 'id')
                logger.info(f"Saved {saved_posts} posts to Firebase")

            # Save comments
            if comments_data:
                collection_name = f"{FIREBASE_CONFIG['collections']['reddit_posts']}_comments"
                saved_comments = self.firebase_manager.batch_save(collection_name, comments_data, 'id')
                logger.info(f"Saved {saved_comments} comments to Firebase")

        except Exception as e:
            logger.error(f"Error saving to Firebase: {e}")

    def scrape_and_save(self,
                       sort_type: str = 'hot',
                       posts_limit: int = 100,
                       include_comments: bool = False,
                       comments_limit: int = 50) -> Dict:
        """
        Complete scraping workflow: scrape posts (and optionally comments) and save to Firebase

        Returns:
            Dictionary with scraping statistics
        """
        start_time = time.time()

        # Scrape posts
        posts_data = self.scrape_posts(sort_type=sort_type, limit=posts_limit)

        comments_data = []
        if include_comments and posts_data:
            # Scrape comments for posts with tickers
            posts_with_tickers = [post for post in posts_data if post.get('ticker_count', 0) > 0][:10]  # Limit to top 10

            for post in posts_with_tickers:
                post_comments = self.scrape_comments(post['id'], limit=comments_limit)
                comments_data.extend(post_comments)
                time.sleep(1)  # Be respectful

        # Save to Firebase
        self.save_to_firebase(posts_data, comments_data if comments_data else None)

        # Calculate statistics
        total_tickers = set()
        for post in posts_data:
            total_tickers.update(post.get('tickers', []))

        stats = {
            'posts_scraped': len(posts_data),
            'comments_scraped': len(comments_data),
            'unique_tickers_found': len(total_tickers),
            'tickers': list(total_tickers),
            'execution_time_seconds': round(time.time() - start_time, 2),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"Scraping complete: {stats}")
        return stats


def test_ticker_extraction():
    """Test the LLM-powered ticker extraction"""
    extractor = EnhancedTickerExtractor(use_llm=True)

    test_cases = [
        # Should work well - clear tickers
        "I'm buying $TSLA calls and $GME puts tomorrow",
        "AAPL earnings looking good, might grab NVDA too",
        "Loading up on $RBLX and HOOD",

        # Should filter out common words
        "THIS IS THE BEST stock for YOU to BUY",
        "I THINK IT WILL GO UP TOMORROW",
        "DOES ANYONE KNOW WHAT TO DO HERE",

        # Context-based detection should work
        "Buying PLTR calls, SOFI puts looking good",
        "GME 420c 12/17, SPY 400p expiry tomorrow",

        # Mixed - should keep only legitimate tickers
        "APPLE $AAPL JUST REPORTED EARNINGS BEAT",
        "Reddit Q2 crushes estimates, RDDT strong outlook",

        # Should handle options syntax
        "TSLA 1000c, SPY hitting 400 resistance"
    ]

    print("🤖 Testing LLM-Powered Ticker Extraction")
    print("=" * 60)

    for i, text in enumerate(test_cases, 1):
        try:
            tickers = extractor.extract_tickers(text)
            print(f"{i}. \"{text}\"")
            print(f"   LLM Result: {tickers}")
            print()
        except Exception as e:
            print(f"{i}. \"{text}\"")
            print(f"   Error: {e}")
            print()

    return extractor


def main():
    """Test the Reddit scraper"""
    # First test ticker extraction
    print("Testing ticker extraction...")
    test_ticker_extraction()

    print("\n" + "="*50)
    print("Testing Reddit scraper...")

    try:
        scraper = RedditScraper()

        # Test scraping
        stats = scraper.scrape_and_save(
            sort_type='hot',
            posts_limit=5,  # Small test
            include_comments=False
        )

        print(f"Scraping results: {stats}")

    except Exception as e:
        print(f"Scraper test failed (expected if no API keys): {e}")
        print("Set up your .env file with Reddit and Firebase credentials to test scraping")


if __name__ == "__main__":
    main()