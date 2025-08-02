"""
Optimized Firebase Manager for Reddit Options App
Uses realistic quota limits based on actual Firebase capacity
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime, timezone, timedelta
import json
import time

# Now import config (after fixing path)
from config.settings import FIREBASE_CONFIG, get_firebase_credentials_dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FirebaseManager:
    """Optimized Firebase manager with realistic quota limits"""

    _instance = None
    _cache = {}
    _cache_timestamps = {}
    CACHE_DURATION = 300  # 5 minutes cache (reduced from 1 hour)

    def __new__(cls):
        """Singleton pattern to reuse Firebase connection"""
        if cls._instance is None:
            cls._instance = super(FirebaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize Firebase connection (only once)"""
        if not self._initialized:
            self.db = None
            # REALISTIC QUOTA MANAGEMENT
            self._firebase_read_count = 0
            self._last_read_reset = time.time()
            self._max_reads_per_hour = 1000  # Realistic: 1000 reads per hour (24,000/day)
            self._max_reads_per_day = 35000   # Use 35k of your 40k daily limit
            self._daily_read_count = 0
            self._last_daily_reset = time.time()

            self._initialize_firebase()
            self._initialized = True

    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                # Get credentials from config
                cred_dict = get_firebase_credentials_dict()

                # Initialize Firebase
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized")
            else:
                logger.info("Firebase Admin SDK already initialized")

            # Get Firestore client
            self.db = firestore.client()
            logger.info("✅ Firebase connection ready")

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise

    def _reset_read_counters(self):
        """Reset read counters for hour and day"""
        current_time = time.time()

        # Reset hourly counter
        if current_time - self._last_read_reset > 3600:  # 1 hour
            self._firebase_read_count = 0
            self._last_read_reset = current_time

        # Reset daily counter
        if current_time - self._last_daily_reset > 86400:  # 24 hours
            self._daily_read_count = 0
            self._last_daily_reset = current_time

    def _can_make_read(self) -> bool:
        """Check if we can make a Firebase read without exceeding limits"""
        self._reset_read_counters()

        hourly_ok = self._firebase_read_count < self._max_reads_per_hour
        daily_ok = self._daily_read_count < self._max_reads_per_day

        if not hourly_ok:
            logger.warning(f"Firebase hourly limit reached ({self._firebase_read_count}/{self._max_reads_per_hour})")
        if not daily_ok:
            logger.warning(f"Firebase daily limit reached ({self._daily_read_count}/{self._max_reads_per_day})")

        return hourly_ok and daily_ok

    def _increment_read_count(self):
        """Increment read count"""
        self._firebase_read_count += 1
        self._daily_read_count += 1
        # Only log every 100 reads to reduce noise
        if self._firebase_read_count % 100 == 0:
            logger.info(f"📊 Firebase reads: {self._firebase_read_count}/hr, {self._daily_read_count}/day")

    def _get_cache_key(self, collection_name: str, filters: str, order_by: str, limit: str) -> str:
        """Generate cache key for query"""
        return f"{collection_name}_{filters}_{order_by}_{limit}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache_timestamps:
            return False

        cache_time = self._cache_timestamps[cache_key]
        return (time.time() - cache_time) < self.CACHE_DURATION

    def _cache_result(self, cache_key: str, result: Any):
        """Cache query result"""
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

    def save_document(self, collection_name: str, document_data: Dict, document_id: str = None) -> str:
        """Save a single document to Firestore"""
        try:
            collection_ref = self.db.collection(collection_name)

            if document_id:
                doc_ref = collection_ref.document(document_id)
                doc_ref.set(document_data, merge=True)
                return document_id
            else:
                # Auto-generate document ID
                _, doc_ref = collection_ref.add(document_data)
                return doc_ref.id

        except Exception as e:
            logger.error(f"Error saving document to {collection_name}: {e}")
            raise

    def batch_save(self, collection_name: str, documents: List[Dict], id_field: str = None) -> int:
        """Save multiple documents in batches with deduplication"""
        if not documents:
            return 0

        try:
            collection_ref = self.db.collection(collection_name)
            batch = self.db.batch()
            batch_size = 200  # Reasonable batch size
            saved_count = 0

            # Smart deduplication to reduce writes
            unique_docs = {}
            for doc_data in documents:
                doc_id = doc_data.get(id_field) if id_field else None
                if doc_id:
                    unique_docs[str(doc_id)] = doc_data
                else:
                    # For documents without ID, use a hash of content
                    content_hash = hash(str(sorted(doc_data.items())))
                    unique_docs[f"hash_{content_hash}"] = doc_data

            logger.info(f"Deduplication: {len(documents)} -> {len(unique_docs)} unique documents")

            for i, (doc_id, doc_data) in enumerate(unique_docs.items()):
                if id_field and id_field in doc_data:
                    doc_ref = collection_ref.document(str(doc_data[id_field]))
                elif doc_id.startswith("hash_"):
                    doc_ref = collection_ref.document()  # Auto-generate
                else:
                    doc_ref = collection_ref.document(doc_id)

                batch.set(doc_ref, doc_data, merge=True)

                # Commit batch when it reaches the limit
                if (i + 1) % batch_size == 0:
                    batch.commit()
                    saved_count += batch_size
                    batch = self.db.batch()  # Start new batch
                    logger.info(f"Saved batch of {batch_size} documents to {collection_name}")
                    time.sleep(0.2)  # Brief pause between batches

            # Commit remaining documents
            remaining = len(unique_docs) % batch_size
            if remaining > 0:
                batch.commit()
                saved_count += remaining
                logger.info(f"Saved final batch of {remaining} documents to {collection_name}")

            logger.info(f"Successfully saved {saved_count} documents to {collection_name}")
            return saved_count

        except Exception as e:
            logger.error(f"Error batch saving to {collection_name}: {e}")
            raise

    def get_document(self, collection_name: str, document_id: str) -> Optional[Dict]:
        """Get a single document by ID with caching"""
        cache_key = f"doc_{collection_name}_{document_id}"

        if self._is_cache_valid(cache_key):
            logger.debug(f"Using cached document: {document_id}")
            return self._cache[cache_key]

        # Check quota limits
        if not self._can_make_read():
            logger.warning(f"Firebase quota exceeded - returning cached data for {document_id}")
            return self._cache.get(cache_key, None)

        try:
            doc_ref = self.db.collection(collection_name).document(document_id)
            doc = doc_ref.get()

            self._increment_read_count()  # Count the read

            result = doc.to_dict() if doc.exists else None
            self._cache_result(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error getting document {document_id} from {collection_name}: {e}")
            return None

    def query_documents(self,
                       collection_name: str,
                       filters: List[tuple] = None,
                       order_by: str = None,
                       limit: int = None,
                       desc: bool = False,
                       use_cache: bool = True) -> List[Dict]:
        """Query documents with smart caching"""

        # Generate cache key
        filters_str = str(filters) if filters else "none"
        cache_key = self._get_cache_key(collection_name, filters_str, str(order_by), str(limit))

        # Try cache first
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"✅ Using cached result for {collection_name}")
            return self._cache[cache_key]

        # Check quota limits
        if not self._can_make_read():
            logger.warning(f"Firebase quota exceeded - returning cached result for {collection_name}")
            # Return expired cache if available, otherwise empty list
            if cache_key in self._cache:
                logger.info("Returning expired cache due to quota limits")
                return self._cache[cache_key]
            else:
                logger.info("No cache available and quota exceeded - returning empty list")
                return []

        try:
            query = self.db.collection(collection_name)

            # Apply filters
            if filters:
                for field, operator, value in filters:
                    query = query.where(filter=FieldFilter(field, operator, value))

            # Apply ordering
            if order_by:
                direction = firestore.Query.DESCENDING if desc else firestore.Query.ASCENDING
                query = query.order_by(order_by, direction=direction)

            # Apply reasonable limits (no need to be ultra-conservative)
            if limit:
                # Use reasonable limits instead of ultra-conservative ones
                max_limit = min(limit, 500)  # Allow up to 500 results
                query = query.limit(max_limit)
                if limit > max_limit:
                    logger.info(f"Query limit reduced from {limit} to {max_limit}")

            # Execute query and COUNT IT
            docs = query.stream()
            self._increment_read_count()  # Count this Firebase read

            results = []
            for doc in docs:
                doc_dict = doc.to_dict()
                doc_dict['_id'] = doc.id  # Include document ID
                results.append(doc_dict)

            # Cache the result
            if use_cache:
                self._cache_result(cache_key, results)

            return results

        except Exception as e:
            logger.error(f"Error querying {collection_name}: {e}")
            # Return cached data even if expired, or empty list
            return self._cache.get(cache_key, [])

    def get_recent_posts(self, limit: int = 100, hours: int = 24, use_cache: bool = True) -> List[Dict]:
        """Get recent Reddit posts with reasonable limits"""
        # Use reasonable limits instead of ultra-conservative
        optimized_limit = min(limit, 200)  # Allow up to 200 posts

        if limit > optimized_limit:
            logger.info(f"Recent posts limit adjusted from {limit} to {optimized_limit}")

        try:
            # Calculate timestamp cutoff
            cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)

            filters = [
                ('created_utc', '>=', cutoff_time)
            ]

            return self.query_documents(
                collection_name=FIREBASE_CONFIG['collections']['reddit_posts'],
                filters=filters,
                order_by='created_utc',
                limit=optimized_limit,
                desc=True,
                use_cache=use_cache
            )

        except Exception as e:
            logger.error(f"Error getting recent posts: {e}")
            return []

    def get_posts_by_ticker(self, ticker: str, limit: int = 50, use_cache: bool = True) -> List[Dict]:
        """Get posts mentioning a specific ticker"""
        try:
            filters = [
                ('tickers', 'array_contains', ticker.upper())
            ]

            return self.query_documents(
                collection_name=FIREBASE_CONFIG['collections']['reddit_posts'],
                filters=filters,
                order_by='created_utc',
                limit=min(limit, 100),  # Allow up to 100 posts per ticker
                desc=True,
                use_cache=use_cache
            )

        except Exception as e:
            logger.error(f"Error getting posts for ticker {ticker}: {e}")
            return []

    def get_trending_tickers(self, hours: int = 24, min_mentions: int = 5, use_cache: bool = True) -> List[Dict]:
        """Get trending tickers with reasonable caching"""
        cache_key = f"trending_{hours}h_{min_mentions}min"

        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"✅ Using cached trending data")
            return self._cache[cache_key]

        try:
            # Get reasonable amount of recent posts
            recent_posts = self.get_recent_posts(limit=300, hours=hours, use_cache=use_cache)

            # Count ticker mentions
            ticker_counts = {}
            ticker_scores = {}

            for post in recent_posts:
                tickers = post.get('tickers', [])
                post_score = post.get('score', 0)

                for ticker in tickers:
                    ticker = ticker.upper()
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
                    ticker_scores[ticker] = ticker_scores.get(ticker, 0) + post_score

            # Filter and format results
            trending = []
            for ticker, count in ticker_counts.items():
                if count >= min_mentions:
                    trending.append({
                        'ticker': ticker,
                        'mention_count': count,
                        'total_score': ticker_scores[ticker],
                        'avg_score': round(ticker_scores[ticker] / count, 2),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })

            # Sort by mention count
            trending.sort(key=lambda x: x['mention_count'], reverse=True)
            result = trending[:25]  # Return top 25 trending

            # Cache the result
            if use_cache:
                self._cache_result(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error getting trending tickers: {e}")
            return []

    def save_sentiment_analysis(self, analysis_data: List[Dict]) -> int:
        """Save sentiment analysis results with deduplication"""
        try:
            collection_name = FIREBASE_CONFIG['collections']['sentiment_data']
            return self.batch_save(collection_name, analysis_data, 'ticker')
        except Exception as e:
            logger.error(f"Error saving sentiment analysis: {e}")
            return 0

    def get_sentiment_overview(self, hours: int = 24, use_cache: bool = True) -> List[Dict]:
        """Get sentiment overview with reasonable caching"""
        cache_key = f"sentiment_overview_{hours}h"

        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"✅ Using cached sentiment overview")
            return self._cache[cache_key]

        # Check quota before making Firebase calls
        if not self._can_make_read():
            logger.warning(f"Firebase quota exceeded - returning cached sentiment")
            return self._cache.get(cache_key, [])

        try:
            # Calculate timestamp cutoff as ISO string
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            cutoff_iso = cutoff_time.isoformat()

            filters = [
                ('timestamp', '>=', cutoff_iso)
            ]

            sentiment_data = self.query_documents(
                collection_name=FIREBASE_CONFIG['collections']['sentiment_data'],
                filters=filters,
                order_by='timestamp',
                limit=200,  # Reasonable limit
                desc=True,
                use_cache=use_cache
            )

            # Get latest sentiment for each ticker
            latest_sentiments = {}
            for item in sentiment_data:
                ticker = item.get('ticker')
                if ticker and ticker not in latest_sentiments:
                    latest_sentiments[ticker] = item

            result = list(latest_sentiments.values())

            # Cache the result
            if use_cache:
                self._cache_result(cache_key, result)

            logger.info(f"Retrieved sentiment overview: {len(result)} tickers")
            return result

        except Exception as e:
            logger.error(f"Error getting sentiment overview: {e}")
            return []

    def clear_cache(self):
        """Clear all cached data"""
        cache_count = len(self._cache)
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info(f"🗑️ Cleared {cache_count} cached queries")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        current_time = time.time()
        valid_entries = sum(1 for timestamp in self._cache_timestamps.values()
                          if (current_time - timestamp) < self.CACHE_DURATION)

        return {
            'total_cached_queries': len(self._cache),
            'valid_cached_queries': valid_entries,
            'cache_hit_potential': f"{valid_entries}/{len(self._cache)}",
            'cache_duration_minutes': self.CACHE_DURATION / 60,
            'firebase_reads_this_hour': self._firebase_read_count,
            'firebase_reads_today': self._daily_read_count,
            'hourly_limit': self._max_reads_per_hour,
            'daily_limit': self._max_reads_per_day
        }

    def delete_old_data(self, collection_name: str, days: int = 30) -> int:
        """Delete old data from a collection"""
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)

            old_docs = self.query_documents(
                collection_name=collection_name,
                filters=[('created_utc', '<', cutoff_time)],
                limit=100,
                use_cache=False  # Don't cache deletion queries
            )

            if not old_docs:
                return 0

            # Delete in reasonable batches
            batch = self.db.batch()
            deleted_count = 0

            for doc in old_docs:
                doc_ref = self.db.collection(collection_name).document(doc['_id'])
                batch.delete(doc_ref)
                deleted_count += 1

                if deleted_count % 100 == 0:
                    batch.commit()
                    batch = self.db.batch()
                    time.sleep(0.5)

            # Commit remaining deletes
            if deleted_count % 100 != 0:
                batch.commit()

            logger.info(f"Deleted {deleted_count} old documents from {collection_name}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting old data from {collection_name}: {e}")
            return 0

    def get_quota_status(self) -> Dict:
        """Get current quota usage status"""
        self._reset_read_counters()

        return {
            'reads_this_hour': self._firebase_read_count,
            'hourly_limit': self._max_reads_per_hour,
            'hourly_remaining': max(0, self._max_reads_per_hour - self._firebase_read_count),
            'reads_today': self._daily_read_count,
            'daily_limit': self._max_reads_per_day,
            'daily_remaining': max(0, self._max_reads_per_day - self._daily_read_count),
            'quota_healthy': (self._firebase_read_count < self._max_reads_per_hour and
                            self._daily_read_count < self._max_reads_per_day)
        }


# Keep original name for compatibility
OptimizedFirebaseManager = FirebaseManager


def main():
    """Test realistic Firebase manager"""
    try:
        fm = FirebaseManager()

        # Show quota status
        quota_status = fm.get_quota_status()
        print(f"📊 Quota Status: {quota_status}")

        # Test reasonable queries
        print("\n🔍 Testing realistic queries...")

        # First call
        start_time = time.time()
        trending1 = fm.get_trending_tickers(hours=24)
        time1 = time.time() - start_time
        print(f"First call: {len(trending1)} tickers in {time1:.2f}s")

        # Second call - should use cache
        start_time = time.time()
        trending2 = fm.get_trending_tickers(hours=24)
        time2 = time.time() - start_time
        print(f"Second call: {len(trending2)} tickers in {time2:.2f}s (cached)")

        # Show final stats
        cache_stats = fm.get_cache_stats()
        print(f"📊 Cache stats: {cache_stats}")

        print("✅ Realistic Firebase manager working!")
        print(f"🎯 Using {fm._max_reads_per_day} of your 40,000 daily Firebase reads")

    except Exception as e:
        print(f"❌ Firebase manager test failed: {e}")


if __name__ == "__main__":
    main()