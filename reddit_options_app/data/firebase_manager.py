"""
Firebase Manager for Reddit Options App
Handles all Firebase Firestore operations
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
from datetime import datetime, timezone
import json

# Now import config (after fixing path)
from config.settings import FIREBASE_CONFIG, get_firebase_credentials_dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FirebaseManager:
    """Manage Firebase Firestore operations"""

    def __init__(self):
        """Initialize Firebase connection"""
        self.db = None
        self._initialize_firebase()

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

            # Test connection
            self._test_connection()

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise

    def _test_connection(self):
        """Test Firebase connection"""
        try:
            # Try to read from a test collection
            test_ref = self.db.collection('_test_connection')
            test_doc = {
                'timestamp': datetime.now(timezone.utc),
                'message': 'Connection test successful'
            }

            # Write and read test document
            doc_ref = test_ref.document('test')
            doc_ref.set(test_doc)

            # Read it back
            doc = doc_ref.get()
            if doc.exists:
                logger.info("✅ Firebase connection test successful")
                # Clean up test document
                doc_ref.delete()
            else:
                raise Exception("Test document not found")

        except Exception as e:
            logger.error(f"Firebase connection test failed: {e}")
            raise

    def save_document(self, collection_name: str, document_data: Dict, document_id: str = None) -> str:
        """
        Save a single document to Firestore

        Args:
            collection_name: Name of the collection
            document_data: Data to save
            document_id: Optional document ID (auto-generated if not provided)

        Returns:
            Document ID of saved document
        """
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
        """
        Save multiple documents in batches

        Args:
            collection_name: Name of the collection
            documents: List of documents to save
            id_field: Field to use as document ID (if None, auto-generate)

        Returns:
            Number of documents saved
        """
        if not documents:
            return 0

        try:
            collection_ref = self.db.collection(collection_name)
            batch = self.db.batch()
            batch_size = 500  # Firestore batch limit
            saved_count = 0

            for i, doc_data in enumerate(documents):
                # Use specified field as document ID or auto-generate
                if id_field and id_field in doc_data:
                    doc_id = str(doc_data[id_field])
                    doc_ref = collection_ref.document(doc_id)
                else:
                    doc_ref = collection_ref.document()  # Auto-generate ID

                batch.set(doc_ref, doc_data, merge=True)

                # Commit batch when it reaches the limit
                if (i + 1) % batch_size == 0:
                    batch.commit()
                    saved_count += batch_size
                    batch = self.db.batch()  # Start new batch
                    logger.info(f"Saved batch of {batch_size} documents to {collection_name}")

            # Commit remaining documents
            remaining = len(documents) % batch_size
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
        """
        Get a single document by ID

        Args:
            collection_name: Name of the collection
            document_id: Document ID

        Returns:
            Document data or None if not found
        """
        try:
            doc_ref = self.db.collection(collection_name).document(document_id)
            doc = doc_ref.get()

            if doc.exists:
                return doc.to_dict()
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting document {document_id} from {collection_name}: {e}")
            return None

    def query_documents(self,
                       collection_name: str,
                       filters: List[tuple] = None,
                       order_by: str = None,
                       limit: int = None,
                       desc: bool = False) -> List[Dict]:
        """
        Query documents with filters

        Args:
            collection_name: Name of the collection
            filters: List of (field, operator, value) tuples
            order_by: Field to order by
            limit: Maximum number of results
            desc: Whether to order in descending order

        Returns:
            List of document dictionaries
        """
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

            # Apply limit
            if limit:
                query = query.limit(limit)

            # Execute query
            docs = query.stream()
            results = []

            for doc in docs:
                doc_dict = doc.to_dict()
                doc_dict['_id'] = doc.id  # Include document ID
                results.append(doc_dict)

            return results

        except Exception as e:
            logger.error(f"Error querying {collection_name}: {e}")
            return []

    def get_recent_posts(self, limit: int = 100, hours: int = 24) -> List[Dict]:
        """
        Get recent Reddit posts

        Args:
            limit: Maximum number of posts
            hours: How many hours back to look

        Returns:
            List of recent posts
        """
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
                limit=limit,
                desc=True
            )

        except Exception as e:
            logger.error(f"Error getting recent posts: {e}")
            return []

    def get_posts_by_ticker(self, ticker: str, limit: int = 50) -> List[Dict]:
        """
        Get posts mentioning a specific ticker

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of posts

        Returns:
            List of posts mentioning the ticker
        """
        try:
            filters = [
                ('tickers', 'array_contains', ticker.upper())
            ]

            return self.query_documents(
                collection_name=FIREBASE_CONFIG['collections']['reddit_posts'],
                filters=filters,
                order_by='created_utc',
                limit=limit,
                desc=True
            )

        except Exception as e:
            logger.error(f"Error getting posts for ticker {ticker}: {e}")
            return []

    def get_trending_tickers(self, hours: int = 24, min_mentions: int = 5) -> List[Dict]:
        """
        Get trending tickers based on mention frequency

        Args:
            hours: Time window to analyze
            min_mentions: Minimum mentions to be considered trending

        Returns:
            List of trending ticker data
        """
        try:
            # Get recent posts
            recent_posts = self.get_recent_posts(limit=1000, hours=hours)

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

            return trending[:20]  # Top 20 trending

        except Exception as e:
            logger.error(f"Error getting trending tickers: {e}")
            return []

    def save_sentiment_analysis(self, analysis_data: List[Dict]) -> int:
        """Save sentiment analysis results"""
        try:
            collection_name = FIREBASE_CONFIG['collections']['sentiment_data']
            return self.batch_save(collection_name, analysis_data)
        except Exception as e:
            logger.error(f"Error saving sentiment analysis: {e}")
            return 0

    def get_ticker_sentiment(self, ticker: str, hours: int = 24) -> Dict:
        """
        Get sentiment analysis for a specific ticker

        Args:
            ticker: Stock ticker symbol
            hours: How many hours back to look

        Returns:
            Latest sentiment data for the ticker
        """
        try:
            # Calculate timestamp cutoff
            cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)

            filters = [
                ('ticker', '==', ticker.upper()),
                ('timestamp', '>=', cutoff_time)
            ]

            sentiment_data = self.query_documents(
                collection_name=FIREBASE_CONFIG['collections']['sentiment_data'],
                filters=filters,
                order_by='timestamp',
                limit=1,
                desc=True
            )

            return sentiment_data[0] if sentiment_data else None

        except Exception as e:
            logger.error(f"Error getting sentiment for ticker {ticker}: {e}")
            return None

    # Replace the get_sentiment_overview method in your data/firebase_manager.py

    def get_sentiment_overview(self, hours: int = 24) -> List[Dict]:
        """
        Get sentiment overview for all tickers

        Args:
            hours: Time window to analyze

        Returns:
            List of ticker sentiment data
        """
        try:
            # Calculate timestamp cutoff as ISO string (to match your data format)
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            cutoff_iso = cutoff_time.isoformat()

            filters = [
                ('timestamp', '>=', cutoff_iso)
            ]

            sentiment_data = self.query_documents(
                collection_name=FIREBASE_CONFIG['collections']['sentiment_data'],
                filters=filters,
                order_by='timestamp',
                desc=True
            )

            # Get latest sentiment for each ticker
            latest_sentiments = {}
            for item in sentiment_data:
                ticker = item.get('ticker')
                if ticker and ticker not in latest_sentiments:
                    latest_sentiments[ticker] = item

            logger.info(f"Retrieved sentiment overview: {len(latest_sentiments)} tickers")
            return list(latest_sentiments.values())

        except Exception as e:
            logger.error(f"Error getting sentiment overview: {e}")

            # Fallback: Get all recent sentiment data without timestamp filtering
            try:
                logger.info("Attempting fallback: getting all sentiment data")
                all_sentiment_data = self.query_documents(
                    collection_name=FIREBASE_CONFIG['collections']['sentiment_data'],
                    order_by='timestamp',
                    limit=100,
                    desc=True
                )

                # Get latest sentiment for each ticker
                latest_sentiments = {}
                for item in all_sentiment_data:
                    ticker = item.get('ticker')
                    if ticker and ticker not in latest_sentiments:
                        latest_sentiments[ticker] = item

                logger.info(f"Fallback retrieved: {len(latest_sentiments)} tickers")
                return list(latest_sentiments.values())

            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return []

    def get_sentiment_trends(self, ticker: str, hours: int = 72) -> List[Dict]:
        """
        Get sentiment trends for a ticker over time

        Args:
            ticker: Stock ticker symbol
            hours: Time window to analyze

        Returns:
            List of sentiment data points over time
        """
        try:
            # Calculate timestamp cutoff
            cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)

            filters = [
                ('ticker', '==', ticker.upper()),
                ('timestamp', '>=', cutoff_time)
            ]

            return self.query_documents(
                collection_name=FIREBASE_CONFIG['collections']['sentiment_data'],
                filters=filters,
                order_by='timestamp',
                desc=False  # Chronological order for trends
            )

        except Exception as e:
            logger.error(f"Error getting sentiment trends for {ticker}: {e}")
            return []

    def save_predictions(self, predictions_data: List[Dict]) -> int:
        """Save ML predictions"""
        try:
            collection_name = FIREBASE_CONFIG['collections']['predictions']
            return self.batch_save(collection_name, predictions_data)
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            return 0

    def save_options_strategies(self, strategies_data: List[Dict]) -> int:
        """Save options strategies"""
        try:
            collection_name = FIREBASE_CONFIG['collections']['options_strategies']
            return self.batch_save(collection_name, strategies_data)
        except Exception as e:
            logger.error(f"Error saving options strategies: {e}")
            return 0

    def delete_old_data(self, collection_name: str, days: int = 30) -> int:
        """
        Delete old data from a collection

        Args:
            collection_name: Collection to clean up
            days: Delete data older than this many days

        Returns:
            Number of documents deleted
        """
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)

            # Query old documents
            old_docs = self.query_documents(
                collection_name=collection_name,
                filters=[('created_utc', '<', cutoff_time)],
                limit=1000  # Delete in batches
            )

            if not old_docs:
                return 0

            # Delete in batches
            batch = self.db.batch()
            deleted_count = 0

            for doc in old_docs:
                doc_ref = self.db.collection(collection_name).document(doc['_id'])
                batch.delete(doc_ref)
                deleted_count += 1

                if deleted_count % 500 == 0:  # Batch limit
                    batch.commit()
                    batch = self.db.batch()

            # Commit remaining deletes
            if deleted_count % 500 != 0:
                batch.commit()

            logger.info(f"Deleted {deleted_count} old documents from {collection_name}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting old data from {collection_name}: {e}")
            return 0


def main():
    """Test Firebase manager"""
    try:
        fm = FirebaseManager()

        # Test document operations
        test_data = {
            'message': 'Hello Firebase!',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'test': True
        }

        # Save test document
        doc_id = fm.save_document('test_collection', test_data, 'test_doc')
        print(f"Saved test document with ID: {doc_id}")

        # Read test document
        retrieved = fm.get_document('test_collection', 'test_doc')
        print(f"Retrieved document: {retrieved}")

        # Test queries
        trending = fm.get_trending_tickers(hours=24)
        print(f"Trending tickers: {trending}")

        print("✅ Firebase manager test completed successfully!")

    except Exception as e:
        print(f"❌ Firebase manager test failed: {e}")


if __name__ == "__main__":
    main()