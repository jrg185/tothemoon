"""
Diagnostic Dashboard to identify loading issues
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import traceback
from typing import Dict, List, Optional

# Configure Streamlit
st.set_page_config(
    page_title="WSB Options Trader - Diagnostic",
    page_icon="🔧",
    layout="wide"
)

# Diagnostic styling
st.markdown("""
<style>
    .diagnostic-header {
        font-size: 2rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }

    .diagnostic-step {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #4CAF50;
    }

    .diagnostic-error {
        background-color: #2D1B00;
        border: 2px solid #FF4444;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }

    .diagnostic-success {
        background-color: #1B2D1B;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class DiagnosticDashboard:
    """Diagnostic dashboard to identify loading issues"""

    def __init__(self):
        self.firebase_manager = None
        self.diagnostic_results = []

    def log_step(self, step: str, status: str = "running", details: str = ""):
        """Log diagnostic step"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        result = {
            'timestamp': timestamp,
            'step': step,
            'status': status,
            'details': details
        }
        self.diagnostic_results.append(result)

        # Display in real-time
        if status == "success":
            st.markdown(f"""
            <div class="diagnostic-success">
            ✅ <strong>{timestamp}</strong> - {step}<br>
            {details}
            </div>
            """, unsafe_allow_html=True)
        elif status == "error":
            st.markdown(f"""
            <div class="diagnostic-error">
            ❌ <strong>{timestamp}</strong> - {step}<br>
            {details}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="diagnostic-step">
            ⏳ <strong>{timestamp}</strong> - {step}<br>
            {details}
            </div>
            """, unsafe_allow_html=True)

    def test_imports(self):
        """Test all necessary imports"""
        self.log_step("Testing imports", "running")

        try:
            # Test Firebase manager import
            from data.firebase_manager import OptimizedFirebaseManager
            self.log_step("Import OptimizedFirebaseManager", "success", "Firebase manager imported successfully")

            # Test other imports
            from config.settings import FIREBASE_CONFIG, FINANCIAL_APIS
            self.log_step("Import config", "success",
                          f"Firebase project: {FIREBASE_CONFIG.get('project_id', 'Not set')}")

            return True

        except Exception as e:
            error_details = f"Import error: {str(e)}\n{traceback.format_exc()}"
            self.log_step("Testing imports", "error", error_details)
            return False

    def test_firebase_connection(self):
        """Test Firebase connection"""
        self.log_step("Testing Firebase connection", "running")

        try:
            from data.firebase_manager import OptimizedFirebaseManager

            # Initialize Firebase manager
            start_time = time.time()
            self.firebase_manager = OptimizedFirebaseManager()
            init_time = time.time() - start_time

            self.log_step("Initialize Firebase", "success", f"Initialized in {init_time:.2f}s")

            # Test basic connection (without writing)
            if hasattr(self.firebase_manager, 'db') and self.firebase_manager.db:
                self.log_step("Firebase client", "success", "Firestore client available")
            else:
                self.log_step("Firebase client", "error", "Firestore client not available")
                return False

            return True

        except Exception as e:
            error_details = f"Firebase connection error: {str(e)}\n{traceback.format_exc()}"
            self.log_step("Testing Firebase connection", "error", error_details)
            return False

    def test_simple_query(self):
        """Test a simple Firebase query"""
        self.log_step("Testing simple Firebase query", "running")

        try:
            if not self.firebase_manager:
                self.log_step("Testing simple Firebase query", "error", "Firebase manager not initialized")
                return False

            # Try a very simple query with small limit
            start_time = time.time()

            # Use the optimized query method
            recent_posts = self.firebase_manager.query_documents(
                collection_name='reddit_posts',
                limit=5,
                order_by='created_utc',
                desc=True,
                use_cache=False  # Don't use cache for diagnostic
            )

            query_time = time.time() - start_time

            self.log_step("Simple query", "success",
                          f"Retrieved {len(recent_posts)} posts in {query_time:.2f}s")

            # Test cache stats
            cache_stats = self.firebase_manager.get_cache_stats()
            self.log_step("Cache stats", "success", f"Cache info: {cache_stats}")

            return True

        except Exception as e:
            error_details = f"Query error: {str(e)}\n{traceback.format_exc()}"
            self.log_step("Testing simple Firebase query", "error", error_details)
            return False

    def test_sentiment_data(self):
        """Test sentiment data retrieval"""
        self.log_step("Testing sentiment data", "running")

        try:
            if not self.firebase_manager:
                self.log_step("Testing sentiment data", "error", "Firebase manager not initialized")
                return False

            start_time = time.time()

            # Try to get sentiment overview with small limit
            sentiment_data = self.firebase_manager.query_documents(
                collection_name='sentiment_analysis',
                limit=3,
                order_by='timestamp',
                desc=True,
                use_cache=False
            )

            query_time = time.time() - start_time

            self.log_step("Sentiment query", "success",
                          f"Retrieved {len(sentiment_data)} sentiment records in {query_time:.2f}s")

            return True

        except Exception as e:
            error_details = f"Sentiment query error: {str(e)}\n{traceback.format_exc()}"
            self.log_step("Testing sentiment data", "error", error_details)
            return False

    def test_price_api(self):
        """Test price API (if available)"""
        self.log_step("Testing price API", "running")

        try:
            from config.settings import FINANCIAL_APIS
            finnhub_key = FINANCIAL_APIS.get('finnhub')

            if not finnhub_key:
                self.log_step("Price API", "success", "No Finnhub key configured (optional)")
                return True

            import requests

            # Simple test request
            url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={finnhub_key}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                self.log_step("Price API", "success", "Finnhub API responding")
            else:
                self.log_step("Price API", "error", f"API returned status {response.status_code}")

            return True

        except Exception as e:
            error_details = f"Price API error: {str(e)}"
            self.log_step("Testing price API", "error", error_details)
            return True  # Not critical

    def run_full_diagnostic(self):
        """Run complete diagnostic sequence"""
        st.markdown('<h1 class="diagnostic-header">🔧 WSB Dashboard Diagnostic</h1>', unsafe_allow_html=True)
        st.markdown("Diagnosing why the dashboard is stuck loading...")

        # Step 1: Test imports
        if not self.test_imports():
            st.error("❌ Import test failed - check your Python environment and file paths")
            return

        # Step 2: Test Firebase connection
        if not self.test_firebase_connection():
            st.error("❌ Firebase connection failed - check your credentials and quota")
            return

        # Step 3: Test simple query
        if not self.test_simple_query():
            st.error("❌ Firebase query failed - likely quota or permission issue")
            return

        # Step 4: Test sentiment data
        if not self.test_sentiment_data():
            st.warning("⚠️ Sentiment data query failed - may not have sentiment data yet")

        # Step 5: Test price API
        self.test_price_api()

        # Summary
        st.markdown("---")
        st.markdown("## 📊 Diagnostic Summary")

        success_count = sum(1 for r in self.diagnostic_results if r['status'] == 'success')
        error_count = sum(1 for r in self.diagnostic_results if r['status'] == 'error')

        if error_count == 0:
            st.success(f"✅ All tests passed! ({success_count} successful)")
            st.info("The main dashboard should work now. The loading issue may have been temporary.")
        elif error_count <= 2:
            st.warning(f"⚠️ Some issues found ({error_count} errors, {success_count} successful)")
            st.info("Main functionality should still work, but with limited features.")
        else:
            st.error(f"❌ Multiple issues found ({error_count} errors, {success_count} successful)")
            st.info("Need to fix configuration issues before dashboard will work.")

        # Recommendations
        st.markdown("## 💡 Recommendations")

        has_firebase_error = any(
            r['status'] == 'error' and 'firebase' in r['step'].lower() for r in self.diagnostic_results)
        has_quota_error = any(
            r['status'] == 'error' and 'quota' in r['details'].lower() for r in self.diagnostic_results)

        if has_quota_error:
            st.markdown("- **Firebase quota exceeded**: Wait 24 hours or upgrade your Firebase plan")
            st.markdown("- **Reduce data usage**: The optimizations should help, but you may need to wait")

        if has_firebase_error:
            st.markdown("- **Check Firebase credentials**: Verify your .env file has correct Firebase settings")
            st.markdown("- **Check Firebase project**: Ensure your project ID is correct and active")

        st.markdown("- **Try the main dashboard again**: Issues may be resolved after running diagnostics")


def main():
    """Run diagnostic dashboard"""
    diagnostic = DiagnosticDashboard()
    diagnostic.run_full_diagnostic()


if __name__ == "__main__":
    main()