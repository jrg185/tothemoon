"""
Main Streamlit application entry point
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.settings import APP_CONFIG, validate_config
from ui.dashboard import Dashboard


def main():
    """Main application function"""
    st.set_page_config(
        page_title="WSB Options Analyzer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.info("Please check your .env file and ensure all required variables are set.")
        return

    # Initialize dashboard
    dashboard = Dashboard()
    dashboard.render()


if __name__ == "__main__":
    main()