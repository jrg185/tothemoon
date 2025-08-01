"""
Dashboard Launcher
Simple script to launch the WSB Sentiment Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'plotly', 'pandas']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False

    return True


def launch_dashboard():
    """Launch the Streamlit dashboard"""

    print("🚀 WSB Sentiment Dashboard Launcher")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        return

    # Check if dashboard file exists
    dashboard_file = Path("dashboard.py")
    if not dashboard_file.exists():
        print("❌ dashboard.py not found in current directory")
        print("💡 Make sure you're in the project root directory")
        return

    # Check data availability
    try:
        from data import FirebaseManager
        fm = FirebaseManager()
        recent_posts = fm.get_recent_posts(limit=5)

        if not recent_posts:
            print("⚠️  No data found in database")
            print("💡 Make sure your continuous scraper is running:")
            print("   python continuous_scraper.py --continuous")
            print("\n🔄 Launching dashboard anyway...")
        else:
            print(f"✅ Found {len(recent_posts)} recent posts")

    except Exception as e:
        print(f"⚠️  Could not check data: {e}")
        print("🔄 Launching dashboard anyway...")

    # Launch Streamlit
    print("\n🌐 Launching dashboard...")
    print("📱 Dashboard will open in your default browser")
    print("🛑 Press Ctrl+C to stop the dashboard\n")

    try:
        # Launch streamlit with the dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "dashboard.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")


if __name__ == "__main__":
    launch_dashboard()