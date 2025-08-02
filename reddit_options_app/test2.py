#!/usr/bin/env python3
"""
Simple debug script to check directory structure and imports
Save as reddit_options_app/debug_setup.py
"""

import os
import sys
from pathlib import Path

print("🔍 DIRECTORY STRUCTURE DEBUG")
print("=" * 50)

# Show current directory
current_dir = Path.cwd()
print(f"📁 Current directory: {current_dir}")

# Show directory contents
print(f"📂 Directory contents:")
for item in sorted(current_dir.iterdir()):
    print(f"   {'📁' if item.is_dir() else '📄'} {item.name}")

# Check for expected directories and files
expected_items = {
    'data': 'directory',
    'ml': 'directory',
    'processing': 'directory',
    'config': 'directory',
    'automated_analysis_engine.py': 'file',
    'continuous_scraper.py': 'file',
    'dashboard.py': 'file'
}

print(f"\n✅ EXPECTED ITEMS CHECK:")
for item_name, item_type in expected_items.items():
    item_path = current_dir / item_name
    if item_path.exists():
        actual_type = 'directory' if item_path.is_dir() else 'file'
        if actual_type == item_type:
            print(f"   ✅ {item_name} ({item_type})")
        else:
            print(f"   ⚠️  {item_name} (expected {item_type}, found {actual_type})")
    else:
        print(f"   ❌ {item_name} (missing {item_type})")

# Test basic imports
print(f"\n🔍 BASIC IMPORT TEST:")

# Add current directory to path
sys.path.insert(0, str(current_dir))

try:
    import data
    print("✅ data module imported")
    print(f"   📍 data location: {data.__file__}")
except Exception as e:
    print(f"❌ data module import failed: {e}")

try:
    import ml
    print("✅ ml module imported")
    print(f"   📍 ml location: {ml.__file__}")
except Exception as e:
    print(f"❌ ml module import failed: {e}")

try:
    from data import firebase_manager
    print("✅ firebase_manager imported")
except Exception as e:
    print(f"❌ firebase_manager import failed: {e}")

# Check if __init__.py files exist
print(f"\n📄 __init__.py FILES CHECK:")
init_files = [
    'data/__init__.py',
    'ml/__init__.py',
    'processing/__init__.py',
    'config/__init__.py'
]

for init_file in init_files:
    init_path = current_dir / init_file
    if init_path.exists():
        print(f"   ✅ {init_file}")
    else:
        print(f"   ❌ {init_file} (missing)")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   1. Make sure you're running this from reddit_options_app/ directory")
print(f"   2. Check that all __init__.py files exist")
print(f"   3. Check that directory structure matches expected layout")