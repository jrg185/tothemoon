"""
Basic tests to ensure project structure works
"""
import pytest
from config.settings import validate_config, APP_CONFIG

def test_config_validation():
    """Test that configuration validation works"""
    # This will pass if .env is set up correctly
    # You might need to mock this for CI/CD
    try:
        validate_config()
        assert True
    except ValueError as e:
        pytest.skip(f"Configuration not complete: {e}")

def test_app_config():
    """Test that app configuration loads"""
    assert 'debug' in APP_CONFIG
    assert 'log_level' in APP_CONFIG
    assert 'streamlit_port' in APP_CONFIG

def test_imports():
    """Test that all modules can be imported"""
    try:
        import data
        import processing
        import ml
        import options
        import ui
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")