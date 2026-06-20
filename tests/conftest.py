"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

# Ensure project root is in path for all tests
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
