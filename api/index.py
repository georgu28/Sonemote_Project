"""
Vercel serverless function wrapper for Sonemote Flask app.
"""
import sys
import os

# Add parent directory to path to import app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Change to parent directory so model file can be found
os.chdir(parent_dir)

# Import the Flask app
from app import app

# Export app for Vercel (Vercel's Python runtime handles WSGI conversion)
__all__ = ['app']

