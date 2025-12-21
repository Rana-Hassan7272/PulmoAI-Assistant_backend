"""
Manual migration script to add pdf_report_path column to visits table.

Run this script to fix the database schema:
    python run_migration.py
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from app.core.init_db import init_db
from app.core.migrations import run_migrations

if __name__ == "__main__":
    print("Initializing database (creating tables if needed)...")
    init_db()
    print("\nRunning database migrations...")
    run_migrations()
    print("\n✅ All migrations completed successfully!")

