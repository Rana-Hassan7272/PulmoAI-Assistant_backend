"""
Database Migration System

Handles database schema migrations for SQLite database.
"""
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Database path
DATABASE_PATH = "medical.db"


def get_db_connection():
    """Get SQLite database connection"""
    return sqlite3.connect(DATABASE_PATH)


def check_column_exists(conn, table_name, column_name):
    """Check if a column exists in a table"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def table_exists(conn, table_name):
    """Check if a table exists in the database"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (table_name,))
    return cursor.fetchone() is not None


def migrate_add_pdf_report_path():
    """Migration: Add pdf_report_path column to visits table"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if table exists first
        if not table_exists(conn, "visits"):
            logger.warning("visits table does not exist. Tables will be created by init_db().")
            conn.close()
            return True  # Return True because tables will be created with the column
        
        # Check if column already exists
        if check_column_exists(conn, "visits", "pdf_report_path"):
            logger.info("Column pdf_report_path already exists in visits table")
            conn.close()
            return True
        
        # Add the column
        cursor.execute("""
            ALTER TABLE visits 
            ADD COLUMN pdf_report_path TEXT
        """)
        
        conn.commit()
        conn.close()
        logger.info("Successfully added pdf_report_path column to visits table")
        return True
        
    except sqlite3.OperationalError as e:
        error_str = str(e).lower()
        if "duplicate column name" in error_str or "already exists" in error_str:
            logger.info("Column pdf_report_path already exists (duplicate column error)")
            return True
        if "no such table" in error_str:
            logger.warning("visits table does not exist yet. It will be created with the column by init_db().")
            return True
        logger.error(f"Error adding pdf_report_path column: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in migration: {e}")
        return False


def run_migrations():
    """Run all pending migrations"""
    logger.info("Running database migrations...")
    
    migrations = [
        ("add_pdf_report_path", migrate_add_pdf_report_path),
    ]
    
    for migration_name, migration_func in migrations:
        logger.info(f"Running migration: {migration_name}")
        try:
            success = migration_func()
            if success:
                logger.info(f"Migration {migration_name} completed successfully")
            else:
                logger.error(f"Migration {migration_name} failed")
        except Exception as e:
            logger.error(f"Error running migration {migration_name}: {e}")
    
    logger.info("Database migrations completed")


if __name__ == "__main__":
    # Run migrations directly
    run_migrations()

