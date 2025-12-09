#!/usr/bin/env python3
"""
Quick verification script to check if all dependencies are installed.
"""
import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name} installed")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} NOT installed: {e}")
        return False

print("=" * 60)
print("Dependency Check")
print("=" * 60)
print()

all_ok = True

# Core dependencies
print("Core Dependencies:")
all_ok &= check_import("fastapi", "fastapi")
all_ok &= check_import("uvicorn", "uvicorn")
all_ok &= check_import("sqlalchemy", "sqlalchemy")
print()

# Database drivers
print("Database Drivers:")
all_ok &= check_import("psycopg2", "psycopg2-binary")
print()

# LLM providers
print("LLM Providers:")
check_import("openai", "openai (optional)")
check_import("google.generativeai", "google-generativeai (optional)")
print()

# Application modules
print("Application Modules:")
try:
    from database import engine, IS_POSTGRESQL, IS_SQLITE
    print(f"✓ database module loaded")
    print(f"  PostgreSQL: {IS_POSTGRESQL}, SQLite: {IS_SQLITE}")
    
    # Test connection
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print(f"✓ Database connection successful")
except Exception as e:
    print(f"✗ database module error: {e}")
    all_ok = False

try:
    from summary_pipeline import summary_queue, write_queue
    print(f"✓ summary_pipeline module loaded")
    print(f"  Summary queue maxsize: {summary_queue.maxsize}")
    print(f"  Write queue maxsize: {write_queue.maxsize}")
except Exception as e:
    print(f"✗ summary_pipeline module error: {e}")
    all_ok = False

print()
print("=" * 60)
if all_ok:
    print("✓ All critical dependencies are installed!")
    print("  You can start the backend server now.")
else:
    print("✗ Some dependencies are missing.")
    print("  Run: pip install -r requirements.txt")
print("=" * 60)

sys.exit(0 if all_ok else 1)

