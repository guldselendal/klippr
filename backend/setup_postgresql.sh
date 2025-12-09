#!/bin/bash
# Quick setup script for PostgreSQL migration

set -e

echo "=========================================="
echo "PostgreSQL Migration Setup"
echo "=========================================="
echo ""

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL not found. Please install PostgreSQL first:"
    echo "   macOS: brew install postgresql@15"
    echo "   Linux: sudo apt-get install postgresql postgresql-contrib"
    exit 1
fi

echo "✓ PostgreSQL found"

# Check if psycopg2 is installed
if ! python3 -c "import psycopg2" 2>/dev/null; then
    echo ""
    echo "Installing PostgreSQL driver (psycopg2-binary)..."
    pip install psycopg2-binary
    echo "✓ psycopg2-binary installed"
else
    echo "✓ psycopg2-binary already installed"
fi

# Check if PostgreSQL is running
if ! pg_isready -q; then
    echo ""
    echo "⚠️  PostgreSQL is not running. Starting PostgreSQL..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start postgresql@15 2>/dev/null || brew services start postgresql 2>/dev/null || true
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start postgresql 2>/dev/null || true
    fi
    sleep 2
fi

if pg_isready -q; then
    echo "✓ PostgreSQL is running"
else
    echo "❌ PostgreSQL is not running. Please start it manually:"
    echo "   macOS: brew services start postgresql@15"
    echo "   Linux: sudo systemctl start postgresql"
    exit 1
fi

# Create database and user if they don't exist
echo ""
echo "Setting up database and user..."

# Try to create database (will fail if exists, which is fine)
createdb klippr 2>/dev/null || echo "  Database 'klippr' already exists"

# Try to create user (will fail if exists, which is fine)
createuser klippr 2>/dev/null || echo "  User 'klippr' already exists"

# Set password and permissions
psql -d postgres -c "ALTER USER klippr WITH PASSWORD 'klippr';" 2>/dev/null || true
psql -d klippr -c "GRANT ALL PRIVILEGES ON DATABASE klippr TO klippr;" 2>/dev/null || true
psql -d klippr -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO klippr;" 2>/dev/null || true

echo "✓ Database and user configured"

# Check if .env exists
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file with PostgreSQL configuration..."
    echo "DATABASE_URL=postgresql://klippr:klippr@localhost:5432/klippr" > .env
    echo "✓ .env file created"
else
    echo ""
    echo "⚠️  .env file already exists. Please ensure it contains:"
    echo "   DATABASE_URL=postgresql://klippr:klippr@localhost:5432/klippr"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test connection: python3 -c 'from database import engine; print(\"✓ Connected\")'"
echo "2. Migrate data: python3 migrate_to_postgresql.py"
echo "3. Start backend: python3 -m uvicorn main:app --reload"
echo ""

