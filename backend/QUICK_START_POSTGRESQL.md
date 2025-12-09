# Quick Start: PostgreSQL Migration

## 3-Step Setup

### Step 1: Install Dependencies

```bash
cd backend

# Install PostgreSQL driver
pip install psycopg2-binary

# Or install all requirements
pip install -r requirements.txt
```

### Step 2: Setup PostgreSQL Database

**Option A: Automated Setup (macOS/Linux)**
```bash
./setup_postgresql.sh
```

**Option B: Manual Setup**

1. **Install PostgreSQL** (if not installed):
   - macOS: `brew install postgresql@15 && brew services start postgresql@15`
   - Linux: `sudo apt-get install postgresql && sudo systemctl start postgresql`

2. **Create database and user**:
   ```bash
   createdb klippr
   createuser klippr
   psql -d klippr -c "ALTER USER klippr WITH PASSWORD 'klippr';"
   psql -d klippr -c "GRANT ALL PRIVILEGES ON DATABASE klippr TO klippr;"
   ```

3. **Configure connection** (create/update `.env`):
   ```bash
   echo "DATABASE_URL=postgresql://klippr:klippr@localhost:5432/klippr" > .env
   ```

### Step 3: Migrate Data

```bash
# Backup SQLite database first!
cp readerz.db readerz.db.backup

# Run migration
python3 migrate_to_postgresql.py
```

## Verify

```bash
# Test connection
python3 -c "from database import engine; print('✓ PostgreSQL connected')"

# Check data
psql -d klippr -c "SELECT COUNT(*) FROM documents;"
psql -d klippr -c "SELECT COUNT(*) FROM chapters;"
```

## Start Application

```bash
# Backend will automatically use PostgreSQL when DATABASE_URL is set
python3 -m uvicorn main:app --reload
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'psycopg2'"**
```bash
pip install psycopg2-binary
```

**"connection refused"**
```bash
# Check if PostgreSQL is running
pg_isready

# Start PostgreSQL
# macOS: brew services start postgresql@15
# Linux: sudo systemctl start postgresql
```

**"database does not exist"**
```bash
createdb klippr
```

**"password authentication failed"**
```bash
# Reset password
psql -d postgres -c "ALTER USER klippr WITH PASSWORD 'klippr';"
```

## Rollback to SQLite

If needed, you can rollback:

1. Update `.env`:
   ```
   DATABASE_URL=sqlite:///./readerz.db
   ```

2. Restore backup:
   ```bash
   cp readerz.db.backup readerz.db
   ```

3. Restart backend

---

**That's it!** Your application now uses PostgreSQL with true concurrent writes.

