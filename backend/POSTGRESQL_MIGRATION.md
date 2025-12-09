# PostgreSQL Migration Guide

## Overview

This guide will help you migrate from SQLite to PostgreSQL to enable true concurrent writes and eliminate single-writer contention.

## Benefits of PostgreSQL

- **True Concurrency**: Multiple writers can operate simultaneously
- **Better Performance**: Optimized for high-concurrency workloads
- **Connection Pooling**: Efficient connection management
- **Advanced Features**: Better indexing, full-text search, etc.
- **Production Ready**: Industry-standard database

## Prerequisites

1. **PostgreSQL installed** (version 12 or higher)
   - macOS: `brew install postgresql@15`
   - Linux: `sudo apt-get install postgresql postgresql-contrib`
   - Windows: Download from [postgresql.org](https://www.postgresql.org/download/)

2. **Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This will install `psycopg2-binary` (PostgreSQL driver).

## Step 1: Install and Start PostgreSQL

### macOS (Homebrew)
```bash
# Install PostgreSQL
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Create database and user
createdb klippr
createuser klippr
psql -d klippr -c "ALTER USER klippr WITH PASSWORD 'klippr';"
psql -d klippr -c "GRANT ALL PRIVILEGES ON DATABASE klippr TO klippr;"
```

### Linux (Ubuntu/Debian)
```bash
# Install PostgreSQL
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Switch to postgres user and create database
sudo -u postgres psql
```

Then in PostgreSQL prompt:
```sql
CREATE DATABASE klippr;
CREATE USER klippr WITH PASSWORD 'klippr';
GRANT ALL PRIVILEGES ON DATABASE klippr TO klippr;
\q
```

### Windows
1. Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
2. During installation, set password for `postgres` user
3. Open pgAdmin or psql and create database:
   ```sql
   CREATE DATABASE klippr;
   CREATE USER klippr WITH PASSWORD 'klippr';
   GRANT ALL PRIVILEGES ON DATABASE klippr TO klippr;
   ```

## Step 2: Configure Database URL

Create or update `.env` file in the `backend` directory:

```bash
# PostgreSQL connection string
DATABASE_URL=postgresql://klippr:klippr@localhost:5432/klippr

# Or with custom credentials:
# DATABASE_URL=postgresql://username:password@localhost:5432/dbname
```

**Connection String Format:**
```
postgresql://[user]:[password]@[host]:[port]/[database]
```

**Common Variations:**
- Local with custom port: `postgresql://klippr:klippr@localhost:5433/klippr`
- Remote server: `postgresql://user:pass@db.example.com:5432/klippr`
- With SSL: `postgresql://user:pass@host:5432/db?sslmode=require`

## Step 3: Test PostgreSQL Connection

```bash
cd backend
python3 -c "
from database import engine
with engine.connect() as conn:
    result = conn.execute('SELECT version()')
    print('✓ PostgreSQL connected:', result.fetchone()[0])
"
```

## Step 4: Migrate Data from SQLite

**⚠️ Important: Backup your SQLite database first!**

```bash
# Backup SQLite database
cp readerz.db readerz.db.backup

# Run migration script
cd backend
python3 migrate_to_postgresql.py
```

The migration script will:
1. Export all data from SQLite
2. Create tables in PostgreSQL
3. Import all data
4. Verify data integrity

**If you have a lot of data**, the migration may take a few minutes. Progress will be shown.

## Step 5: Verify Migration

```bash
# Check document count
psql -d klippr -c "SELECT COUNT(*) FROM documents;"

# Check chapter count
psql -d klippr -c "SELECT COUNT(*) FROM chapters;"

# Verify a sample document
psql -d klippr -c "SELECT id, title FROM documents LIMIT 5;"
```

## Step 6: Update Application

The application will automatically use PostgreSQL when `DATABASE_URL` is set. No code changes needed!

**Restart your backend server:**
```bash
cd backend
python3 -m uvicorn main:app --reload
```

## Step 7: Test the Application

1. Start backend: `cd backend && python3 -m uvicorn main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Test upload: Upload a document and verify it works
4. Check logs: Look for "Connected to PostgreSQL" messages

## Troubleshooting

### Connection Refused
```
Error: connection refused
```
**Solution**: Ensure PostgreSQL is running:
- macOS: `brew services list` (check postgresql@15)
- Linux: `sudo systemctl status postgresql`
- Windows: Check Services panel

### Authentication Failed
```
Error: password authentication failed
```
**Solution**: Verify username/password in `DATABASE_URL`:
```bash
psql -U klippr -d klippr -h localhost
# Enter password when prompted
```

### Database Does Not Exist
```
Error: database "klippr" does not exist
```
**Solution**: Create the database:
```bash
createdb klippr
```

### Permission Denied
```
Error: permission denied for database
```
**Solution**: Grant privileges:
```sql
GRANT ALL PRIVILEGES ON DATABASE klippr TO klippr;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO klippr;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO klippr;
```

### Migration Script Errors
If migration fails:
1. Check PostgreSQL logs: `tail -f /var/log/postgresql/postgresql-*.log` (Linux)
2. Verify SQLite database is readable: `sqlite3 readerz.db "SELECT COUNT(*) FROM documents;"`
3. Check disk space: `df -h` (Linux/macOS)
4. Try migrating in smaller batches (modify `batch_size` in migration script)

## Performance Tuning (Optional)

For optimal performance, you can tune PostgreSQL settings in `postgresql.conf`:

```conf
# Connection settings
max_connections = 100

# Memory settings (adjust based on available RAM)
shared_buffers = 256MB          # 25% of RAM for small systems
effective_cache_size = 1GB      # 50-75% of RAM
work_mem = 16MB                 # Per-connection memory for sorts
maintenance_work_mem = 128MB    # For VACUUM, CREATE INDEX

# Write performance
synchronous_commit = on         # Balance speed/durability
checkpoint_completion_target = 0.9
wal_buffers = 16MB

# Query performance
random_page_cost = 1.1          # For SSDs (default 4.0 for HDDs)
effective_io_concurrency = 200   # For SSDs
```

After changes, restart PostgreSQL:
- macOS: `brew services restart postgresql@15`
- Linux: `sudo systemctl restart postgresql`

## Rollback to SQLite (If Needed)

If you need to rollback:

1. Set `DATABASE_URL` back to SQLite:
   ```bash
   # In .env file
   DATABASE_URL=sqlite:///./readerz.db
   ```

2. Restore SQLite backup:
   ```bash
   cp readerz.db.backup readerz.db
   ```

3. Restart backend server

## Production Considerations

For production deployments:

1. **Use connection pooling**: Already configured (pool_size=10, max_overflow=20)
2. **Set strong password**: Change default `klippr` password
3. **Enable SSL**: Use `?sslmode=require` in connection string
4. **Regular backups**: Set up `pg_dump` cron jobs
5. **Monitor performance**: Use `pg_stat_activity` and `pg_stat_statements`
6. **Tune PostgreSQL**: Adjust `postgresql.conf` based on workload

## Next Steps

After migration:
1. ✅ Test all functionality (upload, read, delete)
2. ✅ Monitor performance (check backend logs)
3. ✅ Keep SQLite backup for 1-2 weeks
4. ✅ Consider implementing the async producer/consumer architecture from `PERFORMANCE_OPTIMIZATION_PLAN.md`

## Support

If you encounter issues:
1. Check PostgreSQL logs
2. Verify `DATABASE_URL` format
3. Test connection with `psql`
4. Review migration script output

---

**Migration Complete!** Your application now uses PostgreSQL with true concurrent write support.

