# Database Optimization Quick Start

## What Changed

1. **SQLite PRAGMAs**: WAL mode, optimized cache, faster writes
2. **Bulk Inserts**: Replaced per-row adds with bulk operations
3. **Indexes**: Added indexes on foreign keys for faster queries
4. **Session Config**: Disabled autoflush and expire_on_commit for bulk ops

## Expected Performance

- **Before**: ~1.5s for 100 chapters
- **After**: ~0.2s for 100 chapters
- **Improvement**: **5-7.5x faster**

## Apply to Existing Database

Run once to add indexes:
```bash
cd backend
python3 migrate_add_indexes.py
```

## Test Performance

```bash
# Baseline (if you have old code)
python3 benchmark_baseline.py

# Optimized (current code)
python3 benchmark_optimized.py
```

## Rollback (if needed)

1. Revert `database.py` - remove PRAGMA event listener
2. Revert `main.py` - change `bulk_insert_mappings()` back to per-row `db.add()`
3. Restart server

## Production Notes

- **Current settings are safe** for development
- For production, consider changing `synchronous=NORMAL` to `synchronous=FULL` in `database.py`
- WAL mode is actually safer than default DELETE mode
- Indexes only help - no downside to keeping them

## Files Changed

- `backend/database.py` - PRAGMAs, session config, indexes
- `backend/main.py` - Bulk inserts in upload endpoints

See `OPTIMIZATION_REPORT.md` for full details.

