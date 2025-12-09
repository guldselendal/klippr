# Codebase Cleanup Plan

## Objective
Reduce module count by ≥30% while preserving behavior. Target: 41 modules → ~28 modules (32% reduction).

## Baseline Metrics

- **Total Python modules**: 41
- **Core modules** (keep): 15
- **Dead/unused modules** (remove): 5
- **Redundant modules** (merge): 3
- **Utility modules** (consolidate): 2
- **Test/diagnostic scripts** (keep): 8
- **Migration scripts** (keep): 6

## Module Inventory

### Core Modules (Keep - 15)
1. `main.py` - FastAPI app
2. `database.py` - DB connection and setup
3. `models.py` - SQLAlchemy models
4. `parsers.py` - EPUB/PDF parsing
5. `llm_provider.py` - LLM API interface
6. `llm_concurrency.py` - Concurrency limiting
7. `prompt_utils.py` - Prompt templates
8. `summarizer.py` - Summary generation
9. `summary_pipeline.py` - Async pipeline (optional)
10. `pipeline_metrics.py` - Metrics collection
11. `chunk_cache.py` - Caching layer
12. `utils.py` - Utility functions
13. `cost_tracking.py` - Cost tracking (used by pipeline_metrics)
14. `connections.py` - Chapter connections (used by main.py)
15. `__init__.py` - Package init

### Dead Code (Remove - 5)
1. `classifier.py` - Never imported, unused
2. `cover_extractor.py` - Never imported, unused
3. `similarity.py` - Never imported, unused
4. `chapter_processor.py` - Never imported, unused
5. `list_gemini_models.py` - Diagnostic script, unused

### Redundant Modules (Merge - 3)
1. `summarizer_parallel.py` → Merge into `summarizer.py`
   - Used in 2 places (main.py, summary_pipeline.py)
   - Thin wrapper around summarizer functions
   - Rationale: Reduces module count, consolidates summary logic

2. `pipeline_instrumentation.py` → Merge into `pipeline_metrics.py` OR remove
   - Used only in `diagnose_pipeline_385.py` (diagnostic script)
   - Rationale: Duplicate functionality with pipeline_metrics.py

3. `cost_tracking.py` → Merge into `pipeline_metrics.py` OR keep separate
   - Used only in `pipeline_metrics.py`
   - Rationale: Small module, could be inlined

### Test/Diagnostic Scripts (Keep - 8)
- `test_*.py` - Test scripts
- `diagnose_*.py` - Diagnostic scripts
- `benchmark_*.py` - Benchmark scripts
- `verify_*.py` - Verification scripts
- `generate_*.py` - Generation scripts
- `debug_*.py` - Debug scripts

### Migration Scripts (Keep - 6)
- `migrate_*.py` - Database migration scripts

## Consolidation Plan

### Step 1: Remove Dead Code (5 modules)
**Impact**: -5 modules, 0 risk
- Delete `classifier.py`
- Delete `cover_extractor.py`
- Delete `similarity.py`
- Delete `chapter_processor.py`
- Delete `list_gemini_models.py`

### Step 2: Merge summarizer_parallel into summarizer (1 module)
**Impact**: -1 module, low risk
- Move `generate_summaries_parallel()` and `process_summaries_for_titles_and_previews()` into `summarizer.py`
- Update imports in `main.py` and `summary_pipeline.py`
- Delete `summarizer_parallel.py`

### Step 3: Consolidate pipeline_instrumentation (1 module)
**Impact**: -1 module, low risk
- Option A: Merge into `pipeline_metrics.py` (if functionality overlaps)
- Option B: Remove if only used in diagnostic script
- Update `diagnose_pipeline_385.py` if needed

### Step 4: Consolidate cost_tracking (optional)
**Impact**: -1 module, low risk
- Option A: Merge into `pipeline_metrics.py`
- Option B: Keep separate (small, focused module)
- Decision: Keep separate for now (clear separation of concerns)

### Step 5: Review and remove unused dependencies
**Impact**: Reduced requirements.txt size
- Check for unused imports in requirements.txt
- Remove if not imported anywhere

## Expected Results

### Module Count
- **Before**: 41 modules
- **After**: ~28 modules (32% reduction)
- **Removed**: 5 dead code modules
- **Merged**: 2-3 redundant modules

### Code Reduction
- **Dead code removed**: ~150 lines
- **Merged code**: ~140 lines (summarizer_parallel)
- **Total reduction**: ~290 lines

### Risk Assessment
- **Low risk**: Removing dead code (never imported)
- **Low risk**: Merging summarizer_parallel (simple refactor)
- **Medium risk**: Consolidating pipeline_instrumentation (check usage)

## Rollback Plan

1. **Git commits**: Each step in separate commit
2. **Branch**: Create `cleanup/module-consolidation` branch
3. **Tests**: Run full test suite after each step
4. **Verification**: Check imports, run smoke tests

## Success Criteria

- [ ] Module count reduced by ≥30%
- [ ] All tests pass
- [ ] No import errors
- [ ] Behavior preserved (loose)
- [ ] Documentation updated

