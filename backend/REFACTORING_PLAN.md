# Backend Refactoring Plan

## Summary
Refactoring to simplify functions, ensure smooth module transitions, and remove redundant code while preserving behavior.

## Key Simplifications

1. **Database Session Management**: Extract to context manager (39 instances → 0 manual close calls)
2. **Utility Functions**: Consolidate duplicate filename/validation logic
3. **Error Handling**: Standardize patterns across endpoints
4. **Response Formatting**: Extract common response builders
5. **LLM Client Initialization**: Unify provider client patterns

## Files to Refactor

### High Priority (Large files with duplication)
- `main.py` (923 lines) - Extract utilities, use context managers, standardize responses
- `llm_provider.py` (361 lines) - Unify client initialization patterns
- `summary_pipeline.py` (897 lines) - Extract helper functions, reduce nesting

### Medium Priority
- `summarizer.py` (466 lines) - Extract chunk processing logic
- `database.py` (208 lines) - Already well-structured, minor improvements

## Metrics

### Before
- Total LOC: ~8,946 lines
- Database session patterns: 39 manual open/close
- Duplicate utilities: 3 functions (clean_filename, normalize_book_name, check_duplicate)
- Inconsistent error handling: Multiple patterns

### Target
- Reduce manual db.close() calls: 39 → 0
- Consolidate utilities: 3 → 1 module
- Standardize error handling: Unified pattern
- Improve code maintainability: Single responsibility per function

