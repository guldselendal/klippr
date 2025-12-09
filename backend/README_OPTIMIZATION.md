# Performance Optimization - Caching and Summaries

## Overview

The application has been optimized to significantly reduce loading times by:

1. **Caching**: Frontend caches document and chapter metadata for 5 minutes
2. **Summaries**: Chapters now have AI-generated summaries stored in the database
3. **Lazy Loading**: Full chapter content is only loaded when actually reading

## Database Migration

If you have an existing database, you need to:

1. **Add the summary column**:
```bash
cd backend
python migrate_add_summaries.py
```

2. **Generate summaries for existing chapters** (optional but recommended):
```bash
python generate_missing_summaries.py
```

Note: Summary generation requires an OpenAI API key. If not set, summaries will be simple truncations of the content.

## Performance Improvements

### Before:
- Loading library: Fetched all chapters with full content (~MBs of data)
- Opening a document: Downloaded all chapter content
- Navigation: Re-fetched all data repeatedly

### After:
- Loading library: Fetches only document metadata (cached)
- Opening a document: Fetches chapter summaries only (~KB of data, cached)
- Reading a chapter: Loads full content only for the current chapter
- Navigation: Uses cached summaries, only loads new chapter content

## API Changes

### New Endpoints:
- `GET /api/chapters/{chapter_id}` - Get single chapter with full content

### Updated Endpoints:
- `GET /api/chapters?include_content=false` - Get all chapters with summaries only (default)
- `GET /api/documents/{document_id}/chapters?include_content=false` - Get document chapters with summaries only (default)

## Frontend Caching

The frontend now uses a cache service that:
- Caches documents list for 5 minutes
- Caches chapter summaries (without content) for 5 minutes
- Caches individual chapter content when loaded
- Automatically invalidates cache on document deletion

## Summary Generation

Summaries are automatically generated when:
- New documents are uploaded
- Using Ollama by default (local, no API key needed)
- Supports multiple providers: Ollama, OpenAI, Gemini, DeepSeek
- Falls back to content truncation if LLM unavailable

### LLM Provider Configuration

The system uses a unified LLM provider interface (`llm_provider.py`) that supports:
- **Ollama** (default) - Local, free, no API key needed
- **OpenAI** - Requires OPENAI_API_KEY
- **Gemini** - Requires GEMINI_API_KEY  
- **DeepSeek** - Requires DEEPSEEK_API_KEY

**Environment Variables:**
- `API_PROVIDER` - Set to "ollama", "openai", "gemini", or "deepseek" (default: "ollama")
- `OLLAMA_URL` - Ollama server URL (default: "http://localhost:11434")
- `OLLAMA_MODEL` - Model name (default: "phi3:mini")
- `OPENAI_MODEL` - OpenAI model (default: "gpt-4-turbo-preview")
- `GEMINI_MODEL` - Gemini model (default: "gemini-1.5-flash")
- `DEEPSEEK_MODEL` - DeepSeek model (default: "deepseek-chat")

**Note:** Summaries always use Ollama regardless of API_PROVIDER setting. To use a different provider, modify `summarizer.py`.

To regenerate summaries for existing chapters:
```bash
python generate_missing_summaries.py
```

