# Upload Failure Fix: Timeout Issue

## Critical Issue Found

**Problem**: Axios client has a 10-second timeout, but uploads with summary generation can take 2-10+ minutes depending on:
- File size and number of chapters
- Ollama processing time (30-120 seconds per chapter)
- Parallel processing overhead

**Impact**: Uploads fail with timeout error before backend completes processing, showing generic "Failed to upload files" message.

## Fix Applied

### File: `frontend/src/api/client.ts`

**Changed**:
- `uploadFile()`: Increased timeout from 10s to **5 minutes** (300,000ms)
- `uploadFiles()`: Increased timeout from 10s to **10 minutes** (600,000ms)

**Rationale**:
- Single file upload: 5 minutes should be sufficient for most documents
- Batch upload: 10 minutes accounts for multiple files being processed sequentially

## Additional Recommendations

### 1. Add Upload Progress Tracking
Currently, status updates are simulated. Consider adding real progress tracking:

```typescript
uploadFile: async (file: File, onProgress?: (progress: number) => void): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 300000,
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress(percentCompleted);
      }
    },
  });
  return response.data;
}
```

### 2. Better Error Messages for Timeout
Update error handling to distinguish timeout from other errors:

```typescript
catch (error: any) {
  if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
    errorMsg = 'Upload timed out. The file may be too large or processing is taking longer than expected. Please try a smaller file or check if Ollama is running.';
  } else {
    errorMsg = error.response?.data?.detail || 'Failed to upload files';
  }
}
```

### 3. Backend: Add Streaming File Upload
Currently, entire file is loaded into memory. Consider streaming for large files:

```python
# Instead of: content = await file.read()
# Use streaming for large files
with open(file_path, "wb") as f:
    async for chunk in file.stream():
        f.write(chunk)
```

### 4. Backend: Add File Size Validation
Add explicit size limits with clear error messages:

```python
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
file_size = 0
async for chunk in file.stream():
    file_size += len(chunk)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({file_size} bytes) exceeds maximum allowed size (100 MB)"
        )
```

## Testing

After applying the fix:

1. **Test small file** (< 1 MB): Should upload successfully
2. **Test medium file** (10-50 MB): Should upload with timeout fix
3. **Test large file** (> 100 MB): May still timeout if processing takes too long
4. **Test with Ollama down**: Should show clear error about Ollama connection
5. **Test batch upload**: Should handle multiple files with extended timeout

## Monitoring

Watch for:
- Uploads still timing out (may need further timeout increase)
- Memory issues with very large files
- Ollama connection failures during processing
- Network errors during long uploads

## Rollback

If issues occur, revert timeout changes:
- `uploadFile`: Remove `timeout: 300000` (use default 10s)
- `uploadFiles`: Remove `timeout: 600000` (use default 10s)

