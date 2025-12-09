# Troubleshooting Report: "Failed to load documents" Error

## Executive Summary

**Root Cause**: The error occurs when the frontend cannot connect to the backend API server (`http://localhost:8000`) on page refresh, most commonly because the backend server is not running or not accessible. The generic error handling masks the actual network/connection issue.

**Impact**: Users see a generic "Failed to load documents" alert without actionable information, making debugging difficult.

## Reproduction Steps

### Environment
- **OS**: macOS (darwin 24.6.0)
- **Browser**: Chrome/Chromium-based (any version)
- **Frontend**: React + Vite dev server (typically `http://localhost:3000`)
- **Backend**: FastAPI + Uvicorn (expected at `http://localhost:8000`)
- **Time**: Occurs on any page refresh when backend is unavailable

### Steps to Reproduce
1. Start frontend dev server: `npm run dev` (runs on port 3000)
2. **Do NOT start backend server** (or stop it if running)
3. Navigate to `http://localhost:3000` in browser
4. Refresh the page (F5 or Cmd+R)
5. **Observed**: Pop-up alert "Failed to load documents" appears immediately

### Expected vs Actual
- **Expected**: Either graceful handling with retry, or clear error message indicating backend is unavailable
- **Actual**: Generic alert with no actionable information

## Evidence

### Console Errors (Expected)
```
Error loading documents: Error: Network Error
    at createError (axios.js:...)
    at XMLHttpRequest.handleError (xhr.js:...)
```

Or:
```
Error loading documents: AxiosError: connect ECONNREFUSED 127.0.0.1:8000
```

### Network Tab (Expected)
- **Request**: `GET http://localhost:8000/api/documents`
- **Status**: `(failed)` or `ERR_CONNECTION_REFUSED`
- **Type**: `xhr` or `fetch`
- **Timing**: Fails immediately (< 100ms)
- **Response**: No response body (connection refused)

### Code Analysis

**Frontend Error Handling** (`frontend/src/components/Library.tsx:24-35`):
```typescript
const loadDocuments = async () => {
  try {
    setLoading(true);
    const docs = await api.getDocuments();
    setDocuments(docs);
  } catch (error) {
    console.error('Error loading documents:', error);
    alert('Failed to load documents');  // ❌ Generic, unhelpful message
  } finally {
    setLoading(false);
  }
};
```

**API Client** (`frontend/src/api/client.ts:41-51`):
```typescript
getDocuments: async (useCache: boolean = true): Promise<Document[]> => {
  if (useCache) {
    const cached = cacheService.getDocuments();
    if (cached) return cached;  // ✅ Cache works, but cleared on refresh
  }
  
  const response = await apiClient.get('/api/documents');
  // ❌ No error handling here, throws to caller
  const documents = response.data.documents;
  cacheService.setDocuments(documents);
  return documents;
}
```

**Cache Service** (`frontend/src/services/cache.ts`):
- Uses in-memory `Map` - **cleared on page refresh**
- No persistence to localStorage/sessionStorage
- Cache duration: 5 minutes

## Analysis

### Top Hypotheses (Ranked)

1. **Backend Server Not Running** (Most Likely - 90%)
   - **Evidence**: Error occurs specifically on refresh when cache is empty
   - **Supporting**: Generic error message suggests network-level failure
   - **Test**: Check if backend process is running on port 8000
   - **Expected Error**: `ECONNREFUSED` or `Network Error`

2. **Backend Server on Different Port** (5%)
   - **Evidence**: Frontend defaults to `http://localhost:8000`
   - **Supporting**: If backend runs on different port, connection fails
   - **Test**: Check `VITE_API_URL` env var and backend port config
   - **Expected Error**: `ECONNREFUSED` on wrong port

3. **CORS Preflight Failure** (3%)
   - **Evidence**: CORS configured but could fail if backend not running
   - **Supporting**: CORS middleware only works if backend responds
   - **Test**: Check Network tab for OPTIONS request failure
   - **Expected Error**: CORS error in console, but connection refused happens first

4. **Race Condition During Bootstrap** (2%)
   - **Evidence**: Error on refresh suggests timing issue
   - **Supporting**: `useEffect` runs immediately, no delay
   - **Test**: Add delay before fetch, check if error persists
   - **Expected Error**: Same network error, but timing-dependent

### Ruled Out
- **Authentication/Session**: No auth implemented in current codebase
- **Cache Corruption**: Cache is in-memory, cleared on refresh
- **Service Worker**: No service worker configured
- **Token Expiry**: No token-based auth system

## Root Cause

**Primary**: Backend server (`http://localhost:8000`) is not running or not accessible when the frontend attempts to fetch documents on page refresh.

**Contributing Factors**:
1. **Generic Error Handling**: Catches all errors and shows unhelpful message
2. **No Retry Logic**: Single attempt, fails immediately
3. **No Health Check**: Frontend doesn't verify backend availability
4. **Cache Cleared on Refresh**: In-memory cache doesn't persist, forcing network request
5. **No User Feedback**: Alert doesn't indicate what went wrong or how to fix

**User Impact Scope**:
- **Severity**: Medium (app unusable, but clear cause)
- **Frequency**: Every page refresh when backend is down
- **Affected Users**: All users when backend is unavailable
- **Workaround**: Start backend server, but error doesn't indicate this

## Fix

### 1. Improved Error Handling with Specific Messages

**File**: `frontend/src/components/Library.tsx`

```typescript
const loadDocuments = async () => {
  try {
    setLoading(true);
    const docs = await api.getDocuments();
    setDocuments(docs);
  } catch (error: any) {
    console.error('Error loading documents:', error);
    
    // Provide specific error messages
    let errorMessage = 'Failed to load documents';
    
    if (error.code === 'ECONNREFUSED' || error.message?.includes('Network Error')) {
      errorMessage = 'Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000';
    } else if (error.response?.status === 500) {
      errorMessage = 'Backend server error. Please check server logs.';
    } else if (error.response?.status === 404) {
      errorMessage = 'Documents endpoint not found. Please check backend configuration.';
    } else if (error.response?.data?.detail) {
      errorMessage = `Error: ${error.response.data.detail}`;
    }
    
    alert(errorMessage);
  } finally {
    setLoading(false);
  }
};
```

### 2. Add Retry Logic with Exponential Backoff

**File**: `frontend/src/api/client.ts`

```typescript
const retryRequest = async <T>(
  requestFn: () => Promise<T>,
  maxRetries = 3,
  delay = 1000
): Promise<T> => {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await requestFn();
    } catch (error: any) {
      // Don't retry on 4xx errors (client errors)
      if (error.response?.status >= 400 && error.response?.status < 500) {
        throw error;
      }
      
      // Last attempt, throw error
      if (attempt === maxRetries - 1) {
        throw error;
      }
      
      // Wait before retry (exponential backoff)
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, attempt)));
    }
  }
  throw new Error('Max retries exceeded');
};

export const api = {
  getDocuments: async (useCache: boolean = true): Promise<Document[]> => {
    if (useCache) {
      const cached = cacheService.getDocuments();
      if (cached) return cached;
    }
    
    return retryRequest(async () => {
      const response = await apiClient.get('/api/documents');
      const documents = response.data.documents;
      cacheService.setDocuments(documents);
      return documents;
    });
  },
  // ... rest of API methods
};
```

### 3. Add Backend Health Check

**File**: `frontend/src/api/client.ts`

```typescript
export const checkBackendHealth = async (): Promise<boolean> => {
  try {
    const response = await apiClient.get('/', { timeout: 2000 });
    return response.status === 200;
  } catch {
    return false;
  }
};

// Use in Library component before loading documents
```

**File**: `frontend/src/components/Library.tsx`

```typescript
useEffect(() => {
  const initialize = async () => {
    const isHealthy = await checkBackendHealth();
    if (!isHealthy) {
      alert('Backend server is not available. Please start the backend server on http://localhost:8000');
      return;
    }
    loadDocuments();
  };
  initialize();
}, []);
```

### 4. Add Axios Interceptor for Better Error Handling

**File**: `frontend/src/api/client.ts`

```typescript
// Add response interceptor for consistent error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Enhance error object with more context
    if (error.code === 'ECONNREFUSED' || error.message?.includes('Network Error')) {
      error.userMessage = 'Cannot connect to backend server. Please ensure the backend is running.';
    } else if (error.response) {
      error.userMessage = error.response.data?.detail || `Server error: ${error.response.status}`;
    } else {
      error.userMessage = 'Network error. Please check your connection.';
    }
    return Promise.reject(error);
  }
);
```

## Validation Plan

### Test Cases

1. **Backend Not Running**
   - Stop backend server
   - Refresh page
   - **Expected**: Clear error message indicating backend is unavailable
   - **Verify**: Error message suggests starting backend

2. **Backend Running**
   - Start backend server
   - Refresh page
   - **Expected**: Documents load successfully
   - **Verify**: No error alert, documents displayed

3. **Backend Slow Response**
   - Add delay to backend endpoint (simulate slow DB)
   - Refresh page
   - **Expected**: Retry logic handles timeout, eventually succeeds or shows timeout error
   - **Verify**: Multiple retry attempts visible in Network tab

4. **Backend Returns 500**
   - Mock backend to return 500 error
   - Refresh page
   - **Expected**: Error message indicates server error, not connection issue
   - **Verify**: Error message is specific to 500 status

5. **Network Interruption**
   - Disconnect network during request
   - **Expected**: Retry logic attempts reconnection
   - **Verify**: Error message after max retries

### Manual Testing Steps

```bash
# Test 1: Backend not running
cd backend && # Don't start server
cd ../frontend && npm run dev
# Open browser, refresh page
# Expected: Clear error message

# Test 2: Backend running
cd backend && python3 -m uvicorn main:app --reload
cd ../frontend && npm run dev
# Open browser, refresh page
# Expected: Documents load successfully

# Test 3: Wrong port
# Set VITE_API_URL=http://localhost:9999
# Start frontend
# Expected: Error message indicates connection refused
```

## Preventative Measures

### 1. Monitoring & Alerts

**Frontend Monitoring**:
- Add error tracking (e.g., Sentry) to capture errors in production
- Log all API failures with context (endpoint, status, error type)
- Track backend availability metrics

**Backend Monitoring**:
- Health check endpoint: `GET /health` returning `{"status": "ok"}`
- Log all requests with correlation IDs
- Monitor response times and error rates

### 2. Automated Tests

**E2E Tests** (Playwright/Cypress):
```typescript
test('should show error when backend is unavailable', async ({ page }) => {
  // Mock backend to be unavailable
  await page.route('http://localhost:8000/api/documents', route => route.abort());
  
  await page.goto('http://localhost:3000');
  await page.waitForSelector('text=Failed to load documents');
  
  // Verify error message is helpful
  const errorText = await page.textContent('.error-message');
  expect(errorText).toContain('backend server');
});
```

**Unit Tests**:
```typescript
describe('loadDocuments', () => {
  it('should show specific error when backend is unavailable', async () => {
    jest.spyOn(api, 'getDocuments').mockRejectedValue({
      code: 'ECONNREFUSED',
      message: 'Network Error'
    });
    
    // Render component and trigger load
    // Verify error message is specific
  });
});
```

### 3. Runbooks

**For Developers**:
1. **Symptom**: "Failed to load documents" alert on refresh
2. **Check**: Is backend running? `curl http://localhost:8000/`
3. **If not running**: Start backend: `cd backend && python3 -m uvicorn main:app --reload`
4. **If running**: Check logs for errors, verify CORS config
5. **If CORS issue**: Verify `allow_origins` includes frontend URL

**For Users**:
- Add "Backend Status" indicator in UI
- Show connection status in footer/header
- Provide "Retry" button when error occurs

### 4. Code Improvements

**Add Health Check Endpoint** (`backend/main.py`):
```python
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
```

**Add Frontend Status Component**:
```typescript
const BackendStatus = () => {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  
  useEffect(() => {
    checkBackendHealth().then(isHealthy => {
      setStatus(isHealthy ? 'online' : 'offline');
    });
  }, []);
  
  return <div className={`backend-status ${status}`}>Backend: {status}</div>;
};
```

## Implementation Priority

1. **Immediate** (Fix Now):
   - Improve error messages in `loadDocuments()` function
   - Add specific error handling for connection refused

2. **Short-term** (This Week):
   - Add retry logic with exponential backoff
   - Add backend health check
   - Add health check endpoint to backend

3. **Medium-term** (This Month):
   - Add error tracking/monitoring
   - Add E2E tests for error scenarios
   - Add backend status indicator in UI

4. **Long-term** (Ongoing):
   - Implement comprehensive error handling across all API calls
   - Add retry logic to all network requests
   - Set up production monitoring and alerts

