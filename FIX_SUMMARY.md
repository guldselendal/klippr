# Fix Summary: "Failed to load documents" Error

## Changes Implemented

### 1. Enhanced Error Messages (`frontend/src/components/Library.tsx`)
- **Before**: Generic "Failed to load documents" alert
- **After**: Specific error messages based on error type:
  - Connection refused: "Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000"
  - Server errors (500): "Backend server error. Please check server logs."
  - Not found (404): "Documents endpoint not found. Please check backend configuration."
  - Other errors: Shows actual error message or response detail

### 2. Retry Logic (`frontend/src/api/client.ts`)
- Added `retryRequest()` helper function with exponential backoff
- Automatically retries failed requests (max 2 retries)
- Skips retry for 4xx client errors (immediate failure)
- Retries network errors and 5xx server errors

### 3. Backend Health Check (`frontend/src/api/client.ts` & `frontend/src/components/Library.tsx`)
- Added `checkBackendHealth()` function
- Checks backend availability before loading documents
- Shows clear message if backend is unavailable
- Prevents unnecessary API calls when backend is down

### 4. Axios Interceptor (`frontend/src/api/client.ts`)
- Added response interceptor for consistent error handling
- Enhances error objects with user-friendly messages
- Handles connection refused, timeouts, and server errors

### 5. Backend Health Endpoint (`backend/main.py`)
- Added `GET /health` endpoint
- Returns `{"status": "ok", "message": "Backend is running"}`
- Used by frontend health check

### 6. Request Timeout (`frontend/src/api/client.ts`)
- Added 10-second timeout to axios client
- Prevents hanging requests
- Better timeout error handling

## Testing

### Manual Test Steps

1. **Test Backend Not Running**:
   ```bash
   # Don't start backend
   cd frontend && npm run dev
   # Open browser, refresh page
   # Expected: "Backend server is not available..." message
   ```

2. **Test Backend Running**:
   ```bash
   # Start backend
   cd backend && python3 -m uvicorn main:app --reload
   # Start frontend
   cd frontend && npm run dev
   # Open browser, refresh page
   # Expected: Documents load successfully
   ```

3. **Test Retry Logic**:
   ```bash
   # Start backend, then stop it during request
   # Expected: Retry attempts, then clear error message
   ```

## Files Modified

1. `frontend/src/components/Library.tsx` - Enhanced error handling and health check
2. `frontend/src/api/client.ts` - Retry logic, interceptor, health check function
3. `backend/main.py` - Health check endpoint

## Expected Behavior After Fix

- **Backend Down**: Clear message indicating backend needs to be started
- **Backend Slow**: Automatic retry with exponential backoff
- **Backend Error**: Specific error message based on error type
- **Network Issues**: Retry logic attempts reconnection
- **Backend Up**: Documents load normally

## Rollback

If issues occur, revert these files to their previous versions:
- `frontend/src/components/Library.tsx`
- `frontend/src/api/client.ts`
- `backend/main.py`

All changes are backward compatible and don't break existing functionality.

