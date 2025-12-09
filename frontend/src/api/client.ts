import axios, { AxiosError } from 'axios';
import { cacheService } from '../services/cache';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  // No default timeout - let individual requests set their own timeouts if needed
});

// Extract user-friendly error message from AxiosError
const getErrorMessage = (error: AxiosError): string => {
  if (error.code === 'ECONNREFUSED' || 
      error.message?.includes('Network Error') ||
      error.message?.includes('ERR_CONNECTION_REFUSED')) {
    return 'Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000';
  }
  if (error.code === 'ECONNABORTED') {
    return 'Request timed out. The backend server may be slow or unavailable.';
  }
  if (error.response) {
    const data = error.response.data as { detail?: string };
    return data?.detail || `Server error: ${error.response.status}`;
  }
  return 'Network error. Please check your connection.';
};

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    (error as AxiosError & { userMessage: string }).userMessage = getErrorMessage(error);
    return Promise.reject(error);
  }
);

// Retry helper with exponential backoff
const retryRequest = async <T>(
  requestFn: () => Promise<T>,
  maxRetries = 2,
  delay = 1000
): Promise<T> => {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await requestFn();
    } catch (error) {
      const axiosError = error as AxiosError;
      // Don't retry on 4xx errors or last attempt
      if ((axiosError.response?.status ?? 0) >= 400 && axiosError.response?.status! < 500) {
        throw error;
      }
      if (attempt === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, attempt)));
    }
  }
  throw new Error('Max retries exceeded');
};

export const checkBackendHealth = async (): Promise<boolean> => {
  try {
    const response = await apiClient.get('/', { timeout: 2000 });
    return response.status === 200;
  } catch {
    return false;
  }
};

export interface Document {
  id: string;
  title: string;
  file_type: string;
  chapter_count: number;
  uploaded_at: string | null;
}

export interface Chapter {
  id: string;
  title: string;
  content?: string;
  summary?: string;
  preview?: string;
  document_id: string;
  document_title: string;
  chapter_number?: number;
}

export interface Connection {
  chapter1: Chapter;
  chapter2: Chapter;
  similarity: number;
  reason: string;
}

const UPLOAD_TIMEOUT = 1800000; // 30 minutes

export const api = {
  getDocuments: async (useCache = true): Promise<Document[]> => {
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

  getDocumentChapters: async (documentId: string, includeContent = false): Promise<Chapter[]> => {
    if (!includeContent) {
      const cached = cacheService.getDocumentChapters(documentId);
      if (cached) return cached;
    }
    const response = await apiClient.get(`/api/documents/${documentId}/chapters`, {
      params: { include_content: includeContent }
    });
    const chapters = response.data.chapters;
    if (!includeContent) {
      cacheService.setDocumentChapters(documentId, chapters);
    }
    return chapters;
  },

  deleteDocument: async (documentId: string): Promise<void> => {
    await apiClient.delete(`/api/documents/${documentId}`);
    cacheService.invalidateDocument(documentId);
    cacheService.delete('documents');
  },

  deleteDocuments: async (documentIds: string[]): Promise<unknown> => {
    const response = await apiClient.post('/api/documents/delete-batch', { document_ids: documentIds });
    documentIds.forEach(id => cacheService.invalidateDocument(id));
    cacheService.delete('documents');
    return response.data;
  },

  getAllChapters: async (includeContent = false): Promise<Chapter[]> => {
    if (!includeContent) {
      const cached = cacheService.getAllChapters();
      if (cached) return cached;
    }
    const response = await apiClient.get('/api/chapters', {
      params: { include_content: includeContent }
    });
    const chapters = response.data.chapters;
    if (!includeContent) {
      cacheService.setAllChapters(chapters);
    }
    return chapters;
  },

  getChapter: async (chapterId: string): Promise<Chapter> => {
    const cached = cacheService.getChapter(chapterId);
    if (cached?.content) return cached;
    const response = await apiClient.get(`/api/chapters/${chapterId}`);
    const chapter = response.data.chapter;
    cacheService.setChapter(chapterId, chapter);
    return chapter;
  },

  uploadFile: async (file: File): Promise<unknown> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post('/api/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: UPLOAD_TIMEOUT,
    });
    // Invalidate cache to ensure fresh data after upload
    cacheService.delete('documents');
    cacheService.delete('all_chapters');
    return response.data;
  },

  uploadFiles: async (files: File[]): Promise<unknown> => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    const response = await apiClient.post('/api/upload/batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: UPLOAD_TIMEOUT,
    });
    // Invalidate cache to ensure fresh data after upload
    cacheService.delete('documents');
    cacheService.delete('all_chapters');
    return response.data;
  },

  getConnections: async (): Promise<Connection[]> => {
    const response = await apiClient.get('/api/connections');
    return response.data.connections;
  },

  summarizeChapter: async (chapterId: string): Promise<Chapter> => {
    // No timeout - summary generation can take a long time (especially for long chapters)
    const response = await apiClient.post(`/api/chapters/${chapterId}/summarize`, {}, {
      timeout: 0, // 0 means no timeout
    });
    const chapter = response.data.chapter;
    cacheService.delete(`chapter_${chapterId}`);
    cacheService.invalidateDocument(chapter.document_id);
    return chapter;
  },
};
