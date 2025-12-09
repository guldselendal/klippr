import { Document, Chapter } from '../api/client';

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

class CacheService {
  private cache = new Map<string, CacheEntry<unknown>>();
  private readonly CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

  set<T>(key: string, data: T): void {
    this.cache.set(key, { data, timestamp: Date.now() });
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;
    if (Date.now() - entry.timestamp > this.CACHE_DURATION) {
      this.cache.delete(key);
      return null;
    }
    return entry.data as T;
  }

  delete(key: string): void {
    this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  // Specific cache methods for documents and chapters
  setDocuments(documents: Document[]): void {
    this.set('documents', documents);
  }

  getDocuments(): Document[] | null {
    return this.get<Document[]>('documents');
  }

  setDocumentChapters(documentId: string, chapters: Chapter[]): void {
    this.set(`document_${documentId}_chapters`, chapters);
  }

  getDocumentChapters(documentId: string): Chapter[] | null {
    return this.get<Chapter[]>(`document_${documentId}_chapters`);
  }

  setAllChapters(chapters: Chapter[]): void {
    this.set('all_chapters', chapters);
  }

  getAllChapters(): Chapter[] | null {
    return this.get<Chapter[]>('all_chapters');
  }

  setChapter(chapterId: string, chapter: Chapter): void {
    this.set(`chapter_${chapterId}`, chapter);
  }

  getChapter(chapterId: string): Chapter | null {
    return this.get<Chapter>(`chapter_${chapterId}`);
  }

  invalidateDocument(documentId: string): void {
    this.delete(`document_${documentId}_chapters`);
    this.delete('all_chapters');
  }

  invalidateAll(): void {
    this.clear();
  }
}

export const cacheService = new CacheService();
