import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import PyPDF2
import pdfplumber
from typing import List, Dict


def parse_epub(file_path: str) -> List[Dict[str, str]]:
    """Parse EPUB file and extract chapters"""
    book = epub.read_epub(file_path)
    chapters = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Extract text from HTML
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            
            # Try to get title from the item name or first heading
            title = item.get_name() or "Untitled Chapter"
            if soup.find(['h1', 'h2', 'h3']):
                title = soup.find(['h1', 'h2', 'h3']).get_text().strip()
            
            # Only add if there's substantial content
            if len(text) > 100:  # Minimum content length
                chapters.append({
                    'title': title,
                    'content': text
                })
    
    # If no chapters found, try to get all text as one chapter
    if not chapters:
        all_text = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                if text:
                    all_text.append(text)
        
        if all_text:
            chapters.append({
                'title': book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "Document",
                'content': ' '.join(all_text)
            })
    
    return chapters


def parse_pdf(file_path: str) -> List[Dict[str, str]]:
    """Parse PDF file and extract chapters"""
    chapters = []
    
    try:
        # Try using pdfplumber first (better for text extraction)
        with pdfplumber.open(file_path) as pdf:
            all_text = []
            current_chapter = None
            current_content = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text:
                    continue
                
                # Try to detect chapter headings (lines that are short and possibly bold/centered)
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Heuristic: if line is short, uppercase, or contains "Chapter", it might be a heading
                    is_likely_heading = (
                        len(line) < 100 and
                        (line.isupper() or 
                         'chapter' in line.lower() or 
                         line[0].isupper() and len(line.split()) < 10)
                    )
                    
                    if is_likely_heading and current_content:
                        # Save previous chapter
                        if current_chapter and current_content:
                            chapters.append({
                                'title': current_chapter,
                                'content': '\n'.join(current_content)
                            })
                        current_chapter = line
                        current_content = []
                    else:
                        current_content.append(line)
                
                all_text.append(text)
            
            # Add final chapter
            if current_chapter and current_content:
                chapters.append({
                    'title': current_chapter,
                    'content': '\n'.join(current_content)
                })
            elif not chapters and all_text:
                # If no chapters detected, split by pages
                pages_per_chapter = max(5, len(pdf.pages) // 10)  # ~10 chapters per document
                for i in range(0, len(all_text), pages_per_chapter):
                    chapter_text = '\n'.join(all_text[i:i + pages_per_chapter])
                    chapters.append({
                        'title': f'Chapter {len(chapters) + 1}',
                        'content': chapter_text
                    })
    
    except Exception as e:
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                all_text = []
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
                
                if all_text:
                    # Split into chapters by page count
                    pages_per_chapter = max(5, len(pdf_reader.pages) // 10)
                    full_text = '\n'.join(all_text)
                    
                    for i in range(0, len(all_text), pages_per_chapter):
                        chapter_text = '\n'.join(all_text[i:i + pages_per_chapter])
                        chapters.append({
                            'title': f'Chapter {len(chapters) + 1}',
                            'content': chapter_text
                        })
        except Exception as e2:
            raise Exception(f"Failed to parse PDF: {str(e2)}")
    
    return chapters

