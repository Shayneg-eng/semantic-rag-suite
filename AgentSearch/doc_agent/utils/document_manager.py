"""
Document Manager: Loads documents and provides access
"""

import os
import sys
import re
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import DOCUMENTS_FOLDER, FOCUS_DOCS_FILE


class DocumentManager:
    def __init__(self):
        self.documents = {}  # doc_id -> full text
        self.metadata = {}   # doc_id -> metadata
        
    def load_documents(self):
        """Load all .txt files from the documents folder"""
        print(f"Loading documents from {DOCUMENTS_FOLDER}...")
        
        if not os.path.exists(DOCUMENTS_FOLDER):
            raise FileNotFoundError(f"Documents folder not found: {DOCUMENTS_FOLDER}")
        
        for filename in os.listdir(DOCUMENTS_FOLDER):
            if filename.endswith('.txt'):
                filepath = os.path.join(DOCUMENTS_FOLDER, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc_id = filename
                    self.documents[doc_id] = content
                    
                    # Calculate metadata
                    self.metadata[doc_id] = {
                        'word_count': len(content.split()),
                        'line_count': len(content.split('\n')),
                        'char_count': len(content),
                        'filename': filename
                    }
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.documents)} documents")
        return len(self.documents)
    
    def chunk_document(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split a document into chunks by paragraphs for easier processing
        """
        chunks = []
        
        # Split by paragraphs (double newlines)
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if para.strip():
                chunks.append(para.strip())
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_objects.append({
                'chunk_id': i,
                'text': chunk_text,
                'doc_id': doc_id
            })
        
        return chunk_objects
    
    def get_document(self, doc_id: str) -> str:
        """Get full document text"""
        return self.documents.get(doc_id)
    
    def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get document metadata"""
        return self.metadata.get(doc_id)
    
    def get_all_doc_ids(self) -> List[str]:
        """Get list of all document IDs"""
        return list(self.documents.keys())
    
    def get_chunk(self, doc_id: str, chunk_id: int) -> Dict[str, Any]:
        """Get a specific chunk"""
        return None
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks from all documents"""
        return []
    
    def initialize(self):
        """Initialize the document manager: load documents"""
        self.load_documents()
        print(f"Document manager initialized with {len(self.documents)} documents")
    
    def _load_cached_embeddings(self):
        """Not used - embeddings removed"""
        pass
    
    def ensure_embedded(self, doc_id: str):
        """Not used - embeddings removed"""
        pass
    
    def _save_embeddings(self):
        """Not used - embeddings removed"""
        pass


# Singleton instance
_doc_manager = None

def get_document_manager() -> DocumentManager:
    """Get or create the document manager singleton"""
    global _doc_manager
    if _doc_manager is None:
        _doc_manager = DocumentManager()
    return _doc_manager
