"""
Focus Manager: Maintains the "documents of interest" list with persistence
"""

import json
import os
import sys
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import FOCUS_DOCS_FILE


class FocusManager:
    def __init__(self):
        self.focused_docs = []  # List of document IDs
        self.reasons = {}       # doc_id -> reason for focusing
        self.load()
    
    def load(self):
        """Load focus list from disk"""
        if os.path.exists(FOCUS_DOCS_FILE):
            try:
                with open(FOCUS_DOCS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.focused_docs = data.get('docs', [])
                    self.reasons = data.get('reasons', {})
                print(f"Loaded {len(self.focused_docs)} focused documents")
            except Exception as e:
                print(f"Error loading focus list: {e}")
                self.focused_docs = []
                self.reasons = {}
        else:
            print("No existing focus list found, starting fresh")
    
    def save(self):
        """Save focus list to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(FOCUS_DOCS_FILE), exist_ok=True)
            
            data = {
                'docs': self.focused_docs,
                'reasons': self.reasons
            }
            with open(FOCUS_DOCS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving focus list: {e}")
    
    def add(self, doc_ids: List[str], reason: Optional[str] = None):
        """Add documents to focus list"""
        added = []
        for doc_id in doc_ids:
            if doc_id not in self.focused_docs:
                self.focused_docs.append(doc_id)
                if reason:
                    self.reasons[doc_id] = reason
                added.append(doc_id)
        
        self.save()
        return added
    
    def remove(self, doc_ids: List[str]):
        """Remove specific documents from focus list"""
        removed = []
        for doc_id in doc_ids:
            if doc_id in self.focused_docs:
                self.focused_docs.remove(doc_id)
                if doc_id in self.reasons:
                    del self.reasons[doc_id]
                removed.append(doc_id)
        
        self.save()
        return removed
    
    def clear(self):
        """Clear all focused documents"""
        count = len(self.focused_docs)
        self.focused_docs = []
        self.reasons = {}
        self.save()
        return count
    
    def get_focused_docs(self) -> List[str]:
        """Get list of focused document IDs"""
        return self.focused_docs.copy()
    
    def is_focused(self, doc_id: str) -> bool:
        """Check if a document is in focus"""
        return doc_id in self.focused_docs
    
    def get_reason(self, doc_id: str) -> Optional[str]:
        """Get the reason for focusing on a document"""
        return self.reasons.get(doc_id)
    
    def count(self) -> int:
        """Get count of focused documents"""
        return len(self.focused_docs)


# Singleton instance
_focus_manager = None

def get_focus_manager() -> FocusManager:
    """Get or create the focus manager singleton"""
    global _focus_manager
    if _focus_manager is None:
        _focus_manager = FocusManager()
    return _focus_manager
