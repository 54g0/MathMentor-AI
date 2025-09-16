import os
import threading
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

_LOCK = threading.Lock()

class VectorDBUpdater:
    def __init__(self, vector_db_root: str = "/home/egg/Documents/agentic_rag_MT/vector_db"):
        self.vector_db_root = vector_db_root
        self.vector_store_dir = os.path.join(vector_db_root, "vector_store")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._vector_store: Optional[FAISS] = None

    def _ensure_loaded(self):
        if self._vector_store is None:
            if not os.path.isdir(self.vector_store_dir):
                raise RuntimeError(f"Vector store dir not found: {self.vector_store_dir}")
            self._vector_store = FAISS.load_local(self.vector_store_dir, self.embeddings, allow_dangerous_deserialization=True)

    def add_qa_pair(self, question: str, answer: str) -> bool:
        """Add a new Q/A pair as a text block and persist FAISS index.
        Returns True if success else False.
        """
        text = f"Q: {question}\nA: {answer}".strip()
        if not text:
            return False
        with _LOCK:
            self._ensure_loaded()
            try:
                self._vector_store.add_texts([text])
                # Persist updated index
                self._vector_store.save_local(self.vector_store_dir)
                return True
            except Exception as e:
                print(f"[VectorDBUpdater] Failed to add pair: {e}")
                return False

# Singleton style convenience
_updater: Optional[VectorDBUpdater] = None

def get_updater() -> VectorDBUpdater:
    global _updater
    if _updater is None:
        _updater = VectorDBUpdater()
    return _updater
