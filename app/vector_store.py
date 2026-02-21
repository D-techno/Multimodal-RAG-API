import faiss
import numpy as np
import os
from app.config import EMBEDDING_DIM
from app.utils import setup_logger

logger = setup_logger()


class FAISSVectorStore:
    def __init__(self, index_path: str = "models/faiss_index.bin"):
        self.index_path = index_path
        self.index = None
        self._initialize_index()

    def _initialize_index(self):
        """
        Initialize FAISS index.
        Using L2 distance (can switch to cosine later).
        """
        if os.path.exists(self.index_path):
            logger.info("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)
        else:
            logger.info("Creating new FAISS index...")
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Add embeddings to index.
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        logger.info(f"Added {embeddings.shape[0]} embeddings to index")

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        """
        Search similar vectors.
        Returns distances and indices.
        """
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices

    def save_index(self):
        """
        Save index to disk.
        """
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        logger.info(f"FAISS index saved to {self.index_path}")

    def get_index_size(self):
        return self.index.ntotal