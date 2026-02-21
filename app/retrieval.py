from app.embeddings import CLIPEmbeddingModel
from app.vector_store import FAISSVectorStore
from app.config import TOP_K
from app.utils import setup_logger

logger = setup_logger()


class RetrievalPipeline:
    def __init__(self):
        self.embedding_model = CLIPEmbeddingModel()
        self.vector_store = FAISSVectorStore()
        self.documents = []

    def add_text_document(self, text: str):
        embedding = self.embedding_model.get_text_embedding(text)
        self.vector_store.add_embeddings(embedding)
        self.documents.append(text)
        logger.info("Document added to vector store.")

    def retrieve(self, query: str, top_k: int = TOP_K):
        if len(self.documents) == 0:
            logger.warning("No documents available for retrieval.")
            return []

        query_embedding = self.embedding_model.get_text_embedding(query)
        distances, indices = self.vector_store.search(query_embedding, top_k)

        retrieved_docs = []

        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                retrieved_docs.append(self.documents[idx])

        return retrieved_docs