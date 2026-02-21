from app.embeddings import CLIPEmbeddingModel
from app.vector_store import FAISSVectorStore

# Initialize
embedding_model = CLIPEmbeddingModel()
vector_store = FAISSVectorStore()

# Generate embeddings
text1 = embedding_model.get_text_embedding("Cisco router device")
text2 = embedding_model.get_text_embedding("Network switch hardware")

# Add embeddings
vector_store.add_embeddings(text1)
vector_store.add_embeddings(text2)

# Save index
vector_store.save_index()

print("Index size:", vector_store.get_index_size())

# Search test
query = embedding_model.get_text_embedding("Networking equipment")
distances, indices = vector_store.search(query, top_k=2)

print("Search distances:", distances)
print("Search indices:", indices)