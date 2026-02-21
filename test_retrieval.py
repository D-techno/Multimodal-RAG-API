from app.retrieval import RetrievalPipeline

pipeline = RetrievalPipeline()

# Add sample docs
pipeline.add_text_document("Cisco networking router device")
pipeline.add_text_document("Enterprise data center switch")
pipeline.add_text_document("Cloud security firewall")

# Query
distances, indices = pipeline.retrieve("network hardware equipment")

print("Distances:", distances)
print("Indices:", indices)