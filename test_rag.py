from app.retrieval import RetrievalPipeline
from app.generation import LLMGenerator

# Initialize components
retrieval = RetrievalPipeline()
llm = LLMGenerator()

# Add sample documents (you can modify these)
retrieval.add_text_document("Cisco manufactures routers.")
retrieval.add_text_document("Cisco builds enterprise network switches.")
retrieval.add_text_document("Cisco develops cybersecurity firewalls.")

# User query
query = "What does Cisco produce?"

print("\n==============================")
print("User Question:")
print(query)
print("==============================")

# Retrieve context
docs = retrieval.retrieve(query)

print("\n--- Retrieved Documents ---")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc}")

# Combine context
context = "\n".join(docs)

# Generate final answer
response = llm.generate(context=context, question=query)

print("\n--- Final Answer ---")
print(response)
print("==============================\n")