from app.embeddings import CLIPEmbeddingModel

model = CLIPEmbeddingModel()

text_embedding = model.get_text_embedding("Cisco networking hardware")
print("Text embedding shape:", text_embedding.shape)

image_embedding = model.get_image_embedding("data/images/WhatsApp Image 2025-04-30 at 16.02.50_dcec1995.jpg")
print("Image embedding shape:", image_embedding.shape)