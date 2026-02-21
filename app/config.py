import torch

# -------- CLIP --------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
NORMALIZE_EMBEDDINGS = True

# -------- Device --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Retrieval --------
TOP_K = 3

# -------- LLM (for later step) --------
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.7