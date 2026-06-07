import numpy as np
from app.utils import setup_logger

logger = setup_logger()


class CLIPEmbeddingModel:
    """
    Lightweight deployment version.
    No CLIP model loading.
    """

    def __init__(self):
        logger.info("Deployment Demo Embedding Mode Enabled")

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        np.random.seed(abs(hash(image_path)) % (2**32))
        return np.random.rand(1, 512).astype(np.float32)

    def get_text_embedding(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1, 512).astype(np.float32)