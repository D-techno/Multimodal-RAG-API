import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from app.config import CLIP_MODEL_NAME, DEVICE, NORMALIZE_EMBEDDINGS
from app.utils import setup_logger

logger = setup_logger()


class CLIPEmbeddingModel:
    def __init__(self):
        logger.info("Loading CLIP model...")
        self.device = DEVICE
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model.eval()
        logger.info(f"CLIP model loaded successfully on {self.device}")

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        if NORMALIZE_EMBEDDINGS:
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding / (norm + 1e-10)
        return embedding

    def _extract_tensor(self, output):
        """
        Handles different transformer versions safely.
        """
        if isinstance(output, torch.Tensor):
            return output
        elif hasattr(output, "image_embeds"):
            return output.image_embeds
        elif hasattr(output, "text_embeds"):
            return output.text_embeds
        elif hasattr(output, "pooler_output"):
            return output.pooler_output
        else:
            raise ValueError("Unexpected output format from CLIP model")

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.no_grad():
                output = self.model.get_image_features(pixel_values=pixel_values)

            tensor = self._extract_tensor(output)

            embedding = tensor.detach().cpu().numpy()
            embedding = self._normalize(embedding)

            logger.info(f"Generated image embedding for: {image_path}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            raise

    def get_text_embedding(self, text: str) -> np.ndarray:
        try:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                output = self.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            tensor = self._extract_tensor(output)

            embedding = tensor.detach().cpu().numpy()
            embedding = self._normalize(embedding)

            logger.info("Generated text embedding")
            return embedding

        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise