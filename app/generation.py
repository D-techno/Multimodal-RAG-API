import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import LLM_MODEL_NAME, DEVICE, MAX_NEW_TOKENS
from app.utils import setup_logger

logger = setup_logger()


class LLMGenerator:
    """
    LLM Generator for causal models like TinyLlama.
    Clean, stable, CPU-safe version.
    """

    def __init__(self):
        logger.info("Loading LLM model...")
        self.device = DEVICE

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME
        ).to(self.device)

        self.model.eval()
        logger.info(f"LLM loaded successfully on {self.device}")

    def build_prompt(self, context: str, question: str) -> str:
        """
        Structured RAG prompt.
        """
        return f"""You are an AI assistant.
Answer using ONLY the provided context.
Be concise.

Context:
{context}

Question:
{question}

Answer:"""

    def generate(self, context: str, question: str) -> str:
        """
        Generate response from LLM with clean output handling.
        """
        try:
            prompt = self.build_prompt(context, question)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1,
                    use_cache=True)



            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # --- Clean output ---
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()

            # Stop at first newline
            response = response.split("\n")[0].strip()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise