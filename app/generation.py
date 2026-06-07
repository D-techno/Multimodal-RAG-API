from app.utils import setup_logger

logger = setup_logger()


class LLMGenerator:
    """
    Lightweight deployment version.
    No TinyLlama loading.
    """

    def __init__(self):
        logger.info("Deployment Demo Mode Enabled")

    def build_prompt(self, context: str, question: str) -> str:
        return f"""
Context:
{context}

Question:
{question}
"""

    def generate(self, context: str, question: str) -> str:
        return f"Retrieved Context: {context}"