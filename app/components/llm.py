"""
llm.py — LLM via Groq (free, fast, reliable).

WHY GROQ INSTEAD OF HUGGINGFACE:
  As of July 2025, HuggingFace Serverless Inference API dropped text-generation
  support for large LLMs entirely. It now routes requests to third-party
  providers (Novita, Together AI, etc.) via chat_completion only — but each
  provider has different model support, causing the two errors you saw:

    Error 1 (via one provider):
      "Mistral-7B is not a chat model" → expected text-generation format

    Error 2 (via Novita provider):
      "not supported for task text-generation ... Supported task: conversational"

  There is no stable way to use Mistral-7B on HF free tier anymore.

  Groq:
    ✅ Free tier: https://console.groq.com (no credit card needed)
    ✅ llama-3.1-8b-instant is significantly better than Mistral-7B
    ✅ No provider routing — direct, consistent API
    ✅ Very fast (Groq runs on custom LPU hardware)
    ✅ langchain-groq is already in the repo's original requirements.txt
"""

from functools import lru_cache
from langchain_groq import ChatGroq
from app.config.config import GROQ_API_KEY
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_llm():
    try:
        if not GROQ_API_KEY:
            raise CustomException(
                "GROQ_API_KEY is not set.\n"
                "  1. Sign up free at https://console.groq.com\n"
                "  2. Create an API key\n"
                "  3. Add GROQ_API_KEY=your_key_here to your .env file"
            )

        logger.info("Loading LLM via Groq (llama-3.1-8b-instant)...")

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1024,
        )

        logger.info("Groq LLM loaded successfully.")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load Groq LLM", e)
        logger.error(str(error_message))
        return None
