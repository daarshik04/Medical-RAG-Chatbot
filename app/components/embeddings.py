from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and cache the embedding model once per process.

    FIX (minor): the original had no caching — every call to load_vector_store()
    or save_vector_store() reloaded the ~80 MB sentence-transformer model from
    disk. @lru_cache ensures it is downloaded and loaded exactly once.
    """
    try:
        logger.info("Initialising HuggingFace embedding model")

        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("HuggingFace embedding model loaded successfully.")
        return model

    except Exception as e:
        error_message = CustomException("Error while loading embedding model", e)
        logger.error(str(error_message))
        raise error_message
