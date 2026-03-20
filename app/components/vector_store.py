import os
from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)


def load_vector_store():
    try:
        embedding_model = get_embedding_model()

        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading existing vector store...")
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.warning(
                f"No vector store found at '{DB_FAISS_PATH}'. "
                "Run 'python -m app.components.data_loader' to build it first."
            )
            return None

    except Exception as e:
        error_message = CustomException("Failed to load vector store", e)
        logger.error(str(error_message))
        return None


def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No chunks provided to save_vector_store()")

        logger.info(f"Building new FAISS vector store from {len(text_chunks)} chunks...")

        embedding_model = get_embedding_model()
        db = FAISS.from_documents(text_chunks, embedding_model)

        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        db.save_local(DB_FAISS_PATH)

        logger.info(f"Vector store saved to '{DB_FAISS_PATH}' ({db.index.ntotal} vectors).")
        return db

    except Exception as e:
        error_message = CustomException("Failed to create new vector store", e)
        logger.error(str(error_message))
        return None
