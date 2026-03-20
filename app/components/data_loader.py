import os
from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.vector_store import save_vector_store
from app.config.config import DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


def process_and_store_pdfs():
    try:
        logger.info("Building the vector store from PDFs...")

        documents = load_pdf_files()
        if not documents:
            raise CustomException("No documents loaded — check your data/ folder.")

        text_chunks = create_text_chunks(documents)
        if not text_chunks:
            raise CustomException("Chunking produced no output.")

        save_vector_store(text_chunks)

        logger.info("Vector store created successfully.")

    except Exception as e:
        error_message = CustomException("Failed to create vector store", e)
        logger.error(str(error_message))


if __name__ == "__main__":
    process_and_store_pdfs()
