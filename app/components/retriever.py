"""
retriever.py — LCEL RAG chain using Groq (ChatGroq = chat model).

Since we're back on ChatGroq (a proper ChatModel), we use ChatPromptTemplate
which is the correct prompt type for chat-based LLMs.
"""

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful medical information assistant with access to a \
curated knowledge base of clinical documents.

Answer using ONLY the information in the context below.
If the context does not contain enough information, say:
"I could not find a confident answer in the provided documents. \
Please consult a qualified healthcare professional."

Never fabricate drug dosages, lab reference ranges, or clinical facts.

Context:
{context}"""

HUMAN_PROMPT = """{question}

⚠️ This information is for educational purposes only. \
Always consult a licensed healthcare professional before making medical decisions."""


def _format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


@lru_cache(maxsize=1)
def get_qa_chain():
    try:
        logger.info("Loading vector store...")
        db = load_vector_store()
        if db is None:
            raise CustomException(
                "Vector store not found. "
                "Run: python -m app.components.data_loader"
            )

        logger.info("Loading LLM...")
        llm = load_llm()
        if llm is None:
            raise CustomException("LLM failed to load. Check GROQ_API_KEY in .env.")

        retriever = db.as_retriever(search_kwargs={"k": 3})

        # ChatPromptTemplate — correct for ChatGroq (chat model)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_PROMPT),
        ])

        chain = (
            {
                "context":  retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("QA chain built successfully.")
        return chain

    except Exception as e:
        error_message = CustomException("Failed to build QA chain", e)
        logger.error(str(error_message))
        return None


def create_qa_chain():
    return get_qa_chain()
