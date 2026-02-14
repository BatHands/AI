from typing import Iterable

from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from src.config import HF_API_KEY, HF_EMBED_MODEL_ID, HF_LLM_MODEL_ID, MILVUS_URI


def _format_docs(docs: Iterable[LCDocument]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_vectorstore(splits: list[LCDocument]) -> Milvus:
    """Create a Milvus vector store from document splits."""
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL_ID)
    return Milvus.from_documents(
        splits,
        embeddings,
        connection_args={"uri": MILVUS_URI},
        drop_old=True,
    )


def build_rag_chain(vectorstore: Milvus):
    """Assemble the RAG chain from retriever, prompt, LLM, and output parser."""
    retriever = vectorstore.as_retriever()

    llm = HuggingFaceEndpoint(
        repo_id=HF_LLM_MODEL_ID,
        huggingfacehub_api_token=HF_API_KEY,
        task="text-generation",
    )
    chat_model = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You're a helpful assistant."),
        HumanMessagePromptTemplate.from_template(
            "Context information is below.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            "Query: {question}\n"
            "Answer:\n"
        ),
    ])

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )
