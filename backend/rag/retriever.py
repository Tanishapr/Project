from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


VECTORSTORE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "embeddings"
    / "pm_kisan"
)


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def search_scheme(question: str, k: int = 6):
    vectorstore = get_vectorstore()

    documents = vectorstore.similarity_search(
        question,
        k=k,
    )

    results = []

    for document in documents:
        results.append(
            {
                "content": document.page_content,
            }
        )

    return results