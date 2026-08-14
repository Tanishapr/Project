from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INPUT_PATH = Path("../../data/schemes/cleaned/pm_kisan_chunks.txt")
VECTORSTORE_PATH = Path("../../data/embeddings/pm_kisan")


def load_chunks():
    text = INPUT_PATH.read_text(encoding="utf-8")

    chunks = text.split("\n\n--- CHUNK ")

    cleaned_chunks = []

    for chunk in chunks:
        chunk = chunk.strip()

        if chunk:
            if not chunk.startswith("--- CHUNK"):
                chunk = "--- CHUNK " + chunk

            cleaned_chunks.append(chunk)

    return cleaned_chunks


def main():
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        chunks,
        embedding=embeddings,
    )

    VECTORSTORE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore.save_local(str(VECTORSTORE_PATH))

    print(f"Vector database saved to: {VECTORSTORE_PATH}")


if __name__ == "__main__":
    main()