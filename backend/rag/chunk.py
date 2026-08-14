from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_PATH = Path("../../data/schemes/cleaned/pm_kisan.txt")
OUTPUT_PATH = Path("../../data/schemes/cleaned/pm_kisan_chunks.txt")


def main():
    text = INPUT_PATH.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n--- PAGE", "\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_text(text)

    formatted_chunks = []

    for i, chunk in enumerate(chunks):
        formatted_chunks.append(
            f"--- CHUNK {i + 1} ---\n{chunk}"
        )

    OUTPUT_PATH.write_text(
        "\n\n".join(formatted_chunks),
        encoding="utf-8",
    )

    print(f"Created {len(chunks)} chunks")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()