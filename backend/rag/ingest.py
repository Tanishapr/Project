from pathlib import Path
from pypdf import PdfReader

PDF_PATH = Path("../../data/schemes/raw/pm_kisan_guidelines.pdf")
OUTPUT_PATH = Path("../../data/schemes/cleaned/pm_kisan.txt")


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            f"\n--- PAGE {page_number} ---\n{text}"
        )

    return "\n".join(pages)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    text = extract_pdf_text(PDF_PATH)

    OUTPUT_PATH.write_text(text, encoding="utf-8")

    print(f"Extracted {len(text)} characters")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()