from pypdf import PdfReader
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_pdf(path: str, company: str, document: str):
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        text = clean_text(text)

        pages.append({
            "text": text,
            "metadata": {
                "company": company,
                "document": document,
                "page": i + 1
            }
        })
    return pages


# Debug / manual testing only
if __name__ == "__main__":
    apple_pages = load_pdf("data/10-Q4-2024-As-Filed.pdf", "Apple", "Apple 10-K")
    print("Apple sample:", apple_pages[0]["metadata"])

    tesla_pages = load_pdf("data/tsla-20231231-gen.pdf", "Tesla", "Tesla 10-K")
    print("Tesla sample:", tesla_pages[0]["metadata"])
