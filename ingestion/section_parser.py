import re
from ingestion.pdf_loader import load_pdf

ITEM_PATTERN = re.compile(r"(ITEM\s+\d+[A-Z]?)\.?", re.IGNORECASE)

def normalize_item(item: str) -> str:
    return item.replace(".", "").title()

def assign_items(pages):
    current_item = None

    for page in pages:
        match = ITEM_PATTERN.search(page["text"])
        if match:
            current_item = normalize_item(match.group(1))

        page["metadata"]["item"] = current_item or "Unknown"

    return pages


# Debug test
if __name__ == "__main__":
    pages = load_pdf("data/10-Q4-2024-As-Filed.pdf", "Apple", "Apple 10-K")
    pages = assign_items(pages)
    print(pages[50]["metadata"])
