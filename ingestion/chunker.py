def chunk_text(pages, chunk_size=900, overlap=150):
    """
    Splits page-level text into overlapping chunks while preserving metadata.

    Args:
        pages (list): Output of section_parser.assign_items()
        chunk_size (int): Number of characters per chunk
        overlap (int): Overlap between consecutive chunks

    Returns:
        list: Chunks with text + metadata
    """

    chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "metadata": metadata.copy()
            })

            start += chunk_size - overlap

    return chunks


# Debug / Validation
if __name__ == "__main__":
    from pdf_loader import load_pdf
    from section_parser import assign_items

    pages = load_pdf("data/10-Q4-2024-As-Filed.pdf", "Apple", "Apple 10-K")
    pages = assign_items(pages)

    chunks = chunk_text(pages)

    print("Total chunks:", len(chunks))
    print("Sample chunk metadata:", chunks[0]["metadata"])
    print("Sample chunk text (first 300 chars):")
    print(chunks[0]["text"][:300])
