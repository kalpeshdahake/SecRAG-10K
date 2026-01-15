import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_dir="vector_db"):
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_dir,
                anonymized_telemetry=False
            )
        )

    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(name=name)

    def add_chunks(self, collection, chunks, embeddings):
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]

            chunk_id = (
                f"{meta.get('company','unknown')}_"
                f"{meta.get('document','doc')}_"
                f"p{meta.get('page','x')}_"
                f"c{i}"
            )

            meta = meta.copy()
            meta["chunk_id"] = chunk_id

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(meta)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
