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
        collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[str(i) for i in range(len(chunks))],
            embeddings=embeddings
        )

    def persist(self):
        self.client.persist()
