class Retriever:
    def __init__(self, collection, top_k=5):
        self.collection = collection
        self.top_k = top_k

    def retrieve(self, query: str):
        results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k
        )

        retrieved_chunks = []
        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0]
        ):
            retrieved_chunks.append({
                "text": doc,
                "metadata": meta
            })

        return retrieved_chunks
