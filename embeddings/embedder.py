from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list):
        """
        Generate embeddings for a list of texts.
        """
        return self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

# Example usage 
if __name__ == "__main__":
    embedder = Embedder()
    vec = embedder.embed_texts(["Apple total revenue increased"])
    print(vec.shape)