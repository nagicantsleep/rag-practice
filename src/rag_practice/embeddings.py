from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()
