from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from .chunking import Chunk
from .embeddings import Embedder


COLLECTION_NAME = "rag_practice"


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    chunk_index: int
    score: float


class VectorStore:
    def __init__(
        self,
        embedder: Embedder,
        path: Path = Path(".qdrant"),
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.embedder = embedder
        self.client = QdrantClient(path=str(path))
        self.collection_name = collection_name

    def recreate_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def index(self, chunks: list[Chunk]) -> None:
        self.recreate_collection()
        if not chunks:
            return

        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        points = [
            models.PointStruct(
                id=i,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.index,
                },
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    text=str(payload.get("text", "")),
                    source=str(payload.get("source", "unknown")),
                    chunk_index=int(payload.get("chunk_index", -1)),
                    score=float(point.score),
                )
            )
        return results
