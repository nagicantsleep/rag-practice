from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    document_id: str

@dataclass(frozen=True)
class StructuredDocument:
    id: str
    text: str
    collection: str
    triples: tuple[Triple, ...]

    @classmethod
    def from_mapping(cls, row: dict) -> "StructuredDocument":
        triples = tuple(Triple(s, r, o, row["id"]) for s, r, o in row.get("triples", []))
        return cls(id=row["id"], text=row["text"], collection=row.get("collection", "default"), triples=triples)

@dataclass(frozen=True)
class MemoryEvent:
    id: str
    memory_key: str
    sequence: int
    text: str
    entity: str
    relation: str
    value: str

    @classmethod
    def from_mapping(cls, row: dict) -> "MemoryEvent":
        return cls(
            id=row["id"], memory_key=row["memory_key"], sequence=int(row["sequence"]),
            text=row["text"], entity=row["entity"], relation=row["relation"], value=row["value"]
        )
