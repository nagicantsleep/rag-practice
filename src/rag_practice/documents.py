from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Document:
    text: str
    source: str


def load_file(path: Path) -> Document:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return Document(text=text.strip(), source=str(path))


def load_directory(directory: Path) -> list[Document]:
    supported = {".pdf", ".txt", ".md"}
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    )
    return [load_file(path) for path in paths]
