"""Visual-document/page-image retrieval controls for M08.6."""

from .colsmol import ColSmolPageRetriever, MODEL_NAME, MODEL_REVISION
from .retrieval import (
    PageAsset,
    PageRetrievalResult,
    VisualDocumentIndex,
    VisualRequest,
    decode_page_payloads,
)

__all__ = [
    "ColSmolPageRetriever",
    "MODEL_NAME",
    "MODEL_REVISION",
    "PageAsset",
    "PageRetrievalResult",
    "VisualDocumentIndex",
    "VisualRequest",
    "decode_page_payloads",
]
