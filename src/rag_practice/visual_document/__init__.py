"""Visual-document/page-image retrieval controls for M08.6."""

from .colsmol import (
    BASE_MODEL_NAME,
    BASE_MODEL_REVISION,
    MODEL_NAME,
    MODEL_REVISION,
    ColSmolPageRetriever,
)
from .retrieval import (
    PageAsset,
    PageRetrievalResult,
    VisualDocumentIndex,
    VisualRequest,
    decode_page_payloads,
)

__all__ = [
    "BASE_MODEL_NAME",
    "BASE_MODEL_REVISION",
    "ColSmolPageRetriever",
    "MODEL_NAME",
    "MODEL_REVISION",
    "PageAsset",
    "PageRetrievalResult",
    "VisualDocumentIndex",
    "VisualRequest",
    "decode_page_payloads",
]
