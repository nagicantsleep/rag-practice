"""Pinned pretrained ColSmol page-image retrieval control for M08.6."""

from __future__ import annotations

from time import perf_counter_ns
from typing import Any

from rag_practice.visual_document.retrieval import PageRetrievalResult, VisualDocumentIndex

MODEL_NAME = "vidore/colSmol-256M"
MODEL_REVISION = "a59110fdf114638b8018e6c9a018907e12f14855"
BASE_MODEL_NAME = "vidore/ColSmolVLM-Instruct-256M-base"
BASE_MODEL_REVISION = "8a0cee6d479200dbce31dbfef88c66175d89cddc"


class ColSmolPageRetriever:
    """Exhaustive text-to-page retrieval using pinned ColSmol checkpoints.

    The retriever receives only the query text and rendered page pixels. Frozen
    OCR text, titles, document ids, qrels, expected answers, and deterministic
    visual features are not used for ranking.

    The ColSmol adapter repository references an upstream base repository whose
    current default revision no longer carries the full model weights. Loading
    the verified historical full-weight base revision explicitly makes the
    adapter composition reproducible rather than depending on mutable ``main``.
    """

    name = "pinned-colsmol-page-image"

    def __init__(self, index: VisualDocumentIndex) -> None:
        import torch
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        from peft import PeftModel

        self.index = index
        self.device = torch.device("cpu")
        load_started = perf_counter_ns()
        base_model = ColIdefics3.from_pretrained(
            BASE_MODEL_NAME,
            revision=BASE_MODEL_REVISION,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            MODEL_NAME,
            revision=MODEL_REVISION,
        ).to(self.device).eval()
        self.processor = ColIdefics3Processor.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
        )
        self.model_load_ms = (perf_counter_ns() - load_started) / 1_000_000

        self.page_ids = [asset.id for asset in index.assets]
        pages = [index.images[page_id] for page_id in self.page_ids]
        build_started = perf_counter_ns()
        batch_images = self.processor.process_images(pages).to(self.device)
        with torch.inference_mode():
            self.image_embeddings = self.model(**batch_images).to("cpu")
        self.index_build_ms = (perf_counter_ns() - build_started) / 1_000_000
        self.embedding_shape = list(self.image_embeddings.shape)
        self.embedding_bytes = int(
            self.image_embeddings.numel() * self.image_embeddings.element_size()
        )

    def retrieve(self, query: str, *, k: int = 3) -> PageRetrievalResult:
        import torch

        batch_queries = self.processor.process_queries([query]).to(self.device)
        with torch.inference_mode():
            query_embeddings = self.model(**batch_queries).to("cpu")
            scores = self.processor.score_multi_vector(
                query_embeddings,
                self.image_embeddings,
            )
        row: Any = scores[0]
        scored = [
            (page_id, float(row[position].item()))
            for position, page_id in enumerate(self.page_ids)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return PageRetrievalResult(
            ranking=tuple(scored[:k]),
            evidence_modalities=("page_image",),
            visual_candidates_scored=len(self.page_ids),
            region_by_page={},
        )
