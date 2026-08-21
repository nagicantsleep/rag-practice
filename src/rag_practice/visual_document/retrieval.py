"""Transparent OCR, page-raster, and fused retrieval for M08.6."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import BytesIO
import base64
import gzip
import json
import re
from pathlib import Path

from PIL import Image

from rag_practice.ir.bm25 import BM25Index
from rag_practice.sources.base import SourceHit, SourceRecord

_COLOR_RGB = {
    "red": (220, 40, 40),
    "green": (40, 170, 80),
    "blue": (40, 100, 220),
    "orange": (235, 150, 40),
    "yellow": (245, 215, 50),
    "magenta": (220, 40, 180),
}
_VISUAL_COLORS = tuple(_COLOR_RGB)
_DOC_NAMES = {
    "alpha operations review": "alpha_operations_review",
    "beta shipping policy": "beta_shipping_policy",
    "beta capacity report": "beta_capacity_report",
    "gamma onboarding guide": "gamma_onboarding_guide",
    "gamma invoice": "gamma_invoice",
}


@dataclass(frozen=True)
class PageAsset:
    id: str
    doc_id: str
    page: int
    title: str
    file_name: str
    ocr_text: str
    facts: dict[str, str]

    @property
    def locator(self) -> str:
        return f"page://benchmark/{self.doc_id}/{self.page}"

    def region_locator(self, region: str) -> str:
        return f"{self.locator}#{region}"

    def as_source_record(self) -> SourceRecord:
        return SourceRecord(
            id=self.id,
            source_type="page_image",
            locator=self.locator,
            title=self.title,
            content=self.ocr_text,
            metadata={"doc_id": self.doc_id, "page": self.page, "file": self.file_name},
        )


@dataclass(frozen=True)
class VisualRequest:
    color: str | None = None
    position: str | None = None
    shape: str | None = None
    taller_than: tuple[str, str] | None = None

    @property
    def has_visual_constraint(self) -> bool:
        return any((self.color, self.position, self.shape, self.taller_than))


@dataclass(frozen=True)
class PageRetrievalResult:
    ranking: tuple[tuple[str, float], ...]
    evidence_modalities: tuple[str, ...]
    visual_candidates_scored: int = 0
    region_by_page: dict[str, str] | None = None

    def region_for(self, page_id: str) -> str | None:
        return (self.region_by_page or {}).get(page_id)


def decode_page_payloads(path: Path) -> dict[str, bytes]:
    """Decode frozen repository-friendly XPM payloads without adding semantics."""
    bundle = json.loads(path.read_text())
    if bundle.get("encoding") != "gzip+base64" or bundle.get("format") != "xpm":
        raise ValueError("expected gzip+base64 encoded XPM page bundle")
    images = bundle.get("images")
    if not isinstance(images, dict):
        raise ValueError("page bundle must contain an images mapping")
    return {
        str(name): gzip.decompress(base64.b64decode(str(encoded)))
        for name, encoded in images.items()
    }


class VisualDocumentIndex:
    """Controlled page index that keeps OCR and raster evidence attributable."""

    name = "controlled-visual-document-pages"

    def __init__(self, benchmark_dir: Path) -> None:
        self.benchmark_dir = benchmark_dir
        self.assets = self._load_assets(benchmark_dir / "records.jsonl")
        self.by_id = {asset.id: asset for asset in self.assets}
        self.page_payloads = decode_page_payloads(benchmark_dir / "images.json")
        self.images: dict[str, Image.Image] = {}
        for asset in self.assets:
            raw = self.page_payloads[asset.file_name]
            with Image.open(BytesIO(raw)) as image:
                self.images[asset.id] = image.convert("RGB")
        if len(self.images) != len(self.assets):
            raise ValueError("every page record must have one raster payload")
        self.ocr_documents = {
            asset.id: f"{asset.title} {asset.ocr_text}" for asset in self.assets
        }
        self.ocr_index = BM25Index(self.ocr_documents)
        self.ocr_chars_indexed = sum(len(text) for text in self.ocr_documents.values())
        self.raster_source_bytes = sum(len(raw) for raw in self.page_payloads.values())
        self.raster_rgb_bytes = sum(image.width * image.height * 3 for image in self.images.values())

    @staticmethod
    def _load_assets(path: Path) -> tuple[PageAsset, ...]:
        assets: list[PageAsset] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            assets.append(
                PageAsset(
                    id=item["id"],
                    doc_id=item["doc_id"],
                    page=int(item["page"]),
                    title=item["title"],
                    file_name=item["file"],
                    ocr_text=item["ocr_text"],
                    facts={str(key): str(value) for key, value in item.get("facts", {}).items()},
                )
            )
        return tuple(assets)

    @staticmethod
    def parse_visual_request(query: str) -> VisualRequest:
        lowered = query.lower()
        color = next(
            (name for name in _VISUAL_COLORS if re.search(rf"\b{name}\b", lowered)),
            None,
        )
        position = None
        for candidate in ("upper-right", "upper-left", "lower-right", "lower-left"):
            if candidate in lowered or candidate.replace("-", " ") in lowered:
                position = candidate
                break
        if position is None and ("on the left" in lowered or "sidebar on the left" in lowered):
            position = "left"
        if position is None and "on the right" in lowered:
            position = "right"
        if "sidebar" in lowered:
            shape = "sidebar"
        elif "stamp" in lowered:
            shape = "stamp"
        elif "cell" in lowered:
            shape = "cell"
        else:
            shape = None
        comparison = re.search(
            r"(red|green|blue|orange|yellow|magenta)\s+bar\s+taller\s+than\s+"
            r"(?:an?\s+)?(red|green|blue|orange|yellow|magenta)\s+bar",
            lowered,
        )
        taller_than = (comparison.group(1), comparison.group(2)) if comparison else None
        return VisualRequest(color=color, position=position, shape=shape, taller_than=taller_than)

    @staticmethod
    def parse_document_constraint(query: str) -> str | None:
        lowered = query.lower()
        return next((doc_id for name, doc_id in _DOC_NAMES.items() if name in lowered), None)

    def _largest_component(
        self, page_id: str, color: str
    ) -> tuple[int, tuple[int, int, int, int]] | None:
        image = self.images[page_id]
        pixels = image.load()
        target = _COLOR_RGB[color]
        remaining = {
            (x, y)
            for y in range(image.height)
            for x in range(image.width)
            if pixels[x, y] == target
        }
        best: list[tuple[int, int]] = []
        while remaining:
            start = remaining.pop()
            queue = deque([start])
            component = [start]
            while queue:
                x, y = queue.popleft()
                for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
                        component.append(neighbor)
            if len(component) > len(best):
                best = component
        # Text glyph quantization can introduce a few palette-colored pixels. Large
        # connected components are the intentionally frozen visual markers.
        if len(best) < 100:
            return None
        xs = [point[0] for point in best]
        ys = [point[1] for point in best]
        return len(best), (min(xs), min(ys), max(xs) + 1, max(ys) + 1)

    def _region(self, page_id: str, color: str) -> str | None:
        component = self._largest_component(page_id, color)
        if component is None:
            return None
        _, (x0, y0, x1, y1) = component
        image = self.images[page_id]
        width, height = image.size
        box_width, box_height = x1 - x0, y1 - y0
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        if box_width <= width * 0.22 and box_height >= height * 0.55:
            if center_x < width * 0.30:
                return "left"
            if center_x > width * 0.70:
                return "right"
        horizontal = "left" if center_x < width / 3 else "right" if center_x > 2 * width / 3 else "center"
        vertical = "upper" if center_y < height / 3 else "lower" if center_y > 2 * height / 3 else "center"
        if horizontal == "center" and vertical == "center":
            return "center"
        if horizontal == "center":
            return vertical
        if vertical == "center":
            return horizontal
        return f"{vertical}-{horizontal}"

    def _shape_matches(self, page_id: str, color: str, shape: str) -> bool:
        component = self._largest_component(page_id, color)
        if component is None:
            return False
        _, (x0, y0, x1, y1) = component
        width, height = x1 - x0, y1 - y0
        if shape == "sidebar":
            return height > 4 * width
        if shape == "stamp":
            return width / height > 1.5
        if shape == "cell":
            return 1.2 < width / height < 2.5 and height < 100
        return True

    def _visual_score(self, page_id: str, request: VisualRequest) -> float:
        if request.taller_than is not None:
            left = self._largest_component(page_id, request.taller_than[0])
            right = self._largest_component(page_id, request.taller_than[1])
            if left is None or right is None:
                return 0.0
            left_height = left[1][3] - left[1][1]
            right_height = right[1][3] - right[1][1]
            return 1.0 if left_height > right_height else 0.0
        if request.color is None or self._largest_component(page_id, request.color) is None:
            return 0.0
        if request.shape and not self._shape_matches(page_id, request.color, request.shape):
            return 0.0
        if request.position and self._region(page_id, request.color) != request.position:
            return 0.0
        return 1.0

    def retrieve_ocr(self, query: str, *, k: int = 3) -> PageRetrievalResult:
        return PageRetrievalResult(tuple(self.ocr_index.search(query, k=k)), ("ocr_text",), 0, {})

    def retrieve_pages(self, query: str, *, k: int = 3) -> PageRetrievalResult:
        request = self.parse_visual_request(query)
        if not request.has_visual_constraint:
            return PageRetrievalResult((), ("page_image",), 0, {})
        ranking: list[tuple[str, float]] = []
        regions: dict[str, str] = {}
        for asset in self.assets:
            score = self._visual_score(asset.id, request)
            if score <= 0.0:
                continue
            ranking.append((asset.id, score))
            if request.color:
                region = self._region(asset.id, request.color)
                if region:
                    regions[asset.id] = region
        ranking.sort(key=lambda item: (-item[1], item[0]))
        return PageRetrievalResult(tuple(ranking[:k]), ("page_image",), len(self.assets), regions)

    def retrieve_hybrid(self, query: str, *, k: int = 3) -> PageRetrievalResult:
        request = self.parse_visual_request(query)
        doc_id = self.parse_document_constraint(query)
        eligible = [asset for asset in self.assets if doc_id is None or asset.doc_id == doc_id]
        text_scores = dict(self.ocr_index.search(query, k=len(self.assets)))
        max_text = max(text_scores.values(), default=0.0)
        if not request.has_visual_constraint:
            ranking = [
                (asset.id, text_scores.get(asset.id, 0.0))
                for asset in eligible
                if text_scores.get(asset.id, 0.0) > 0.0
            ]
            ranking.sort(key=lambda item: (-item[1], item[0]))
            return PageRetrievalResult(tuple(ranking[:k]), ("ocr_text",), 0, {})
        ranking: list[tuple[str, float]] = []
        regions: dict[str, str] = {}
        for asset in eligible:
            visual_score = self._visual_score(asset.id, request)
            if visual_score <= 0.0:
                continue
            normalized_text = text_scores.get(asset.id, 0.0) / max_text if max_text > 0.0 else 0.0
            ranking.append((asset.id, 0.7 * visual_score + 0.3 * normalized_text))
            if request.color:
                region = self._region(asset.id, request.color)
                if region:
                    regions[asset.id] = region
        ranking.sort(key=lambda item: (-item[1], item[0]))
        return PageRetrievalResult(
            tuple(ranking[:k]),
            ("ocr_text", "page_image"),
            len(eligible),
            regions,
        )

    def answer(
        self,
        query: str,
        result: PageRetrievalResult,
        *,
        allow_text: bool,
        allow_pixels: bool,
    ) -> str:
        if not result.ranking:
            return "NO_EVIDENCE"
        asset = self.by_id[result.ranking[0][0]]
        lowered = query.lower()
        request = self.parse_visual_request(query)
        if "hotline" in lowered:
            return asset.facts.get("hotline", "UNKNOWN") if allow_text else "UNSUPPORTED_TEXT_EVIDENCE"
        if "shipping sla" in lowered:
            return asset.facts.get("shipping_sla", "UNKNOWN") if allow_text else "UNSUPPORTED_TEXT_EVIDENCE"
        if "q2 revenue" in lowered:
            return asset.facts.get("q2_revenue", "UNKNOWN") if allow_text else "UNSUPPORTED_TEXT_EVIDENCE"
        if "where" in lowered and request.color:
            if not allow_pixels:
                return "UNSUPPORTED_VISUAL_EVIDENCE"
            return result.region_for(asset.id) or "UNKNOWN"
        if request.has_visual_constraint:
            return asset.id if allow_pixels else "UNSUPPORTED_VISUAL_EVIDENCE"
        return asset.title if allow_text else "UNSUPPORTED_TEXT_EVIDENCE"

    def search(self, query: str, *, limit: int = 5) -> list[SourceHit]:
        result = self.retrieve_hybrid(query, k=limit)
        hits: list[SourceHit] = []
        for rank, (page_id, score) in enumerate(result.ranking, start=1):
            asset = self.by_id[page_id]
            region = result.region_for(page_id)
            hits.append(
                SourceHit(
                    record=asset.as_source_record(),
                    score=score,
                    rank=rank,
                    details={
                        "retrieval": "ocr_page_fusion",
                        "evidence_modalities": result.evidence_modalities,
                        "visual_candidates_scored": result.visual_candidates_scored,
                        "region": region,
                        "region_locator": asset.region_locator(region) if region else None,
                    },
                )
            )
        return hits
