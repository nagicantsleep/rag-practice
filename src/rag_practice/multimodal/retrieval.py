"""Transparent text, pixel, and fused retrieval for M08.5."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path

from rag_practice.ir.bm25 import BM25Index
from rag_practice.sources.base import SourceHit, SourceRecord
from .ppm import RasterImage, read_p3_ppm

_COLORS = ("red", "green", "blue", "yellow")
_POSITIONS = ("upper-left", "upper-right", "lower-left", "lower-right")


@dataclass(frozen=True)
class ImageAsset:
    id: str
    title: str
    caption: str
    site: str
    kind: str
    file_name: str
    quarter: str | None = None

    @property
    def locator(self) -> str:
        return f"image://benchmark/{self.file_name}"

    def as_source_record(self) -> SourceRecord:
        metadata: dict[str, object] = {"site": self.site, "kind": self.kind, "file": self.file_name}
        if self.quarter is not None:
            metadata["quarter"] = self.quarter
        return SourceRecord(id=self.id, source_type="image", locator=self.locator, title=self.title, content=self.caption, metadata=metadata)


@dataclass(frozen=True)
class VisualRequest:
    color: str | None = None
    position: str | None = None
    taller_than: tuple[str, str] | None = None

    @property
    def has_visual_constraint(self) -> bool:
        return self.color is not None or self.position is not None or self.taller_than is not None


@dataclass(frozen=True)
class RetrievalResult:
    ranking: tuple[tuple[str, float], ...]
    evidence_modalities: tuple[str, ...]
    visual_candidates_scored: int = 0


class MultimodalImageIndex:
    name = "controlled-multimodal-images"

    def __init__(self, benchmark_dir: Path) -> None:
        self.benchmark_dir = benchmark_dir
        self.images_dir = benchmark_dir / "images"
        self.assets = self._load_assets(benchmark_dir / "records.jsonl")
        self.by_id = {asset.id: asset for asset in self.assets}
        self.images: dict[str, RasterImage] = {asset.id: read_p3_ppm(self.images_dir / asset.file_name) for asset in self.assets}
        self.image_bytes_indexed = sum((self.images_dir / asset.file_name).stat().st_size for asset in self.assets)
        text_documents = {
            asset.id: " ".join(part for part in (asset.title, asset.caption, asset.site, asset.kind, asset.quarter or "") if part)
            for asset in self.assets
        }
        self.text_chars_indexed = sum(len(text) for text in text_documents.values())
        self.text_index = BM25Index(text_documents)

    @staticmethod
    def _load_assets(path: Path) -> tuple[ImageAsset, ...]:
        assets = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            assets.append(ImageAsset(id=item["id"], title=item["title"], caption=item["caption"], site=item["site"], kind=item["kind"], file_name=item["file"], quarter=item.get("quarter")))
        return tuple(assets)

    @staticmethod
    def parse_visual_request(query: str) -> VisualRequest:
        lowered = query.lower()
        color = next((name for name in _COLORS if name in lowered), None)
        position = next((name for name in _POSITIONS if name in lowered or name.replace("-", " ") in lowered), None)
        comparison = re.search(r"(red|green|blue|yellow)\s+bar\s+taller\s+than\s+(?:the\s+)?(red|green|blue|yellow)\s+bar", lowered)
        taller_than = (comparison.group(1), comparison.group(2)) if comparison else None
        return VisualRequest(color=color, position=position, taller_than=taller_than)

    @staticmethod
    def parse_metadata_constraints(query: str) -> tuple[str | None, str | None]:
        lowered = query.lower()
        site = next((name for name in ("alpha", "beta", "north", "south") if re.search(rf"\b{name}\b", lowered)), None)
        if "panel" in lowered:
            kind = "panel"
        elif "chart" in lowered:
            kind = "chart"
        elif "diagram" in lowered:
            kind = "diagram"
        else:
            kind = None
        return site, kind

    def _visual_score(self, asset_id: str, request: VisualRequest) -> float:
        image = self.images[asset_id]
        if request.taller_than is not None:
            left, right = request.taller_than
            left_count, right_count = image.color_count(left), image.color_count(right)
            return 1.0 if left_count > right_count and left_count > 0 else 0.0
        if request.color and request.position:
            return 1.0 if image.dominant_quadrant(request.color) == request.position else 0.0
        if request.color:
            return 1.0 if image.color_count(request.color) > 0 else 0.0
        return 0.0

    def retrieve_text(self, query: str, *, k: int = 3) -> RetrievalResult:
        return RetrievalResult(tuple(self.text_index.search(query, k=k)), ("text_surrogate",), 0)

    def retrieve_pixels(self, query: str, *, k: int = 3) -> RetrievalResult:
        request = self.parse_visual_request(query)
        if not request.has_visual_constraint:
            return RetrievalResult((), ("image",), 0)
        ranking = [(asset.id, self._visual_score(asset.id, request)) for asset in self.assets]
        ranking = [item for item in ranking if item[1] > 0.0]
        ranking.sort(key=lambda item: (-item[1], item[0]))
        return RetrievalResult(tuple(ranking[:k]), ("image",), len(self.assets))

    def retrieve_multimodal(self, query: str, *, k: int = 3) -> RetrievalResult:
        request = self.parse_visual_request(query)
        site, kind = self.parse_metadata_constraints(query)
        text_scores = dict(self.text_index.search(query, k=len(self.assets)))
        max_text = max(text_scores.values(), default=0.0)
        eligible = [asset for asset in self.assets if (site is None or asset.site == site) and (kind is None or asset.kind == kind)]
        if not request.has_visual_constraint:
            ranking = [(asset.id, text_scores.get(asset.id, 0.0)) for asset in eligible if text_scores.get(asset.id, 0.0) > 0.0]
            ranking.sort(key=lambda item: (-item[1], item[0]))
            return RetrievalResult(tuple(ranking[:k]), ("text_surrogate",), 0)
        ranking = []
        for asset in eligible:
            visual_score = self._visual_score(asset.id, request)
            if visual_score <= 0.0:
                continue
            normalized_text = text_scores.get(asset.id, 0.0) / max_text if max_text > 0.0 else 0.0
            ranking.append((asset.id, 0.6 * visual_score + 0.4 * normalized_text))
        ranking.sort(key=lambda item: (-item[1], item[0]))
        return RetrievalResult(tuple(ranking[:k]), ("text_surrogate", "image"), len(eligible))

    def answer(self, query: str, result: RetrievalResult, *, allow_text: bool, allow_pixels: bool) -> str:
        if not result.ranking:
            return "NO_EVIDENCE"
        asset = self.by_id[result.ranking[0][0]]
        request = self.parse_visual_request(query)
        lowered = query.lower()
        if request.has_visual_constraint and not allow_pixels:
            return "UNSUPPORTED_VISUAL_EVIDENCE"
        if "where" in lowered and allow_pixels and request.color:
            return self.images[asset.id].dominant_quadrant(request.color) or "UNKNOWN"
        if request.has_visual_constraint and allow_pixels:
            return asset.title
        if not allow_text:
            return "UNSUPPORTED_TEXT_EVIDENCE"
        if "north" in lowered:
            return asset.site
        if "q2" in lowered:
            return asset.quarter or "UNKNOWN"
        return asset.title

    def search(self, query: str, *, limit: int = 5) -> list[SourceHit]:
        result = self.retrieve_multimodal(query, k=limit)
        return [
            SourceHit(record=self.by_id[asset_id].as_source_record(), score=score, rank=rank, details={"retrieval": "multimodal_fusion", "evidence_modalities": result.evidence_modalities, "visual_candidates_scored": result.visual_candidates_scored})
            for rank, (asset_id, score) in enumerate(result.ranking, start=1)
        ]
