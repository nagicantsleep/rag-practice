"""Pinned pretrained CLIP text-to-image retrieval for the M08.5 control."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from .ppm import RasterImage, read_p3_ppm

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPTextToImageRetriever:
    """Small exhaustive CLIP retriever with explicit image preprocessing.

    The benchmark images are square 8x8 P3 rasters. We resize them directly to
    the model's configured square image size, rescale to [0, 1], and apply the
    canonical CLIP normalization constants. No caption, title, site, or qrel is
    exposed to image embedding or ranking.
    """

    def __init__(
        self,
        image_paths: Mapping[str, Path],
        *,
        model_name: str,
        revision: str,
        device: str = "cpu",
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("CLIPTextToImageRetriever requires the 'neural' and 'pretrained' extras") from exc

        started = perf_counter()
        if tokenizer is None or model is None:
            try:
                from transformers import AutoTokenizer, CLIPModel
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("CLIPTextToImageRetriever requires the 'pretrained' extra") from exc
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = CLIPModel.from_pretrained(model_name, revision=revision, use_safetensors=True)

        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.image_ids = sorted(image_paths)
        self.images = {image_id: read_p3_ppm(image_paths[image_id]) for image_id in self.image_ids}
        if hasattr(self.model, "to"):
            self.model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.model_load_ms = (perf_counter() - started) * 1000
        self.index_build_ms = 0.0
        self.image_vectors = self._embed_images()

    @property
    def dimensions(self) -> int:
        return int(self.image_vectors.shape[1])

    def logical_index_bytes(self) -> int:
        return int(self.image_vectors.numel() * self.image_vectors.element_size())

    def _image_size(self) -> int:
        vision_config = getattr(getattr(self.model, "config", None), "vision_config", None)
        return int(getattr(vision_config, "image_size", 224))

    def _preprocess_image(self, image: RasterImage):
        torch = self.torch
        values = torch.tensor(image.pixels, dtype=torch.float32, device=self.device)
        values = values.reshape(image.height, image.width, 3).permute(2, 0, 1)
        values = values / float(image.max_value)
        values = torch.nn.functional.interpolate(
            values.unsqueeze(0),
            size=(self._image_size(), self._image_size()),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        mean = torch.tensor(CLIP_IMAGE_MEAN, dtype=values.dtype, device=self.device).view(3, 1, 1)
        std = torch.tensor(CLIP_IMAGE_STD, dtype=values.dtype, device=self.device).view(3, 1, 1)
        return (values - mean) / std

    def _embed_images(self):
        torch = self.torch
        started = perf_counter()
        batch = torch.stack([self._preprocess_image(self.images[image_id]) for image_id in self.image_ids])
        with torch.no_grad():
            vectors = self.model.get_image_features(pixel_values=batch)
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        self.index_build_ms = (perf_counter() - started) * 1000
        return vectors

    def search(self, query: str, *, k: int = 3) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        torch = self.torch
        encoded = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in encoded.items()}
        with torch.no_grad():
            vector = self.model.get_text_features(**encoded)
            vector = torch.nn.functional.normalize(vector, dim=-1)[0]
            scores = self.image_vectors @ vector
        ranked = [(image_id, float(scores[index].item())) for index, image_id in enumerate(self.image_ids)]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]
