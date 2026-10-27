"""OCR engine interface and registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from ..config import PageConfig


@dataclass
class OcrLine:
    text: str
    box: list[float]  # x0, y0, x1, y1 relative to the region crop
    par: int = -1  # engine's own paragraph id, if any
    conf: float = -1.0  # recognition confidence 0–1; -1 = unknown


@dataclass
class OcrResult:
    engine: str
    lines: list[OcrLine] = field(default_factory=list)
    markdown: str | None = None  # pre-formatted output (tables from VLMs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OcrResult:
        return cls(d["engine"], [OcrLine(**ln) for ln in d.get("lines", [])], d.get("markdown"))


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray, kind: str) -> OcrResult: ...


_ENGINES: dict[tuple, OcrEngine] = {}


def get_engine(name: str, cfg: PageConfig) -> OcrEngine:
    if name == "tesseract":
        key = (name, tuple(cfg.languages), cfg.tesseract.model_dump_json())
        if key not in _ENGINES:
            from .tesseract import TesseractEngine

            _ENGINES[key] = TesseractEngine(cfg.languages, cfg.tesseract)
    elif name == "paddle_vl":
        key = (name, cfg.paddle_vl.model_dump_json())
        if key not in _ENGINES:
            from .paddle_vl import PaddleVLEngine

            _ENGINES[key] = PaddleVLEngine(cfg.paddle_vl)
    elif name == "paddle_ocr":
        key = (name, cfg.paddle_ocr.model_dump_json())
        if key not in _ENGINES:
            from .paddle_ocr import PaddleOcrEngine

            _ENGINES[key] = PaddleOcrEngine(cfg.paddle_ocr)
    elif name == "ensemble":
        key = (name, cfg.spell.enabled, cfg.spell.dictionary, cfg.paddle_ocr.model_dump_json(),
               tuple(cfg.languages), cfg.tesseract.model_dump_json())
        if key not in _ENGINES:
            from ..spell import Speller
            from .ensemble import EnsembleEngine

            known = Speller(cfg.spell.dictionary).known if cfg.spell.enabled else None
            _ENGINES[key] = EnsembleEngine(get_engine("paddle_ocr", cfg), get_engine("tesseract", cfg), known)
    else:
        raise ValueError(f"unknown OCR engine {name!r} (ensemble | paddle_ocr | tesseract | paddle_vl)")
    return _ENGINES[key]
