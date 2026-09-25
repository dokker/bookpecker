"""PaddleOCR-VL (0.9B VLM) engine, run on single region crops.

Experimental: needs `paddleocr>=3.3` and a PaddlePaddle build (see README). The layout step is
done by bookpecker, so the pipeline's own layout detection is switched off and the crop is
recognised as one element with a task prompt matching the region kind.
"""

from __future__ import annotations

import numpy as np

from ..config import PaddleVLCfg
from .base import OcrLine, OcrResult

PROMPTS = {"table": "table", "text": "ocr", "title": "ocr", "caption": "ocr", "sidebar": "ocr"}


def _texts(res) -> list[str]:
    data = res.json if isinstance(getattr(res, "json", None), dict) else {}
    data = data.get("res", data)
    blocks = data.get("parsing_res_list") or []
    texts = [str(b.get("block_content", "")) for b in blocks if b.get("block_content")]
    if texts:
        return texts
    md = getattr(res, "markdown", None)
    if isinstance(md, dict) and md.get("markdown_texts"):
        return [str(md["markdown_texts"])]
    return []


class PaddleVLEngine:
    name = "paddle_vl"

    def __init__(self, cfg: PaddleVLCfg):
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("OCR engine 'paddle_vl' needs paddleocr>=3.3 (see README)") from e
        kwargs = {"use_layout_detection": False, "use_doc_orientation_classify": False,
                  "use_doc_unwarping": False, **cfg.options}
        if cfg.device:
            kwargs["device"] = cfg.device
        self._pipe = PaddleOCRVL(**kwargs)

    def recognize(self, image: np.ndarray, kind: str) -> OcrResult:
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        texts: list[str] = []
        for res in self._pipe.predict(
            image,
            use_layout_detection=False,
            prompt_label=PROMPTS.get(kind, "ocr"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        ):
            texts += _texts(res)
        joined = "\n\n".join(t.strip() for t in texts if t.strip())
        if kind == "table":
            return OcrResult(self.name, [], joined)
        # no per-line geometry from the VLM: blank lines delimit paragraphs (`par`), and the
        # paragraph logic falls back to punctuation rules
        lines: list[OcrLine] = []
        par = 0
        for raw in joined.splitlines():
            if raw.strip():
                lines.append(OcrLine(raw.strip(), [], par))
            elif lines and lines[-1].par == par:
                par += 1
        return OcrResult(self.name, lines)
