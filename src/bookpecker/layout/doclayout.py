"""Layout via PaddleOCR's PP-DocLayout models (runs on CPU or GPU)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import DocLayoutCfg
from .base import Box, Region

# PP-DocLayout(_plus) / PP-DocLayoutV2 labels → our region kinds
LABEL_MAP = {
    "text": "text", "abstract": "text", "content": "text", "reference": "text",
    "reference_content": "text", "algorithm": "text", "footnote": "text",
    "vertical_text": "text", "formula": "text", "display_formula": "text",
    "paragraph_title": "title", "doc_title": "title",
    "table": "table",
    "image": "figure", "chart": "figure", "seal": "figure",
    "header_image": "ornament", "footer_image": "ornament",
    "figure_title": "caption", "table_title": "caption", "chart_title": "caption",
    "vision_footnote": "caption",
    "header": "header", "footer": "footer", "number": "page_number",
    "aside_text": "text",
    "formula_number": "text",
}


@lru_cache(maxsize=2)
def _model(model_name: str, device: str | None):
    try:
        from paddleocr import LayoutDetection
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "layout_engine 'doclayout' needs PaddleOCR: pip install -e '.[paddle]' "
            "plus paddlepaddle (see README), or set layout_engine: heuristic"
        ) from e
    kwargs = {"model_name": model_name}
    if device:
        kwargs["device"] = device
    return LayoutDetection(**kwargs)


def detect(image: np.ndarray, cfg: DocLayoutCfg) -> list[Region]:
    model = _model(cfg.model_name, cfg.device)
    if image.ndim == 2:  # models expect 3-channel input
        image = np.repeat(image[:, :, None], 3, axis=2)
    regions: list[Region] = []
    for res in model.predict(image, batch_size=1, layout_nms=True, threshold=cfg.threshold):
        data = res.json if isinstance(res.json, dict) else {}
        boxes = data.get("res", data).get("boxes", [])
        for b in boxes:
            label = str(b.get("label", "text"))
            kind = LABEL_MAP.get(label, "text")
            regions.append(Region(Box.from_list(b["coordinate"]), kind, float(b.get("score", 1.0)),
                                  f"doclayout:{label}"))
    return regions
