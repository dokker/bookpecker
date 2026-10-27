"""PP-OCRv5 engine: text line detection + latin-script recognition (GPU or CPU).

Much more robust than Tesseract on the condensed, curved lines next to the spine, and it gives
line geometry for paragraph reconstruction. Its weak spot is ű/ő read as ú/ó – the spell step
(`spell.py`) repairs those against the Hungarian dictionary.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import PaddleOcrCfg
from .base import OcrLine, OcrResult

PAD = 3  # pixels above/below each detected line before recognition
PAD_X = 0.5  # left/right slack, share of the line height


@lru_cache(maxsize=2)
def _models(det_model: str, rec_model: str, device: str | None):
    try:
        from paddleocr import TextDetection, TextRecognition
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("OCR engine 'paddle_ocr' needs PaddleOCR: pip install -e '.[paddle]' "
                           "plus paddlepaddle (see README)") from e
    kw = {"device": device} if device else {}
    return TextDetection(model_name=det_model, **kw), TextRecognition(model_name=rec_model, **kw)


def group_rows(boxes: list[list[float]], min_overlap: float = 0.5) -> list[list[int]]:
    """Indices of line fragments grouped into visual rows (top → bottom, left → right).

    The detector may split one printed line at a wide gap; fragments that overlap vertically by
    `min_overlap` of the smaller height belong to the same row.
    """
    rows: list[list[int]] = []
    for i in sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2):
        b = boxes[i]
        for row in rows:
            r = boxes[row[-1]]
            overlap = min(b[3], r[3]) - max(b[1], r[1])
            if overlap >= min_overlap * min(b[3] - b[1], r[3] - r[1]):
                row.append(i)
                break
        else:
            rows.append([i])
    return [sorted(row, key=lambda i: boxes[i][0]) for row in rows]


class PaddleOcrEngine:
    name = "paddle_ocr"

    def __init__(self, cfg: PaddleOcrCfg):
        self.cfg = cfg
        self._det, self._rec = _models(cfg.det_model, cfg.rec_model, cfg.device)

    def recognize(self, image: np.ndarray, kind: str) -> OcrResult:
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        h, w = image.shape[:2]
        det = next(iter(self._det.predict(image, batch_size=1)))
        polys = det.json.get("res", {}).get("dt_polys", [])
        boxes, crops = [], []
        for poly in polys:
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            # detector boxes clip a thin line-end hyphen: generous horizontal slack
            pad_x = max(PAD, int(PAD_X * (max(ys) - min(ys))))
            x0, y0 = max(0, int(min(xs)) - pad_x), max(0, int(min(ys)) - PAD)
            x1, y1 = min(w, int(max(xs)) + pad_x), min(h, int(max(ys)) + PAD)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            boxes.append([float(x0), float(y0), float(x1), float(y1)])
            crops.append(image[y0:y1, x0:x1])
        if not crops:
            return OcrResult(self.name, [])
        texts: list[tuple[str, float]] = []
        for res in self._rec.predict(crops, batch_size=self.cfg.batch_size):
            j = res.json.get("res", {})
            texts.append((str(j.get("rec_text", "")).strip(), float(j.get("rec_score", 0.0))))

        lines: list[OcrLine] = []
        for row in group_rows(boxes):
            parts = [i for i in row if texts[i][0] and texts[i][1] >= self.cfg.min_score]
            if not parts:
                continue
            box = [min(boxes[i][0] for i in parts), min(boxes[i][1] for i in parts),
                   max(boxes[i][2] for i in parts), max(boxes[i][3] for i in parts)]
            score = min(texts[i][1] for i in parts)
            lines.append(OcrLine(" ".join(texts[i][0] for i in parts), box, conf=round(score, 3)))
        return OcrResult(self.name, lines)
