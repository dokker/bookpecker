"""Tesseract (CPU) engine with line geometry for paragraph reconstruction."""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..config import TesseractCfg
from .base import OcrLine, OcrResult

PAD = 20


class TesseractEngine:
    name = "tesseract"

    def __init__(self, languages: list[str], cfg: TesseractCfg):
        import pytesseract

        self._tess = pytesseract
        self.lang = "+".join(languages)
        self.cfg = cfg
        if cfg.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = os.path.abspath(cfg.tessdata_dir)
        available = set(pytesseract.get_languages(config=""))
        missing = [lang for lang in languages if lang not in available]
        if missing:
            raise RuntimeError(
                f"tesseract language data missing: {', '.join(missing)} "
                f"(Arch/Manjaro: sudo pacman -S {' '.join('tesseract-data-' + m for m in missing)})"
            )

    def recognize(self, image: np.ndarray, kind: str) -> OcrResult:
        img = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=255)
        config = f"--oem {self.cfg.oem} --psm {self.cfg.psm} -c preserve_interword_spaces=0 {self.cfg.extra}"
        data = self._tess.image_to_data(img, lang=self.lang, config=config,
                                        output_type=self._tess.Output.DICT)
        lines: dict[tuple, dict] = {}
        pars: dict[tuple, int] = {}
        for i, word in enumerate(data["text"]):
            word = (word or "").strip()
            if not word or float(data["conf"][i]) < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            pkey = key[:2]
            pars.setdefault(pkey, len(pars))
            x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
            ln = lines.setdefault(key, {"words": [], "box": [x, y, x + w, y + h], "par": pars[pkey]})
            ln["words"].append(word)
            b = ln["box"]
            ln["box"] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
        out = []
        for ln in lines.values():  # tesseract reading order
            box = [ln["box"][0] - PAD, ln["box"][1] - PAD, ln["box"][2] - PAD, ln["box"][3] - PAD]
            out.append(OcrLine(" ".join(ln["words"]), [float(v) for v in box], ln["par"]))
        return OcrResult(self.name, out)
