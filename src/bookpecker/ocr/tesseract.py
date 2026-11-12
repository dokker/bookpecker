"""Tesseract (CPU) engine with line geometry for paragraph reconstruction."""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..config import TesseractCfg
from .base import OcrLine, OcrResult

PAD = 20
_ENV_OVERRIDES: dict[str, str] = {}


def _apply_env() -> None:
    """pytesseract hands its module-level `environ` to every tesseract process it starts."""
    import pytesseract.pytesseract as pt

    pt.environ = {**os.environ, **_ENV_OVERRIDES} if _ENV_OVERRIDES else os.environ


def limit_threads(n: int | None) -> None:
    """OpenMP threads per tesseract process (None: tesseract's default).

    Only the tesseract child processes get the limit – setting OMP_THREAD_LIMIT in our own
    environment would also cap Paddle's CPU kernels in this process. A limit the user exported
    is left alone.
    """
    if n and "OMP_THREAD_LIMIT" not in os.environ:
        _ENV_OVERRIDES["OMP_THREAD_LIMIT"] = str(n)
    else:
        _ENV_OVERRIDES.pop("OMP_THREAD_LIMIT", None)
    _apply_env()


class TesseractEngine:
    name = "tesseract"
    cpu_only = True  # thread safe (one tesseract process per call): may run in a worker pool

    def __init__(self, languages: list[str], cfg: TesseractCfg):
        import pytesseract

        self._tess = pytesseract
        self.lang = "+".join(languages)
        self.cfg = cfg
        if cfg.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = os.path.abspath(cfg.tessdata_dir)
            _apply_env()
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
            conf = float(data["conf"][i])
            if not word or conf < 0:
                continue
            if conf < self.cfg.min_word_conf and not any(c.isdigit() for c in word):
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            pkey = key[:2]
            pars.setdefault(pkey, len(pars))
            x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
            ln = lines.setdefault(key, {"words": [], "box": [x, y, x + w, y + h], "par": pars[pkey]})
            ln["words"].append((word, conf))
            b = ln["box"]
            ln["box"] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
        out = []
        for ln in lines.values():  # tesseract reading order
            words = strip_edge_junk(ln["words"], self.cfg.edge_junk_conf)
            if not words or is_junk_line(words):
                continue
            box = [ln["box"][0] - PAD, ln["box"][1] - PAD, ln["box"][2] - PAD, ln["box"][3] - PAD]
            conf = sum(c for _, c in ln["words"]) / len(ln["words"]) / 100
            out.append(OcrLine(" ".join(words), [float(v) for v in box], ln["par"], round(conf, 3)))
        return OcrResult(self.name, out)


# common one/two letter Hungarian words – not ornament scraps
HU_SHORT = {"a", "az", "s", "és", "is", "ez", "e", "ha", "le", "be", "ki", "ne", "se", "el", "ő",
            "ők", "én", "te", "mi", "ti", "ma", "no", "jó", "ő", "fő", "kő", "ló", "tó", "új", "év",
            "út", "id", "pl", "km", "kp", "tp", "ép", "és", "--", "–", "—", "-"}

SHORT_WORDS = {"a", "s", "e", "é", "ó", "ő", "A", "S", "É", "Ó", "Ő", "—", "–", "-"}


def strip_edge_junk(words: list[tuple[str, float]], max_conf: float) -> list[str]:
    """Drop 1–2 character low-confidence scraps at line ends (ornament edges, gutter shadow)."""
    def junk(w: str, c: float) -> bool:
        return len(w) <= 2 and c < max_conf and w not in SHORT_WORDS and not w.isdigit()

    while words and junk(*words[0]):
        words = words[1:]
    while words and junk(*words[-1]):
        words = words[:-1]
    return [w for w, _ in words]


def is_junk_line(words: list[str]) -> bool:
    """Ornament texture read as letters: mostly 1–2 character scraps, e.g. "d Z H 2 s"."""
    if len(words) < 3:
        return False
    def scrap(w: str) -> bool:
        core = w.strip(".,;:!?\"'()[]")
        return len(core) <= 2 and core.lower() not in HU_SHORT

    scraps = sum(1 for w in words if scrap(w))
    return scraps / len(words) >= 0.6
