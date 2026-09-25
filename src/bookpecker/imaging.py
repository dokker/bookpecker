"""Rasterisation and page image preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import PreprocessCfg


def rasterize(pdf: Path, page: int, dpi: int, out: Path) -> None:
    """Render PDF page (1-based) to a grayscale PNG."""
    import pymupdf

    with pymupdf.open(pdf) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out))


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write {path}")


def normalize(gray: np.ndarray) -> np.ndarray:
    """Stretch so the paper level becomes white and the darkest ink black."""
    lo = float(np.percentile(gray, 1))
    hi = float(np.percentile(gray, 90))  # most pixels are paper
    if hi - lo < 30:
        return gray
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def flatten(gray: np.ndarray) -> np.ndarray:
    """Remove slowly varying background (spine shadow): divide by a blurred paper estimate."""
    k = max(15, (min(gray.shape) // 40) | 1)  # well above stroke width, below shadow scale
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((k // 4 | 1, k // 4 | 1), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 8)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_skew(gray: np.ndarray, max_deg: float) -> float:
    """Angle (deg) that makes text lines horizontal: maximise row-profile variance."""
    small = gray
    scale = 1000.0 / max(gray.shape)
    if scale < 1:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    center = (w / 2, h / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rot.sum(axis=1)))

    best = max(np.arange(-max_deg, max_deg + 1e-9, 0.25), key=score)
    fine = max(np.arange(best - 0.25, best + 0.25 + 1e-9, 0.05), key=score)
    return float(round(fine, 2))


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def preprocess(gray: np.ndarray, cfg: PreprocessCfg) -> tuple[np.ndarray, float]:
    angle = 0.0
    if cfg.flatten:
        gray = flatten(gray)
    if cfg.normalize:
        gray = normalize(gray)
    if cfg.deskew:
        angle = estimate_skew(gray, cfg.max_skew_deg)
        gray = rotate(gray, angle)
    return gray, angle
