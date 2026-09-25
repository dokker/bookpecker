"""Model-free layout inside the margin box.

1. words are joined into line fragments (horizontal dilation smaller than a column gutter);
2. fragments sharing a text row with another fragment are "columnar", lone ones are "wide";
3. fragments are stacked top → bottom into blocks when they are close, of similar glyph height,
   horizontally aligned and of compatible type (a columnar line never joins a wide block and vice
   versa, except short lone lines that fit inside the block, e.g. a column's last line).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import HeuristicCfg
from .base import Box, Region
from .frames import binarize_inv, line_mask


def estimate_em(bw: np.ndarray) -> float:
    """Median glyph height (px) from connected components; ~x-height…cap-height."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return 20.0
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    ok = (hs >= 5) & (hs <= 300) & (ws <= hs * 5)
    return float(np.median(hs[ok])) if ok.any() else 20.0


@dataclass
class _Frag:
    box: Box
    columnar: bool = False


@dataclass
class _Block:
    box: Box
    columnar: bool
    line_h: float
    frags: list[_Frag] = field(default_factory=list)


def estimate_word_gap(bw: np.ndarray, em: float) -> float:
    """Typical gap between words: high percentile of gaps between horizontally adjacent glyphs."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 2:
        return 0.5 * em
    s = stats[1:]
    s = s[(s[:, cv2.CC_STAT_HEIGHT] >= 0.3 * em) & (s[:, cv2.CC_STAT_HEIGHT] <= 3 * em)]
    if len(s) < 10:
        return 0.5 * em
    order = np.argsort(s[:, cv2.CC_STAT_LEFT])
    s = s[order]
    x0, y0 = s[:, cv2.CC_STAT_LEFT], s[:, cv2.CC_STAT_TOP]
    x1, y1 = x0 + s[:, cv2.CC_STAT_WIDTH], y0 + s[:, cv2.CC_STAT_HEIGHT]
    gaps = []
    for i in range(len(s)):
        # nearest glyph to the right on the same text line
        j = np.searchsorted(x0, x1[i])
        best = None
        for k in range(j, min(len(s), j + 40)):
            ov = min(y1[i], y1[k]) - max(y0[i], y0[k])
            if ov > 0.3 * em:
                best = x0[k] - x1[i]
                break
        if best is not None and 0 < best < 4 * em:
            gaps.append(best)
    return float(np.percentile(gaps, 90)) if len(gaps) >= 10 else 0.5 * em


def _fragments(bw: np.ndarray, em: float, cfg: HeuristicCfg) -> list[_Frag]:
    if cfg.h_kernel_em:
        kx = max(3, int(cfg.h_kernel_em * em))
    else:
        kx = int(np.clip(1.5 * estimate_word_gap(bw, em), 0.8 * em, 2.5 * em)) + 1
    joined = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    frags = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.35 * em and w < 2 * em:  # dust, dots, stray marks
            continue
        frags.append(_Frag(Box(x + kx // 2, y, x + w - kx // 2, y + h)))
    frags = _join_row_neighbours(frags)
    # rows: fragments overlapping vertically by >50% of the smaller height
    frags.sort(key=lambda f: f.box.cy)
    for i, f in enumerate(frags):
        for g in frags[i + 1:]:
            if g.box.y0 > f.box.y1:
                break
            ov = min(f.box.y1, g.box.y1) - max(f.box.y0, g.box.y0)
            if ov > 0.5 * min(f.box.h, g.box.h) and (g.box.x0 > f.box.x1 or f.box.x0 > g.box.x1):
                f.columnar = g.columnar = True
    return frags


def _same_row(a: Box, b: Box) -> bool:
    return min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5 * min(a.h, b.h)


def _join_row_neighbours(frags: list[_Frag]) -> list[_Frag]:
    """Join same-row fragments whose gap is word-sized for their own height (big titles)."""
    frags = sorted(frags, key=lambda f: f.box.x0)
    changed = True
    while changed:
        changed = False
        for i, f in enumerate(frags):
            for g in frags[i + 1:]:
                gap = g.box.x0 - f.box.x1
                if 0 <= gap < 0.6 * min(f.box.h, g.box.h) and _same_row(f.box, g.box):
                    f.box = f.box.union(g.box)
                    frags.remove(g)
                    changed = True
                    break
            if changed:
                break
    return frags


def _blocks(frags: list[_Frag], em: float) -> list[_Block]:
    blocks: list[_Block] = []
    tol = 1.5 * em
    for f in sorted(frags, key=lambda f: (f.box.y0, f.box.x0)):
        best = None
        for b in blocks:
            gap = f.box.y0 - b.box.y1
            h = max(f.box.h, b.line_h)
            if gap > 1.1 * h or gap < -0.5 * h:
                continue
            ratio = max(f.box.h, b.line_h) / max(1.0, min(f.box.h, b.line_h))
            if ratio > 1.4:
                continue
            ov = min(f.box.x1, b.box.x1) - max(f.box.x0, b.box.x0)
            if ov < 0.5 * min(f.box.w, b.box.w):
                continue
            fits_inside = f.box.x0 >= b.box.x0 - tol and f.box.x1 <= b.box.x1 + tol
            if f.columnar != b.columnar and not (not f.columnar and fits_inside):
                continue
            if best is None or gap < best[0]:
                best = (gap, b)
        if best:
            b = best[1]
            b.box = b.box.union(f.box)
            b.frags.append(f)
            b.line_h = float(np.median([g.box.h for g in b.frags]))
        else:
            blocks.append(_Block(f.box, f.columnar, f.box.h, [f]))
    return blocks


def _classify(bw: np.ndarray, block: _Block, em: float, cfg: HeuristicCfg) -> str:
    x0, y0, x1, y1 = block.box.as_int()
    crop = bw[y0:y1, x0:x1]
    if crop.size == 0:
        return "text"
    n, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n <= 1:
        return "text"
    comp = stats[1:]
    areas = comp[:, cv2.CC_STAT_AREA].astype(float)
    hs = comp[:, cv2.CC_STAT_HEIGHT]
    ws = comp[:, cv2.CC_STAT_WIDTH]
    huge = (hs > 3.5 * em) | (ws > 12 * em)
    if areas.sum() and areas[huge].sum() / areas.sum() > cfg.figure_big_ink:
        return "figure"
    glyph = (hs >= 4) & ~huge
    if glyph.any():
        med = float(np.median(hs[glyph]))
        if med > cfg.title_ratio * em and len(block.frags) <= 3:
            return "title"
    return "text"


def detect(gray: np.ndarray, content: Box, cfg: HeuristicCfg) -> tuple[list[Region], float]:
    """Return regions (page coordinates) and the estimated text height `em`."""
    x0, y0, x1, y1 = content.as_int()
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return [], 20.0
    bw = binarize_inv(sub)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # specks
    em = estimate_em(bw)
    text_bw = cv2.subtract(bw, line_mask(bw))  # rules would glue columns together

    regions = []
    for b in _blocks(_fragments(text_bw, em, cfg), em):
        kind = _classify(bw, b, em, cfg)
        regions.append(Region(Box(b.box.x0 + x0, b.box.y0 + y0, b.box.x1 + x0, b.box.y1 + y0),
                              kind, 1.0, f"heuristic:{kind}"))
    return regions, em
