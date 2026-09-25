"""Sidebar box detection: ruled frames and tinted background panels."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import SidebarCfg
from .base import Box, merge_overlapping


def binarize_inv(gray: np.ndarray) -> np.ndarray:
    """Ink = 255, paper = 0."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def line_mask(bw: np.ndarray, min_frac: float = 0.12) -> np.ndarray:
    """Long horizontal and vertical rules."""
    h, w = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * min_frac)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * min_frac * 0.5))))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk) | cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)


def _boxes_from_mask(mask: np.ndarray, min_w: int, min_h: int, ox: float, oy: float) -> list[Box]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_w and h >= min_h:
            out.append(Box(x + ox, y + oy, x + w + ox, y + h + oy))
    return out


def detect_frames(gray: np.ndarray, content: Box, cfg: SidebarCfg, em: float) -> list[Box]:
    """Candidate sidebar boxes inside the content area (page coordinates)."""
    x0, y0, x1, y1 = content.as_int()
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return []
    h, w = sub.shape
    min_w, min_h = int(0.2 * w), int(max(3 * em, 0.04 * h))
    frames: list[Box] = []

    if cfg.lines:
        lines = line_mask(binarize_inv(sub))
        closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        # a ruled box has rules on (at least) top and bottom: fill it via its outer contour
        filled = np.zeros_like(closed)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            bx, by, bw_, bh = cv2.boundingRect(c)
            if bw_ >= min_w and bh >= min_h:
                cv2.rectangle(filled, (bx, by), (bx + bw_, by + bh), 255, -1)
        frames += _boxes_from_mask(filled, min_w, min_h, x0, y0)

    if cfg.shading:
        paper = float(np.percentile(sub, 90))
        tint = ((sub < paper - cfg.shade_delta) & (sub > 90)).astype(np.uint8) * 255
        k = max(3, int(em * 1.5))
        tint = cv2.morphologyEx(tint, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        tint = cv2.morphologyEx(tint, cv2.MORPH_CLOSE, np.ones((k * 2, k * 2), np.uint8))
        frames += _boxes_from_mask(tint, min_w, min_h, x0, y0)

    # drop frames that are basically the whole content box (page borders)
    frames = [f for f in frames if f.area < 0.85 * content.area]
    return merge_overlapping(frames)
