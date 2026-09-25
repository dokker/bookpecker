"""Margin mask (mirrored by page side) and region filtering."""

from __future__ import annotations

import numpy as np

from ..config import Margins, PageConfig
from .base import Box, Region


def content_box(width: int, height: int, side: str, m: Margins) -> Box:
    """Page area inside the margins. Right (recto) pages have the spine on the left."""
    left, right = (m.inner, m.outer) if side == "right" else (m.outer, m.inner)
    return Box(width * left, height * m.top, width * (1 - right), height * (1 - m.bottom))


def ink_ratio(binary_inv: np.ndarray, box: Box) -> float:
    x0, y0, x1, y1 = box.as_int()
    crop = binary_inv[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return float(crop.mean() / 255.0) if crop.size else 0.0


def filter_regions(
    regions: list[Region],
    content: Box,
    page_w: int,
    page_h: int,
    cfg: PageConfig,
    binary_inv: np.ndarray | None = None,
) -> list[Region]:
    """Mark (not delete) regions to drop; clip the survivors to the content box."""
    min_area = cfg.min_region_area * page_w * page_h
    for r in regions:
        if r.dropped:
            continue
        if r.kind in cfg.drop_kinds:
            r.dropped = f"kind:{r.kind}"
        elif r.box.overlap_ratio(content) < cfg.keep_inside_ratio:
            r.dropped = "margin"
        elif r.box.area < min_area:
            r.dropped = "tiny"
        else:
            if cfg.clip_to_content:
                r.box = r.box.intersect(content)
            if binary_inv is not None and r.kind != "table" and ink_ratio(binary_inv, r.box) < 0.003:
                r.dropped = "empty"
    return regions


def assign_groups(regions: list[Region], frames: list[Box]) -> list[Box]:
    """Put kept regions whose centre lies in a frame into that frame's group.

    Frames that are really tables or figures (one such region covers most of the frame) or that
    hold no text are discarded. Returns the frames actually used, re-indexed.
    """
    used: list[Box] = []
    for frame in frames:
        inside = [r for r in regions if r.dropped is None and frame.contains_point(r.box.cx, r.box.cy)]
        if not inside:
            continue
        if any(r.kind in ("table", "figure") and r.box.area > 0.6 * frame.area for r in inside):
            continue
        if not any(r.kind in ("text", "title", "sidebar", "caption") for r in inside):
            continue
        gid = len(used)
        used.append(frame)
        for r in inside:
            if r.group is None:
                r.group = gid
    return used
