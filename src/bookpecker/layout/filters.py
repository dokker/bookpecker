"""Margin mask (mirrored by page side) and region filtering."""

from __future__ import annotations

import numpy as np

from ..config import Margins, PageConfig
from .base import Box, Region, merge_overlapping


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
    ex, ey = cfg.edge_touch * page_w, cfg.edge_touch * page_h
    for r in regions:
        if r.dropped:
            continue
        b = r.box
        if r.kind in cfg.drop_kinds:
            r.dropped = f"kind:{r.kind}"
        elif cfg.edge_touch and (b.x0 <= ex or b.y0 <= ey or b.x1 >= page_w - ex or b.y1 >= page_h - ey):
            r.dropped = "edge"  # text cut by the page edge: facing page / scan border
        elif r.box.overlap_ratio(content) < cfg.keep_inside_ratio:
            r.dropped = "margin"
        elif r.box.area < min_area:
            r.dropped = "tiny"
        else:
            if cfg.clip_to_content:
                r.box = r.box.intersect(content)
            if binary_inv is not None and r.kind != "table" and ink_ratio(binary_inv, r.box) < 0.003:
                r.dropped = "empty"
    dedupe(regions)
    return regions


def dedupe(regions: list[Region], inside: float = 0.8) -> None:
    """Layout models may emit a block *and* its parts (e.g. a two-line title three times).

    A region holding two or more same-kind regions is dropped as a container; otherwise the
    smaller region nested in a same-kind one is dropped as a duplicate.
    """
    kept = [r for r in regions if r.dropped is None]
    children = {
        id(b): [a for a in kept if a is not b and a.kind == b.kind and a.box.overlap_ratio(b.box) >= inside]
        for b in kept
    }
    for b in kept:
        if len(children[id(b)]) >= 2:
            b.dropped = "container"
    for b in kept:
        if b.dropped:
            continue
        for a in children[id(b)]:
            if a.dropped is None and a.box.area <= b.box.area:
                a.dropped = "duplicate"


def assign_groups(regions: list[Region], frames: list[Box]) -> list[Box]:
    """Put kept regions whose centre lies in a frame into that frame's group.

    Frames that are really tables or figures (one such region covers most of the frame) or that
    hold no text are discarded. Returns the frames actually used, re-indexed.
    """
    pictures = [r.box for r in regions if r.kind in ("figure", "ornament")]
    frames = merge_overlapping([
        f for f in frames if sum(f.intersect(p).area for p in pictures) < 0.3 * f.area
    ])
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
