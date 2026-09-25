"""Reading order for pages that mix full-width blocks and N columns.

Regions that cross a column boundary ("spanning") cut the page into horizontal bands. Inside a
band, regions are read column by column (left → right), each column top → bottom. Sidebar groups
are ordered as a single unit and then their members are ordered recursively inside the frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Box, Region


@dataclass
class _Unit:
    box: Box
    members: list[Region]
    group: int | None


def detect_gutters(boxes: list[Box], content: Box, full_width_ratio: float) -> list[float]:
    """x positions of vertical white gaps between narrow blocks (column gutters)."""
    cw = content.w
    if cw <= 0:
        return []
    bins = 400
    cover = [0] * bins
    narrow = [b for b in boxes if b.w < full_width_ratio * cw]
    for b in narrow:
        a = int((max(b.x0, content.x0) - content.x0) / cw * bins)
        z = int((min(b.x1, content.x1) - content.x0) / cw * bins)
        for i in range(max(0, a), min(bins, z + 1)):
            cover[i] += 1
    # a few blocks (e.g. a centred title) may bridge a gutter on busy pages
    allowed = len(narrow) // 10
    gutters: list[float] = []
    lo, hi = int(0.1 * bins), int(0.9 * bins)
    i = lo
    while i < hi:
        if cover[i] <= allowed:
            j = i
            while j < hi and cover[j] <= allowed:
                j += 1
            has_left = any(cover[k] > allowed for k in range(0, i))
            has_right = any(cover[k] > allowed for k in range(j, bins))
            if has_left and has_right and (j - i) >= 2:
                gutters.append(content.x0 + (i + j) / 2 / bins * cw)
            i = j
        else:
            i += 1
    return gutters


def column_bounds(
    boxes: list[Box], content: Box, columns: int | str, full_width_ratio: float
) -> list[float]:
    """Boundaries between columns. Uses detected gutters when they agree with `columns`."""
    detected = detect_gutters(boxes, content, full_width_ratio)
    if columns == "auto":
        return detected
    n = int(columns)
    if n <= 1:
        return []
    if len(detected) == n - 1:
        return detected
    return [content.x0 + content.w * k / n for k in range(1, n)]


def _is_spanning(box: Box, bounds: list[float], content: Box, full_width_ratio: float) -> bool:
    if not bounds:
        return True
    if box.w >= full_width_ratio * content.w:
        return True
    tol = 0.04 * content.w
    return any(box.x0 < b - tol and box.x1 > b + tol for b in bounds)


def _column_of(box: Box, bounds: list[float]) -> int:
    return sum(1 for b in bounds if box.cx > b)


def order_boxes(
    boxes: list[Box], content: Box, columns: int | str, full_width_ratio: float
) -> list[int]:
    """Return indices of `boxes` in reading order."""
    if not boxes:
        return []
    bounds = column_bounds(boxes, content, columns, full_width_ratio)
    if not bounds:
        return sorted(range(len(boxes)), key=lambda i: (boxes[i].y0, boxes[i].x0))

    spanning = sorted(
        (i for i, b in enumerate(boxes) if _is_spanning(b, bounds, content, full_width_ratio)),
        key=lambda i: boxes[i].y0,
    )
    span_set = set(spanning)
    rest = [i for i in range(len(boxes)) if i not in span_set]

    # band k = regions above spanning[k]; last band = below all spanning regions
    bands: list[list[int]] = [[] for _ in range(len(spanning) + 1)]
    for i in rest:
        cy = boxes[i].cy
        k = 0
        while k < len(spanning) and cy > boxes[spanning[k]].cy:
            k += 1
        bands[k].append(i)

    out: list[int] = []
    for k, band in enumerate(bands):
        band.sort(key=lambda i: (_column_of(boxes[i], bounds), boxes[i].y0, boxes[i].x0))
        out.extend(band)
        if k < len(spanning):
            out.append(spanning[k])
    return out


def order_regions(
    regions: list[Region],
    frames: list[Box],
    content: Box,
    columns: int | str,
    full_width_ratio: float,
    sidebar_placement: str = "inline",
) -> list[Region]:
    """Assign `.order` to kept regions (dropped ones get None). Returns kept regions in order."""
    kept = [r for r in regions if r.dropped is None]
    for r in regions:
        r.order = None

    units: list[_Unit] = []
    groups: dict[int, list[Region]] = {}
    for r in kept:
        if r.group is None:
            units.append(_Unit(r.box, [r], None))
        else:
            groups.setdefault(r.group, []).append(r)
    for gid, members in groups.items():
        box = frames[gid] if gid < len(frames) else members[0].box
        for m in members[1:]:
            box = box.union(m.box)
        inner = order_boxes([m.box for m in members], box, "auto", full_width_ratio)
        units.append(_Unit(box, [members[i] for i in inner], gid))

    idx = order_boxes([u.box for u in units], content, columns, full_width_ratio)
    ordered_units = [units[i] for i in idx]
    if sidebar_placement == "page_end":
        ordered_units = [u for u in ordered_units if u.group is None] + [
            u for u in ordered_units if u.group is not None
        ]

    result: list[Region] = []
    for u in ordered_units:
        result.extend(u.members)
    for n, r in enumerate(result):
        r.order = n
    return result
