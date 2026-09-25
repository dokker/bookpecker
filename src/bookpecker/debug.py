"""Debug overlay: margin box, regions coloured by kind, reading order, dropped regions."""

from __future__ import annotations

import cv2
import numpy as np

from .layout.base import PageLayout

COLORS = {  # BGR
    "text": (200, 120, 0), "title": (0, 140, 255), "table": (160, 0, 160),
    "figure": (0, 160, 0), "sidebar": (140, 140, 0), "caption": (0, 200, 200),
    "header": (90, 90, 90), "footer": (90, 90, 90), "page_number": (90, 90, 90),
    "ornament": (90, 90, 90),
}
DROPPED = (0, 0, 220)
FRAME = (180, 60, 220)


def overlay(gray: np.ndarray, layout: PageLayout) -> np.ndarray:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tint = img.copy()
    thick = max(2, gray.shape[1] // 600)
    font = max(0.6, gray.shape[1] / 1600)

    # outside of the margin box greyed out
    c = layout.content
    x0, y0, x1, y1 = c.as_int()
    mask = np.ones(gray.shape, bool)
    mask[max(0, y0):y1, max(0, x0):x1] = False
    tint[mask] = (tint[mask] * 0.5 + np.array([60, 60, 200]) * 0.5).astype(np.uint8)
    img = cv2.addWeighted(img, 0.4, tint, 0.6, 0)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), thick)

    for f in layout.frames:
        a, b, cc, d = f.as_int()
        cv2.rectangle(img, (a, b), (cc, d), FRAME, thick * 2)

    for r in layout.regions:
        a, b, cc, d = r.box.as_int()
        if r.dropped:
            cv2.rectangle(img, (a, b), (cc, d), DROPPED, thick)
            cv2.line(img, (a, b), (cc, d), DROPPED, thick)
            cv2.putText(img, r.dropped, (a + 4, max(b + 20, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        font * 0.6, DROPPED, max(1, thick // 2))
            continue
        col = COLORS.get(r.kind, (0, 0, 0))
        cv2.rectangle(img, (a, b), (cc, d), col, thick)
        label = f"{r.order}:{r.kind}" + (f" g{r.group}" if r.group is not None else "")
        cv2.putText(img, label, (a + 4, b + int(28 * font)), cv2.FONT_HERSHEY_SIMPLEX,
                    font, col, thick)
    return img
