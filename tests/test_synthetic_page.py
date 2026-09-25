"""End-to-end layout on a drawn page: margins, ornaments, mixed columns (no OCR needed)."""

import cv2
import numpy as np

from bookpecker.config import PageConfig
from bookpecker.pipeline import build_layout

W, H = 1700, 2400
WORDS = "a lovag belépett a terembe ahol sárkány várt rá kincsekkel".split()


def text_block(img, x0, y0, x1, n_lines, scale=1.0, thickness=2, line_h=40):
    rng = np.random.default_rng(x0 + y0)
    for i in range(n_lines):
        x, y = x0, y0 + i * line_h + int(28 * scale)
        while True:
            w = str(rng.choice(WORDS))
            (tw, _), _ = cv2.getTextSize(w, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            if x + tw > x1:
                break
            cv2.putText(img, w, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thickness)
            x += tw + int(10 * scale)
    return y0 + n_lines * line_h


def draw_page():
    img = np.full((H, W), 255, np.uint8)
    cv2.putText(img, "ELOHANG - 12. FEJEZET", (600, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)  # running head
    cv2.putText(img, "37", (1500, 2350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)  # page number
    cv2.rectangle(img, (1620, 300), (1690, 2100), 0, -1)  # ornament strip on the outer edge
    cv2.putText(img, "A SARKANY", (600, 240), cv2.FONT_HERSHEY_SIMPLEX, 2.6, 0, 6)  # title
    text_block(img, 150, 330, 780, 10)
    text_block(img, 860, 330, 1500, 10)
    text_block(img, 150, 800, 1500, 5)  # full-width paragraph
    text_block(img, 150, 1100, 780, 20)
    text_block(img, 860, 1100, 1500, 20)
    return img


def test_heuristic_layout_orders_mixed_columns_and_drops_margins():
    cfg = PageConfig(layout_engine="heuristic", columns=2, side="right",
                     margins={"top": 0.05, "bottom": 0.04, "inner": 0.05, "outer": 0.04})
    layout = build_layout(draw_page(), 37, cfg)
    kept = layout.kept()
    assert kept, "no regions found"
    # nothing kept outside the margin box: running head, page number, ornament strip
    for r in kept:
        assert r.box.y0 >= layout.content.y0 - 1 and r.box.y1 <= layout.content.y1 + 1
        assert r.box.x1 <= layout.content.x1 + 1
    assert kept[0].kind == "title"

    def band(r):  # expected reading sequence key
        if r.box.y1 < 300:
            return 0
        if r.box.y1 < 780:
            return 1 if r.box.cx < 820 else 2
        if r.box.y1 < 1080:
            return 3
        return 4 if r.box.cx < 820 else 5

    seq = [band(r) for r in kept]
    assert seq == sorted(seq), seq
    assert set(seq) == {0, 1, 2, 3, 4, 5}
