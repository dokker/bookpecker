"""Per-page staged pipeline with content-hash caching.

Every stage's cache key = hash(stage version, the config it depends on, upstream key), stored in
work/<book>/state/NNNN.json. A stage re-runs only when its key changes (or when forced).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from . import assemble as asm
from .config import Book, OcrCfg, PageConfig
from .debug import overlay
from .imaging import preprocess, rasterize, read_gray, write_png
from .layout.base import PageLayout
from .layout.filters import assign_groups, content_box, filter_regions
from .layout.frames import binarize_inv, detect_frames
from .layout.heuristic import estimate_em
from .layout.order import order_regions
from .ocr.base import OcrResult, get_engine

STAGES = ["rasterize", "preprocess", "layout", "ocr"]
VERSIONS = {"rasterize": 1, "preprocess": 1, "layout": 1, "ocr": 1}
OCR_KINDS = {"text", "title", "caption", "sidebar", "table"}

Log = Callable[[str], None]


def _key(stage: str, cfg: object, upstream: str) -> str:
    blob = json.dumps({"v": VERSIONS[stage], "cfg": cfg, "up": upstream}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class PagePaths:
    def __init__(self, book: Book, page: int):
        w, n = book.work_dir, f"{page:04d}"
        self.state = w / "state" / f"{n}.json"
        self.raw = w / "raw" / f"{n}.png"
        self.prep = w / "prep" / f"{n}.png"
        self.layout = w / "layout" / f"{n}.json"
        self.debug = w / "debug" / f"{n}.png"
        self.ocr = w / "ocr" / f"{n}.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def build_layout(gray: np.ndarray, page: int, cfg: PageConfig) -> PageLayout:
    h, w = gray.shape
    content = content_box(w, h, cfg.side, cfg.margins)
    bw = binarize_inv(gray)
    if cfg.layout_engine == "doclayout":
        from .layout import doclayout

        regions = doclayout.detect(gray, cfg.doclayout)
        x0, y0, x1, y1 = content.as_int()
        em = estimate_em(bw[y0:y1, x0:x1])
    else:
        from .layout import heuristic

        regions, em = heuristic.detect(gray, content, cfg.heuristic)
    for i, r in enumerate(regions):
        r.id = i
    filter_regions(regions, content, w, h, cfg, bw)
    frames = detect_frames(gray, content, cfg.sidebars, em) \
        if (cfg.sidebars.lines or cfg.sidebars.shading) else []
    frames = assign_groups(regions, frames)
    # engine-labelled sidebars without a detected frame form their own group
    for r in regions:
        if r.dropped is None and r.kind == "sidebar" and r.group is None:
            r.group = len(frames)
            frames.append(r.box)
    order_regions(regions, frames, content, cfg.columns, cfg.full_width_ratio, cfg.sidebars.placement)
    return PageLayout(page, cfg.side, w, h, content, regions, frames)


def run_ocr(gray: np.ndarray, layout: PageLayout, cfg: PageConfig) -> dict[int, OcrResult]:
    results: dict[int, OcrResult] = {}
    h, w = gray.shape
    for r in layout.kept():
        if r.kind not in OCR_KINDS:
            continue
        x0, y0, x1, y1 = r.box.as_int()
        pad = 6
        crop = gray[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        if crop.size == 0:
            continue
        engine = get_engine(cfg.ocr.engine_for(r.kind), cfg)
        results[r.id] = engine.recognize(crop, r.kind)
    return results


def process_page(book: Book, page: int, force_from: str | None = None,
                 ocr_engine: str | None = None, until: str = "ocr", log: Log = print) -> None:
    cfg = book.page_config(page)
    if ocr_engine:
        cfg.ocr = OcrCfg(default=ocr_engine)
    p = PagePaths(book, page)
    state = _load_json(p.state)
    forced = set(STAGES[STAGES.index(force_from):]) if force_from else set()

    def fresh(stage: str, key: str, *outputs: Path) -> bool:
        return stage not in forced and state.get(stage) == key and all(o.exists() for o in outputs)

    def done(stage: str, key: str) -> None:
        state[stage] = key
        _save_json(p.state, state)

    st = book.pdf.stat()
    k_r = _key("rasterize", {"dpi": cfg.dpi, "pdf": [str(book.pdf), st.st_size, st.st_mtime_ns],
                             "page": page}, "")
    if not fresh("rasterize", k_r, p.raw):
        log(f"  p{page:04d} rasterize")
        rasterize(book.pdf, page, cfg.dpi, p.raw)
        done("rasterize", k_r)
    if until == "rasterize":
        return

    k_p = _key("preprocess", cfg.preprocess.model_dump(), k_r)
    if not fresh("preprocess", k_p, p.prep):
        log(f"  p{page:04d} preprocess")
        img, angle = preprocess(read_gray(p.raw), cfg.preprocess)
        write_png(p.prep, img)
        state["skew_deg"] = angle
        done("preprocess", k_p)
    if until == "preprocess":
        return

    layout_cfg = cfg.model_dump(include={
        "side", "margins", "columns", "full_width_ratio", "layout_engine", "heuristic", "doclayout",
        "sidebars", "drop_kinds", "keep_inside_ratio", "min_region_area", "clip_to_content"})
    if layout_cfg["layout_engine"] == "heuristic":
        layout_cfg.pop("doclayout")
    else:
        layout_cfg.pop("heuristic")
    k_l = _key("layout", layout_cfg, k_p)
    gray = None
    if not fresh("layout", k_l, p.layout, p.debug):
        log(f"  p{page:04d} layout ({cfg.layout_engine}, {cfg.side})")
        gray = read_gray(p.prep)
        layout = build_layout(gray, page, cfg)
        _save_json(p.layout, layout.to_dict())
        write_png(p.debug, overlay(gray, layout))
        done("layout", k_l)
    if until == "layout":
        return

    ocr_cfg = cfg.model_dump(include={"ocr", "languages", "tesseract", "paddle_vl"})
    k_o = _key("ocr", ocr_cfg, k_l)
    if not fresh("ocr", k_o, p.ocr):
        log(f"  p{page:04d} ocr ({cfg.ocr.default})")
        gray = gray if gray is not None else read_gray(p.prep)
        layout = PageLayout.from_dict(_load_json(p.layout))
        results = run_ocr(gray, layout, cfg)
        _save_json(p.ocr, {str(k): v.to_dict() for k, v in results.items()})
        done("ocr", k_o)


def assemble_book(book: Book, pages: list[int], log: Log = print) -> Path:
    """Write out/<book>/book.md plus per-page Markdown and JSON for the processed pages."""
    all_items: list[asm.Item] = []
    page_dir = book.out_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    per_page: list[tuple[int, list[asm.Item]]] = []
    mode = "flat"
    for page in pages:
        p = PagePaths(book, page)
        if not (p.layout.exists() and p.ocr.exists()):
            log(f"  p{page:04d} skipped in assembly (not processed yet)")
            continue
        mode = book.page_config(page).heading_levels
        layout = PageLayout.from_dict(_load_json(p.layout))
        ocr = {int(k): OcrResult.from_dict(v) for k, v in _load_json(p.ocr).items()}
        items = asm.page_items(layout, ocr, book.printed_page(page))
        per_page.append((page, items))
        all_items.extend(items)
        _save_json(page_dir / f"{page:04d}.json", {
            "page": page, "side": layout.side,
            "regions": [{**r.to_dict(), "text": _region_text(ocr.get(r.id))} for r in layout.regions],
        })

    asm.assign_heading_levels(all_items, mode)
    for page, items in per_page:  # copies: merging mutates paragraph text
        text = asm.render(asm.merge_continuations([_copy(i) for i in items]))
        (page_dir / f"{page:04d}.md").write_text(text, encoding="utf-8")

    body = asm.render(asm.merge_continuations(all_items))
    out = book.out_dir / "book.md"
    out.write_text(f"# {book.title}\n\n{body}", encoding="utf-8")
    return out


def _copy(item: asm.Item) -> asm.Item:
    return asm.Item(**vars(item))


def _region_text(res: OcrResult | None) -> str | None:
    if res is None:
        return None
    return res.markdown if res.markdown is not None else "\n".join(ln.text for ln in res.lines)
