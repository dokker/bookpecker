"""Per-page staged pipeline with content-hash caching.

Every stage's cache key = hash(stage version, the config it depends on, upstream key), stored in
work/<book>/state/NNNN.json. A stage re-runs only when its key changes (or when forced).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import Executor, Future
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
from .spell import Speller, get_speller

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


def region_crop(gray: np.ndarray, layout: PageLayout, r, pad_x: float) -> np.ndarray:
    """Region plus some slack; other regions reaching into the slack are painted white.

    `pad_x` (share of page width) is added on both sides: layout boxes and the margin mask clip
    tightly set line ends and their hyphens – on the spine side, but on warped scans on the
    fore-edge side as well.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = r.box.as_int()
    left = right = int(pad_x * w)
    py = 6
    cx0, cy0, cx1, cy1 = max(0, x0 - left), max(0, y0 - py), min(w, x1 + right), min(h, y1 + py)
    crop = gray[cy0:cy1, cx0:cx1].copy()
    keep = np.zeros(crop.shape, bool)
    keep[max(0, y0 - cy0):y1 - cy0, max(0, x0 - cx0):x1 - cx0] = True
    for o in layout.regions:
        # anything else in the slack – a neighbouring column, picture, page number – is not ours;
        # regions overlapping this one (containers, duplicates) are left alone
        if o is r or o.box.intersect(r.box).area > 0:
            continue
        a, b, c, d = o.box.as_int()
        m = np.zeros(crop.shape, bool)
        m[max(0, b - cy0):max(0, d - cy0), max(0, a - cx0):max(0, c - cx0)] = True
        crop[m & ~keep] = 255
    return crop


def run_ocr(gray: np.ndarray, layout: PageLayout, cfg: PageConfig) -> dict[int, OcrResult]:
    return start_ocr(gray, layout, cfg)()


def start_ocr(gray: np.ndarray, layout: PageLayout, cfg: PageConfig,
              pool: Executor | None = None) -> Callable[[], dict[int, OcrResult]]:
    """OCR every text region; returns a callable that yields the results.

    With a thread `pool`, CPU-only work (Tesseract, also the ensemble's second engine) is
    submitted first so it runs while this thread drives the GPU engines; the returned callable
    then waits for it and merges – it may be called from another thread, so the next page's
    GPU work can start meanwhile. Without a pool everything runs right here.
    """
    jobs = []
    for r in layout.kept():
        if r.kind not in OCR_KINDS:
            continue
        crop = region_crop(gray, layout, r, cfg.crop_pad_x)
        if crop.size == 0:
            continue
        jobs.append((r.id, get_engine(cfg.ocr.engine_for(r.kind), cfg), crop, r.kind))
    background: dict[int, Future] = {}
    if pool is not None:
        for rid, engine, crop, kind in jobs:
            if hasattr(engine, "merge"):  # ensemble: the secondary engine is Tesseract
                if getattr(engine.secondary, "cpu_only", False):
                    background[rid] = pool.submit(engine.secondary.recognize, crop, kind)
            elif getattr(engine, "cpu_only", False):
                background[rid] = pool.submit(engine.recognize, crop, kind)
    foreground: dict[int, OcrResult] = {}
    for rid, engine, crop, kind in jobs:
        if rid not in background:
            foreground[rid] = engine.recognize(crop, kind)
        elif hasattr(engine, "merge"):
            foreground[rid] = engine.primary.recognize(crop, kind)

    def finish() -> dict[int, OcrResult]:
        results: dict[int, OcrResult] = {}
        for rid, engine, _, _ in jobs:
            if rid not in background:
                results[rid] = foreground[rid]
            elif hasattr(engine, "merge"):
                results[rid] = engine.merge(foreground[rid], background[rid].result())
            else:
                results[rid] = background[rid].result()
        return results

    return finish


def process_page(book: Book, page: int, force_from: str | None = None,
                 ocr_engine: str | None = None, until: str = "ocr", log: Log = print,
                 pool: Executor | None = None) -> Callable[[], None] | None:
    """Bring one page's stages up to date (up to `until`).

    With a thread `pool` for CPU-only OCR work, the OCR stage is only started: the returned
    callable waits for it and writes the result (run it on any single thread). Otherwise – and
    whenever OCR was cached or not requested – everything is done on return and None is returned.
    """
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
        "sidebars", "drop_kinds", "keep_inside_ratio", "edge_touch", "min_region_area", "clip_to_content"})
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

    ocr_cfg = cfg.model_dump(include={
        **dict.fromkeys(["ocr", "languages", "tesseract", "paddle_ocr", "paddle_vl", "crop_pad_x"], True),
        "spell": {"enabled", "dictionary"},  # the ensemble engine consults the dictionary
    })
    k_o = _key("ocr", ocr_cfg, k_l)
    if not fresh("ocr", k_o, p.ocr):
        log(f"  p{page:04d} ocr ({cfg.ocr.default})")
        gray = gray if gray is not None else read_gray(p.prep)
        layout = PageLayout.from_dict(_load_json(p.layout))
        finish = start_ocr(gray, layout, cfg, pool)

        def complete() -> None:
            results = finish()
            _save_json(p.ocr, {str(k): v.to_dict() for k, v in results.items()})
            done("ocr", k_o)

        if pool is None:
            complete()
        else:
            return complete
    return None


def assemble_book(book: Book, pages: list[int], log: Log = print) -> Path:
    """Write out/<book>/book.md plus per-page Markdown and JSON for the processed pages.

    Also writes out/<book>/quality.md: pages ranked by share of words unknown to the dictionary
    and by low-confidence OCR lines – where proofreading is most needed.
    """
    all_items: list[asm.Item] = []
    page_dir = book.out_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    per_page: list[tuple[int, list[asm.Item]]] = []
    quality: list[dict] = []
    page_json: dict[int, dict] = {}
    ocr_by_page: dict[int, dict[int, OcrResult]] = {}
    mode = "flat"
    speller = None
    for page in pages:
        p = PagePaths(book, page)
        if not (p.layout.exists() and p.ocr.exists()):
            log(f"  p{page:04d} skipped in assembly (not processed yet)")
            continue
        cfg = book.page_config(page)
        mode = cfg.heading_levels
        speller = get_speller(cfg.spell)
        layout = PageLayout.from_dict(_load_json(p.layout))
        ocr = {int(k): OcrResult.from_dict(v) for k, v in _load_json(p.ocr).items()}
        fixes = [(f.pattern, f.repl) for f in cfg.replacements]
        items = asm.page_items(layout, ocr, book.printed_page(page), fixes, speller,
                               cfg.spell.fix_diacritics, cfg.spell.dehyphenate)
        per_page.append((page, items))
        all_items.extend(items)
        page_json[page] = {
            "page": page, "side": layout.side,
            "regions": [{**r.to_dict(), "text": _region_text(ocr.get(r.id))} for r in layout.regions],
        }
        ocr_by_page[page] = ocr

    known = speller.known if speller is not None else None
    asm.assign_heading_levels(all_items, mode)
    for page, items in per_page:  # copies: merging mutates paragraph text
        merged = _refix(asm.merge_continuations([_copy(i) for i in items], known), speller)
        (page_dir / f"{page:04d}.md").write_text(asm.render(merged), encoding="utf-8")
        q = page_quality(page, merged, ocr_by_page[page], speller)
        quality.append(q)
        _save_json(page_dir / f"{page:04d}.json", {**page_json[page], "quality": q})

    body = asm.render(_refix(asm.merge_continuations(all_items, known), speller))
    out = book.out_dir / "book.md"
    out.write_text(f"# {book.title}\n\n{body}", encoding="utf-8")
    (book.out_dir / "quality.md").write_text(quality_report(book.title, quality), encoding="utf-8")
    return out


def _refix(items: list[asm.Item], speller: Speller | None) -> list[asm.Item]:
    """Words rejoined across a column/page break get the diacritic repair too."""
    if speller is not None:
        for it in items:
            if it.type == "para":
                it.text = speller.fix_text(it.text)
    return items


LOW_CONF = 0.85


def page_quality(page: int, items: list[asm.Item], ocr: dict[int, OcrResult],
                 speller: Speller | None) -> dict:
    lines = [ln for res in ocr.values() for ln in res.lines]
    low = [ln.text for ln in lines if 0 <= ln.conf < LOW_CONF]
    q: dict = {"page": page, "lines": len(lines), "low_conf_lines": len(low)}
    if speller is not None:
        text = "\n".join(it.text for it in items if it.type != "marker")
        n, unknown = speller.unknown_words(text)
        q |= {"words": n, "unknown": len(unknown), "unknown_ratio": round(len(unknown) / n, 4) if n else 0.0,
              "unknown_words": sorted(set(unknown))}
    return q


def quality_report(title: str, quality: list[dict]) -> str:
    """Markdown summary: worst pages first, then the most frequent unknown words."""
    rows = sorted(quality, key=lambda q: (-q.get("unknown_ratio", 0), -q["low_conf_lines"]))
    out = [f"# {title} – OCR minőség", "",
           "Ismeretlen szó: a szótárban nem szereplő (3+ betűs) szavak aránya – OCR-hiba, név vagy "
           "játékkifejezés. A gyakori neveket/kifejezéseket érdemes a `spell.words` listába felvenni.", "",
           "| oldal | szavak | ismeretlen | arány | bizonytalan sorok |", "|---:|---:|---:|---:|---:|"]
    for q in rows:
        out.append(f"| {q['page']} | {q.get('words', '–')} | {q.get('unknown', '–')} | "
                   f"{q.get('unknown_ratio', 0):.1%} | {q['low_conf_lines']}/{q['lines']} |")
    counts = Counter(w for q in quality for w in q.get("unknown_words", []))
    if counts:
        out += ["", "## Leggyakoribb ismeretlen szavak (oldalak száma)", "",
                ", ".join(f"{w} ({n})" for w, n in counts.most_common(80))]
    return "\n".join(out) + "\n"


def _copy(item: asm.Item) -> asm.Item:
    return asm.Item(**vars(item))


def _region_text(res: OcrResult | None) -> str | None:
    if res is None:
        return None
    return res.markdown if res.markdown is not None else "\n".join(ln.text for ln in res.lines)
