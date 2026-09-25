"""Book configuration: defaults → config/defaults.yaml → books/<slug>/book.yaml → page_overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Side = Literal["left", "right"]
RegionKind = Literal[
    "text", "title", "table", "figure", "sidebar", "caption",
    "header", "footer", "page_number", "ornament",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Margins(_Model):
    """Ignored page edges as a fraction of page width/height.

    `inner` is the spine side, `outer` the fore-edge; they are mirrored by page side.
    """

    top: float = Field(0.05, ge=0, lt=0.5)
    bottom: float = Field(0.05, ge=0, lt=0.5)
    inner: float = Field(0.04, ge=0, lt=0.5)
    outer: float = Field(0.05, ge=0, lt=0.5)


class PreprocessCfg(_Model):
    deskew: bool = True
    max_skew_deg: float = 3.0
    normalize: bool = True  # stretch contrast so paper ≈ white
    flatten: bool = True  # divide out uneven lighting (spine shadow, scanner falloff)


class HeuristicCfg(_Model):
    h_kernel_em: float | None = None  # word joining width in glyph heights; None = from word gaps
    title_ratio: float = 1.45  # glyph height vs page median → title
    figure_big_ink: float = 0.5  # share of ink in "huge" components → figure


class DocLayoutCfg(_Model):
    model_name: str = "PP-DocLayout_plus-L"
    threshold: float = 0.4
    device: str | None = None  # e.g. "cpu", "gpu:0"; None = paddle default


class SidebarCfg(_Model):
    lines: bool = True  # boxes drawn with rules
    shading: bool = True  # tinted/grey background boxes
    shade_delta: int = 22  # how much darker than paper a tint must be
    placement: Literal["inline", "page_end"] = "inline"


class OcrCfg(_Model):
    """Engine per region kind; `default` is used for kinds not listed."""

    default: str = "tesseract"
    text: str | None = None
    title: str | None = None
    caption: str | None = None
    sidebar: str | None = None
    table: str | None = None

    def engine_for(self, kind: str) -> str:
        return getattr(self, kind, None) or self.default


class TesseractCfg(_Model):
    psm: int = 6
    oem: int = 1
    min_word_conf: int = 30  # words without digits below this confidence are dropped (ornament noise)
    edge_junk_conf: int = 70  # 1–2 char tokens at a line end below this are dropped
    tessdata_dir: str | None = None
    extra: str = ""


class PaddleVLCfg(_Model):
    device: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)  # passed to PaddleOCRVL(...)


class Replacement(_Model):
    """Regex fix applied to every paragraph/heading (multiline mode), e.g. OCR confusions."""

    pattern: str
    repl: str


class PageConfig(_Model):
    """Everything that may differ page by page (via page_overrides)."""

    side: Side | None = None  # None → derived from parity
    dpi: int = 300
    preprocess: PreprocessCfg = Field(default_factory=PreprocessCfg)
    margins: Margins = Field(default_factory=Margins)
    columns: int | Literal["auto"] = "auto"
    full_width_ratio: float = 0.6  # wider than this share of the content box → spans columns
    layout_engine: Literal["doclayout", "heuristic"] = "doclayout"
    heuristic: HeuristicCfg = Field(default_factory=HeuristicCfg)
    doclayout: DocLayoutCfg = Field(default_factory=DocLayoutCfg)
    sidebars: SidebarCfg = Field(default_factory=SidebarCfg)
    drop_kinds: list[RegionKind] = Field(
        default_factory=lambda: ["header", "footer", "page_number", "ornament", "figure"]
    )
    keep_inside_ratio: float = 0.5  # min share of a region inside the margin box to keep it
    edge_touch: float = 0.01  # drop regions within this share of a page edge (bleed-through, borders); 0 = off
    min_region_area: float = 0.0004  # share of page area
    clip_to_content: bool = True
    crop_pad_x: float = 0.012  # OCR slack on the spine side of regions (share of page width)
    languages: list[str] = Field(default_factory=lambda: ["hun"])
    ocr: OcrCfg = Field(default_factory=OcrCfg)
    tesseract: TesseractCfg = Field(default_factory=TesseractCfg)
    paddle_vl: PaddleVLCfg = Field(default_factory=PaddleVLCfg)
    heading_levels: Literal["flat", "auto"] = "flat"
    replacements: list[Replacement] = Field(default_factory=list)


BOOK_ONLY_KEYS = {"title", "pdf", "first_page_side", "skip_pages", "page_number_offset", "page_overrides"}


def parse_pages(spec: Any, last: int | None = None) -> set[int]:
    """Parse `[1, "3-5", "8-", "10,12"]` / `"1-4,7"` / `7` into a set of 1-based page numbers."""
    if spec is None:
        return set()
    items = spec if isinstance(spec, list) else [spec]
    pages: set[int] = set()
    for item in items:
        if isinstance(item, int):
            pages.add(item)
            continue
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                start = int(a)
                if b.strip():
                    end = int(b)
                elif last is not None:
                    end = last
                else:
                    raise ValueError(f"open page range {part!r} needs the page count")
                if end < start:
                    raise ValueError(f"bad page range {part!r}")
                pages.update(range(start, end + 1))
            else:
                pages.add(int(part))
    return pages


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Book:
    """A book: its paths, book-level settings and per-page resolved configs."""

    def __init__(self, root: Path, slug: str, raw: dict, config_dir: Path | None = None):
        self.root = root
        self.slug = slug
        self.raw = raw
        self.config_dir = config_dir or root / "books" / slug  # relative `pdf` paths start here
        self.title: str = raw.get("title", slug)
        pdf = raw.get("pdf")
        if not pdf:
            raise ValueError(f"{slug}: book.yaml needs a 'pdf' path")
        pdf_path = Path(pdf).expanduser()
        if not pdf_path.is_absolute():
            candidates = [self.config_dir / pdf_path, root / pdf_path]
            pdf_path = next((c for c in candidates if c.exists()), candidates[0])
        self.pdf = pdf_path
        self.first_page_side: Side = raw.get("first_page_side", "right")
        self.page_number_offset: int = int(raw.get("page_number_offset", 0))
        self._skip_spec = raw.get("skip_pages", [])
        self._overrides: list[tuple[Any, dict]] = list((raw.get("page_overrides") or {}).items())
        self._base = {k: v for k, v in raw.items() if k not in BOOK_ONLY_KEYS}
        PageConfig.model_validate(self._base)  # fail fast on typos
        for spec, over in self._overrides:
            PageConfig.model_validate(deep_merge(self._base, over or {}))
            parse_pages(spec, last=10**6)
        self._page_count: int | None = None

    # --- paths ---------------------------------------------------------------------------
    @property
    def work_dir(self) -> Path:
        return self.root / "work" / self.slug

    @property
    def out_dir(self) -> Path:
        return self.root / "out" / self.slug

    # --- pages ---------------------------------------------------------------------------
    @property
    def page_count(self) -> int:
        if self._page_count is None:
            import pymupdf

            with pymupdf.open(self.pdf) as doc:
                self._page_count = doc.page_count
        return self._page_count

    def skipped(self) -> set[int]:
        return parse_pages(self._skip_spec, last=self.page_count)

    def pages(self, selection: str | None = None) -> list[int]:
        wanted = parse_pages(selection, last=self.page_count) if selection else set(
            range(1, self.page_count + 1)
        )
        skip = self.skipped()
        return sorted(p for p in wanted if 1 <= p <= self.page_count and p not in skip)

    def side_of(self, page: int) -> Side:
        """Parity side of a PDF page (1-based). Skipped pages still count."""
        first_right = self.first_page_side == "right"
        odd = page % 2 == 1
        return "right" if odd == first_right else "left"

    def page_config(self, page: int) -> PageConfig:
        merged = self._base
        for spec, over in self._overrides:
            if page in parse_pages(spec, last=self._page_count or 10**6):
                merged = deep_merge(merged, over or {})
        cfg = PageConfig.model_validate(merged)
        if cfg.side is None:
            cfg.side = self.side_of(page)
        return cfg

    def printed_page(self, page: int) -> int:
        return page + self.page_number_offset


# --- loading -------------------------------------------------------------------------------

def find_root(start: Path | None = None) -> Path:
    env = os.environ.get("BOOKPECKER_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for d in [here, *here.parents]:
        if (d / "books").is_dir():
            return d
    return here


def load_defaults(root: Path) -> dict:
    path = root / "config" / "defaults.yaml"
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def config_path(root: Path, slug: str) -> Path | None:
    """books/<slug>.yaml or books/<slug>/book.yaml."""
    for path in (root / "books" / f"{slug}.yaml", root / "books" / slug / "book.yaml"):
        if path.exists():
            return path
    return None


def load_book(root: Path, slug: str) -> Book:
    path = config_path(root, slug)
    if path is None:
        raise FileNotFoundError(f"no book config books/{slug}.yaml or books/{slug}/book.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Book(root, slug, deep_merge(load_defaults(root), raw), path.parent)


def list_books(root: Path) -> list[str]:
    books = root / "books"
    if not books.is_dir():
        return []
    slugs = {p.stem for p in books.glob("*.yaml")} | {p.parent.name for p in books.glob("*/book.yaml")}
    return sorted(s for s in slugs if not s.startswith("_"))
