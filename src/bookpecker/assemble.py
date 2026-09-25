"""Ordered, OCR'd regions → Markdown (per page and whole book)."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

from . import textfix
from .layout.base import PageLayout
from .ocr.base import OcrResult


@dataclass
class Item:
    type: str  # para | heading | table | caption | marker
    text: str
    page: int
    group: tuple[int, int] | None = None  # (page, frame id) for sidebar content
    height: float = 0.0  # median glyph height of headings, for heading levels
    printed: int | None = None
    level: int = 2


# --- tables ----------------------------------------------------------------------------------

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self.spans = False
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            a = dict(attrs)
            if int(a.get("rowspan") or 1) > 1 or int(a.get("colspan") or 1) > 1:
                self.spans = True
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_md(html: str) -> str:
    """Simple HTML tables → Markdown; tables with merged cells stay HTML."""
    if "<table" not in html.lower():
        return html.strip()
    p = _TableParser()
    p.feed(html)
    rows = [r for r in p.rows if r]
    if p.spans or not rows:
        return html.strip()
    width = max(len(r) for r in rows)
    rows = [[c.replace("|", "\\|") for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


# --- items -----------------------------------------------------------------------------------

def _median_height(res: OcrResult) -> float:
    hs = sorted(ln.box[3] - ln.box[1] for ln in res.lines if len(ln.box) == 4)
    return float(hs[len(hs) // 2]) if hs else 0.0


def page_items(layout: PageLayout, ocr: dict[int, OcrResult], printed: int | None = None) -> list[Item]:
    items = [Item("marker", "", layout.page, printed=printed)]
    for r in layout.kept():
        res = ocr.get(r.id)
        if res is None:
            continue
        group = (layout.page, r.group) if r.group is not None else None
        if r.kind == "table":
            if res.markdown:
                text = html_table_to_md(res.markdown)
            else:  # engine without table support: one row per line
                text = "  \n".join(textfix.clean(ln.text) for ln in res.lines if ln.text.strip())
            if text:
                items.append(Item("table", text, layout.page, group))
        elif r.kind == "title":
            text = textfix.single_line(res.lines)
            if text:
                items.append(Item("heading", text, layout.page, group, _median_height(res)))
        elif r.kind == "caption":
            text = textfix.single_line(res.lines)
            if text:
                items.append(Item("caption", text, layout.page, group))
        else:
            for para in textfix.paragraphs(res.lines):
                items.append(Item("para", para, layout.page, group))
    return items


def merge_continuations(items: list[Item]) -> list[Item]:
    """Join paragraphs split by a column or page break.

    Page markers, captions, tables and sidebars that sat between the two halves end up after the
    joined paragraph; a heading always ends the paragraph.
    """
    out: list[Item] = []
    last: dict[tuple[int, int] | None, Item] = {}  # open paragraph per flow (main text / sidebar)
    for it in items:
        if it.type == "para":
            prev = last.get(it.group)
            if prev is not None and textfix.should_continue(prev.text, it.text):
                prev.text = textfix.join_lines(prev.text, it.text)
                continue
            last[it.group] = it
        elif it.type == "heading":
            last.pop(it.group, None)
        out.append(it)
    return out


def assign_heading_levels(items: list[Item], mode: str) -> None:
    """Set `level` on headings. `auto` clusters glyph heights; otherwise (or if unclear) all ##."""
    heads = [it for it in items if it.type == "heading" and it.group is None]
    for it in heads:
        it.level = 2
    if mode != "auto":
        return
    hs = sorted({round(it.height) for it in heads if it.height > 0})
    if len(hs) < 2:
        return
    clusters: list[list[int]] = [[hs[0]]]
    for h in hs[1:]:
        if h / clusters[-1][-1] > 1.2:
            clusters.append([h])
        else:
            clusters[-1].append(h)
    if not 2 <= len(clusters) <= 3:
        return
    level_of = {}
    for lvl, cl in enumerate(reversed(clusters), start=1):  # biggest → #
        for h in cl:
            level_of[h] = lvl
    for it in heads:
        if it.height > 0:
            it.level = level_of.get(round(it.height), 2)


def render(items: list[Item], markers: bool = True) -> str:
    blocks: list[str] = []
    quote: list[str] = []
    quote_group = None

    def flush():
        nonlocal quote, quote_group
        if quote:
            blocks.append("\n>\n".join("\n".join("> " + ln for ln in b.splitlines()) for b in quote))
        quote, quote_group = [], None

    for it in items:
        if it.type == "marker":
            if markers:
                flush()
                label = f"page {it.printed}" if it.printed is not None and it.printed != it.page \
                    else f"page {it.page}"
                if it.printed is not None and it.printed != it.page:
                    label += f" (pdf {it.page})"
                blocks.append(f"<!-- {label} -->")
            continue
        if it.type == "heading":
            text = f"**{it.text}**" if it.group else "#" * it.level + " " + it.text
        elif it.type == "caption":
            text = f"*{it.text}*"
        else:
            text = it.text
        if it.group is not None:
            if quote_group != it.group:
                flush()
                quote_group = it.group
            quote.append(text)
        else:
            flush()
            blocks.append(text)
    flush()
    return "\n\n".join(blocks).strip() + "\n"
