from bookpecker import assemble as asm
from bookpecker.layout.base import Box, PageLayout, Region
from bookpecker.ocr.base import OcrLine, OcrResult


def res(*texts, h=30):
    return OcrResult("t", [OcrLine(t, [0, i * 40, 1000, i * 40 + h], 0) for i, t in enumerate(texts)])


def page(n, regions):
    return PageLayout(n, "right", 1200, 1600, Box(0, 0, 1200, 1600), regions)


def reg(i, kind="text", order=0, group=None):
    return Region(Box(0, i * 100, 1000, i * 100 + 90), kind, id=i, order=order, group=group)


def test_paragraph_continues_across_page_break_marker_after():
    p1 = page(1, [reg(0, "title", 0), reg(1, order=1)])
    p2 = page(2, [reg(0, order=0)])
    items = asm.page_items(p1, {0: res("A kezdet"), 1: res("A lovag belépett a")}) \
        + asm.page_items(p2, {0: res("terembe és leült.")})
    asm.assign_heading_levels(items, "flat")
    md = asm.render(asm.merge_continuations(items))
    assert md == ("<!-- page 1 -->\n\n## A kezdet\n\nA lovag belépett a terembe és leült.\n\n"
                  "<!-- page 2 -->\n")


def test_heading_stops_continuation_and_sidebar_is_quoted():
    p = page(3, [reg(0, order=0), reg(1, "title", 1, group=0), reg(2, order=2, group=0),
                 reg(3, order=3)])
    items = asm.page_items(p, {0: res("mondat közepe"), 1: res("Tipp"),
                               2: res("Keretes szöveg."), 3: res("folytatódik itt.")}, printed=5)
    md = asm.render(asm.merge_continuations(items))
    assert "<!-- page 5 (pdf 3) -->" in md
    assert "> **Tipp**\n>\n> Keretes szöveg." in md
    assert "mondat közepe folytatódik itt." in md  # sidebar between halves does not break it


def test_heading_levels_auto_and_flat():
    items = [asm.Item("heading", "Fejezet", 1, height=60), asm.Item("heading", "Alfejezet", 1, height=40),
             asm.Item("heading", "Alfejezet 2", 1, height=41)]
    asm.assign_heading_levels(items, "auto")
    assert [i.level for i in items] == [1, 2, 2]
    asm.assign_heading_levels(items, "flat")
    assert [i.level for i in items] == [2, 2, 2]


def test_html_table_to_md():
    html = "<table><tr><th>Dobás</th><th>Eredmény</th></tr><tr><td>1-2</td><td>Ork | vezér</td></tr></table>"
    assert asm.html_table_to_md(html) == "| Dobás | Eredmény |\n| --- | --- |\n| 1-2 | Ork \\| vezér |"
    spanned = "<table><tr><td colspan='2'>x</td></tr></table>"
    assert asm.html_table_to_md(spanned) == spanned


def test_replacements_fix_percent_confusion():
    fixes = [(r"^(\d{1,3})(?:96|90|06)(?=\s*[-–—])", r"\1%")]
    assert asm.apply_replacements("2096 — Ismered a mérgeket.", fixes) == "20% — Ismered a mérgeket."
    assert asm.apply_replacements("10090 — Magas szintű.", fixes) == "100% — Magas szintű."
    assert asm.apply_replacements("Az 1896 — év", fixes) == "Az 1896 — év"
