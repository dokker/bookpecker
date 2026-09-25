from bookpecker.layout.base import Box, Region
from bookpecker.layout.order import detect_gutters, order_regions

CONTENT = Box(100, 100, 1100, 1500)


def R(name, x0, y0, x1, y1, kind="text", group=None):
    r = Region(Box(x0, y0, x1, y1), kind, source=name, group=group)
    return r


def names(regions):
    return [r.source for r in regions]


def test_mixed_full_width_and_two_columns():
    regs = [
        R("B1", 620, 260, 1100, 600),   # right column, first band
        R("T", 300, 100, 900, 180, "title"),
        R("A1", 100, 250, 580, 400),
        R("A2", 100, 420, 580, 620),
        R("W", 100, 650, 1100, 800),    # full-width paragraph
        R("B2", 620, 850, 1100, 1400),
        R("A3", 100, 850, 580, 1400),
    ]
    out = order_regions(regs, [], CONTENT, 2, 0.6)
    assert names(out) == ["T", "A1", "A2", "B1", "W", "A3", "B2"]
    assert [r.order for r in out] == list(range(7))


def test_centred_narrow_title_crossing_gutter_is_spanning():
    regs = [
        R("A", 100, 300, 580, 900), R("B", 620, 300, 1100, 900),
        R("T", 500, 950, 700, 1000, "title"),  # narrow, but bridges the gutter
        R("C", 100, 1050, 580, 1400), R("D", 620, 1050, 1100, 1400),
    ]
    assert names(order_regions(regs, [], CONTENT, 2, 0.6)) == ["A", "B", "T", "C", "D"]


def test_three_columns_auto():
    regs = [R(n, x, 200, x + 300, 1400) for n, x in (("c", 800, ), ("a", 100), ("b", 450))]
    assert names(order_regions(regs, [], CONTENT, "auto", 0.6)) == ["a", "b", "c"]
    assert len(detect_gutters([r.box for r in regs], CONTENT, 0.6)) == 2


def test_single_column():
    regs = [R("b", 100, 500, 1100, 600), R("a", 100, 200, 1100, 300)]
    assert names(order_regions(regs, [], CONTENT, 1, 0.6)) == ["a", "b"]


def test_sidebar_group_inline_and_page_end():
    frames = [Box(620, 300, 1100, 700)]
    def regs():
        return [
            R("A1", 100, 200, 580, 800), R("A2", 100, 820, 580, 1400),
            R("S2", 640, 500, 1080, 680, group=0), R("S1", 640, 320, 1080, 480, "title", group=0),
            R("B1", 620, 720, 1100, 1400),
        ]
    assert names(order_regions(regs(), frames, CONTENT, 2, 0.6)) == ["A1", "A2", "S1", "S2", "B1"]
    assert names(order_regions(regs(), frames, CONTENT, 2, 0.6, "page_end")) == \
        ["A1", "A2", "B1", "S1", "S2"]


def test_dropped_regions_get_no_order():
    a, b = R("a", 100, 200, 1100, 300), R("b", 100, 400, 1100, 500)
    b.dropped = "margin"
    assert names(order_regions([a, b], [], CONTENT, 1, 0.6)) == ["a"]
    assert b.order is None
