from bookpecker.ocr.base import OcrLine
from bookpecker.textfix import join_lines, paragraphs, should_continue, single_line

H = 30  # line height used in fixtures


def lines(*specs):
    """specs: (text, x0, x1) on consecutive lines, or (text, x0, x1, extra_gap)."""
    out, y = [], 0
    for s in specs:
        text, x0, x1, *gap = s
        y += gap[0] if gap else 0
        out.append(OcrLine(text, [x0, y, x1, y + H], 0))
        y += H + 10
    return out


def test_dehyphenation_rules():
    assert join_lines("a var-", "ázsló ment") == "a varázsló ment"
    assert join_lines("a Kard-", "Mágia") == "a Kard-Mágia"
    assert join_lines("kis-", "és nagybetű") == "kis- és nagybetű"
    assert join_lines("fel-", "vagy lemegy") == "fel- vagy lemegy"
    assert join_lines("egy –", "kettő") == "egy – kettő"
    assert join_lines("egy", "kettő") == "egy kettő"


def test_no_break_inside_sentence_even_with_gap_or_indent():
    ls = lines(("A lovag belépett a", 0, 800), ("terembe, ahol", 60, 500, 40), ("vár rá a sárkány.", 0, 700))
    assert paragraphs(ls) == ["A lovag belépett a terembe, ahol vár rá a sárkány."]


def test_paragraph_by_indent_short_line_and_gap():
    ls = lines(
        ("Első bekezdés első sora, amely hosszú", 0, 1000),
        ("és itt ér véget.", 0, 400),                        # short + terminal → break
        ("Második bekezdés, szintén hosszú sor", 60, 1000),
        ("folytatódik és vége van.", 0, 1000),               # full width + terminal, next indented
        ("Harmadik bekezdés behúzással kezdődik", 60, 1000),
        ("és teljes sorral zárul.", 0, 1000),
        ("Negyedik, nagy térköz után.", 0, 1000, 45),
    )
    assert paragraphs(ls) == [
        "Első bekezdés első sora, amely hosszú és itt ér véget.",
        "Második bekezdés, szintén hosszú sor folytatódik és vége van.",
        "Harmadik bekezdés behúzással kezdődik és teljes sorral zárul.",
        "Negyedik, nagy térköz után.",
    ]


def test_sentence_end_mid_paragraph_is_not_a_break():
    ls = lines(("Ez egy mondat. Ez egy másik mondat, ami", 0, 1000),
               ("tovább tart. Itt pedig egy harmadik", 0, 1000),
               ("mondat kezdődik.", 0, 380))
    assert len(paragraphs(ls)) == 1


def test_bullets_always_break():
    ls = lines(("Felszerelés:", 0, 300), ("• kard", 0, 150), ("• pajzs", 0, 160))
    assert paragraphs(ls) == ["Felszerelés:", "• kard", "• pajzs"]


def test_lines_without_geometry_use_par_ids():
    ls = [OcrLine("Első sor és", [], 0), OcrLine("vége.", [], 0), OcrLine("Új bekezdés.", [], 1),
          OcrLine("Mondat közben", [], 1), OcrLine("nem törünk.", [], 2)]
    assert paragraphs(ls) == ["Első sor és vége.", "Új bekezdés. Mondat közben nem törünk."]


def test_single_line_and_continuation():
    assert single_line([OcrLine("A SÁRKÁNY", [], 0), OcrLine("BARLANGJA", [], 0)]) == "A SÁRKÁNY BARLANGJA"
    assert should_continue("a lovag belépett a", "terembe.")
    assert should_continue("ahol Gandalf", "Szürke várt.")
    assert not should_continue("Vége.", "következő")
    assert not should_continue("felsorolás", "• elem")


def test_tesseract_junk_filters():
    from bookpecker.ocr.tesseract import is_junk_line, strip_edge_junk

    assert is_junk_line(["d", "Z", "H", "2", "s"])
    assert not is_junk_line(["a", "kard", "és", "a", "pajzs"])
    assert strip_edge_junk([("í", 60), ("30%", 13), ("kard", 95), ("[1", 40)], 70) == ["30%", "kard"]
    assert strip_edge_junk([("a", 40), ("kard", 95)], 70) == ["a", "kard"]
