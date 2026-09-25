import pytest

from bookpecker.config import Book, parse_pages
from bookpecker.layout.filters import content_box


def make_book(tmp_path, **raw):
    raw.setdefault("pdf", "x.pdf")
    book = Book(tmp_path, "t", raw)
    book._page_count = 100
    return book


def test_parse_pages():
    assert parse_pages(["1-3", 7, "9,11"]) == {1, 2, 3, 7, 9, 11}
    assert parse_pages("98-", last=100) == {98, 99, 100}
    assert parse_pages(None) == set()
    with pytest.raises(ValueError):
        parse_pages("5-3")


def test_parity_and_skip_keep_original_side(tmp_path):
    book = make_book(tmp_path, first_page_side="right", skip_pages=["1-2"])
    assert book.side_of(1) == "right"
    assert book.side_of(2) == "left"
    assert book.side_of(3) == "right"  # skipping pages does not shift parity
    assert book.pages("1-5") == [3, 4, 5]
    left_first = make_book(tmp_path, first_page_side="left")
    assert left_first.side_of(1) == "left"


def test_page_overrides_merge(tmp_path):
    book = make_book(tmp_path, columns=2, margins={"top": 0.1},
                     page_overrides={"10-12": {"columns": 1, "margins": {"bottom": 0.2}},
                                     "12": {"side": "left"}})
    assert book.page_config(9).columns == 2
    cfg = book.page_config(11)
    assert cfg.columns == 1
    assert cfg.margins.top == 0.1 and cfg.margins.bottom == 0.2
    assert book.page_config(11).side == "right"
    assert book.page_config(12).side == "left"


def test_unknown_key_fails_fast(tmp_path):
    with pytest.raises(Exception):
        make_book(tmp_path, colums=2)


def test_margins_mirror_by_side():
    from bookpecker.config import Margins

    m = Margins(top=0.1, bottom=0.1, inner=0.05, outer=0.2)
    right = content_box(1000, 1000, "right", m)  # spine on the left
    left = content_box(1000, 1000, "left", m)  # spine on the right
    assert (right.x0, right.x1) == (50, 800)
    assert (left.x0, left.x1) == (200, 950)


def test_flat_and_folder_configs(tmp_path):
    from bookpecker.config import list_books, load_book

    (tmp_path / "books" / "b").mkdir(parents=True)
    (tmp_path / "books" / "a.yaml").write_text('pdf: "a.pdf"\n', encoding="utf-8")
    (tmp_path / "books" / "b" / "book.yaml").write_text('pdf: "b.pdf"\n', encoding="utf-8")
    (tmp_path / "books" / "_template.yaml").write_text('pdf: ""\n', encoding="utf-8")
    assert list_books(tmp_path) == ["a", "b"]
    assert load_book(tmp_path, "a").pdf == tmp_path / "books" / "a.pdf"
    assert load_book(tmp_path, "b").pdf == tmp_path / "books" / "b" / "b.pdf"


def test_new_book_fills_template_lines():
    from bookpecker.cli import _set_yaml_value

    tpl = 'title: ""\npdf: ""                 # comment\nfoo: 1\n'
    out = _set_yaml_value(_set_yaml_value(tpl, "pdf", "/x/Codex alapkönyv.pdf"), "title", "Codex")
    assert out == 'title: "Codex"\npdf: "/x/Codex alapkönyv.pdf"                 # comment\nfoo: 1\n'
    assert _set_yaml_value(tpl, "title", "Old: BOOK").startswith('title: "Old: BOOK"')
