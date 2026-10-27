import shutil

import pytest

from bookpecker.ocr.base import OcrLine
from bookpecker.ocr.ensemble import merge_lines, merge_words
from bookpecker.ocr.paddle_ocr import group_rows
from bookpecker.textfix import join_lines

WORDS = {"fejfájást", "gyengébb", "gyorsan", "tűnik", "az", "anyag", "nem", "el", "nagyhatású",
         "ellenmérgek", "ellen", "mérgek", "kard", "mágia", "kontakt", "méreg", "Ki", "vagy",
         "meg", "körül"}


def known(w: str) -> bool:
    return w in WORDS or w.lower() in WORDS


def merged(primary: str, secondary: str) -> str:
    return " ".join(merge_words(primary.split(), secondary.split(), known))


def test_dash_only_secondary_saw_is_inserted():
    assert merged("fejfájást gyengébb, gyorsan", "fejfájást — gyengébb, gyorsan") == \
        "fejfájást – gyengébb, gyorsan"
    assert merged("Ki vagy?", "– Ki vagy?") == "– Ki vagy?"
    assert merged("meg 89 körül", "meg 8-9 körül") == "meg 8-9 körül"


def test_dictionary_picks_accents_and_splits_glued_words():
    assert merged("az anyag nem túnik el", "az anyag nem tűnik el") == "az anyag nem tűnik el"
    assert merged("nagyhatásúellenmérgek,", "nagyhatású ellenmérgek,") == "nagyhatású ellenmérgek,"


def test_secondary_never_adds_words_or_breaks_known_ones():
    assert merged("az anyag el", "az anyag nem el") == "az anyag el"
    assert merged("ellenmérgek", "ellen mérgek") == "ellenmérgek"  # primary already a word
    assert merged("fejfájást gyengébb", "fejfájást gyengéb") == "fejfájást gyengébb"
    assert merged("Veszélyérzet", "Veszé: lyérzet") == "Veszélyérzet"


def test_lines_are_paired_by_row():
    p = [OcrLine("fejfájást gyengébb", [0, 0, 500, 30]), OcrLine("gyorsan", [0, 40, 200, 70])]
    s = [OcrLine("gyorsan", [0, 41, 190, 69]), OcrLine("fejfájást — gyengébb", [2, 1, 498, 31])]
    assert [ln.text for ln in merge_lines(p, s, known)] == ["fejfájást – gyengébb", "gyorsan"]


def test_group_rows_joins_split_fragments():
    boxes = [[300, 0, 500, 30], [0, 2, 250, 31], [0, 40, 400, 70]]
    assert group_rows(boxes) == [[1, 0], [2]]


def test_dictionary_guarded_dehyphenation():
    assert join_lines("a kard-", "mágia", known) == "a kard-mágia"  # "kardmágia" unknown
    assert join_lines("az ellen-", "mérgek", known) == "az ellenmérgek"
    assert join_lines("a gyen-", "gébb", known) == "a gyengébb"  # halves are not words
    assert join_lines("a kard-", "mágia") == "a kardmágia"  # no dictionary: old rule


@pytest.mark.skipif(shutil.which("hunspell") is None, reason="hunspell CLI not installed")
def test_speller_repairs_only_unambiguous_vowel_confusions():
    from bookpecker.spell import Speller

    try:
        sp = Speller("hu_HU")
    except RuntimeError:
        pytest.skip("hu_HU dictionary not installed")
    assert sp.fix_text("Az anyag nem túnik el, egyszerúen.") == "Az anyag nem tűnik el, egyszerűen."
    assert sp.fix_text("Abryss városa") == "Abryss városa"  # unknown name stays
    n, unknown = sp.unknown_words("A mérgező Abryss")
    assert n == 2 and unknown == ["Abryss"]


def test_similar_known_secondary_word_replaces_misread():
    WORDS.add("lőfegyver")
    assert merged("egy lőfegóver", "egy lőfegyver") == "egy lőfegyver"
    assert merged("az kard", "az kerd") == "az kard"  # known primary word is never replaced


def test_line_end_hyphen_from_secondary():
    assert merged("és kevésbé al.", "és kevésbé al-") == "és kevésbé al-"
    assert merged("különböző lehe", "különböző lehe-") == "különböző lehe-"


def test_unsure_primary_line_accepts_less_similar_known_word():
    WORDS.add("tűrés-próbát")
    assert merged("türbátt.", "tűrés-próbát.") == "türbátt."  # confident line: keep
    assert " ".join(merge_words(["türbátt."], ["tűrés-próbát."], known, low_conf=True)) == "tűrés-próbát."


def test_digit_lookalikes():
    assert merged("csak l-szer lehet", "csak 1-szer lehet") == "csak 1-szer lehet"
    assert merged("csak l-szer", "csak l-szer") == "csak l-szer"


def test_non_hungarian_letters_yield_to_secondary():
    assert merged("Abrýss szerte", "Abryss szerte") == "Abryss szerte"


@pytest.mark.skipif(shutil.which("hunspell") is None, reason="hunspell CLI not installed")
def test_book_words_and_their_inflections_are_known():
    from bookpecker.spell import Speller

    try:
        sp = Speller("hu_HU", ["mentalior"])
    except RuntimeError:
        pytest.skip("hu_HU dictionary not installed")
    assert sp.known("mentalior") and sp.known("mentaliorokat") and sp.known("Mentaliornak")
    assert not sp.known("mentaliorxxxxxxxx")
