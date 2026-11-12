"""PP-OCRv5 + Tesseract word-level merge, arbitrated by the dictionary.

PP-OCRv5 reads the condensed spine-side lines and line-end hyphens far better, so it provides
the text and line geometry. Its latin recogniser has no en dash and sometimes glues two words
together; Tesseract gets those right. Each primary line is aligned word by word with the
Tesseract words on the same visual line (by accent-free keys), then:

- a standalone dash Tesseract saw between two aligned words is inserted (as "–");
- where the keys agree but the primary word is unknown and Tesseract's is a word, Tesseract's
  spelling wins (accents);
- where an unknown primary token equals several Tesseract tokens letter for letter (glued
  words), Tesseract's split wins;
- dashes inside a token that only Tesseract saw ("8-9" vs "89") and line-end hyphens the
  primary read as "." or lost ("al." vs "al-") are taken over;
- a misread primary word ("lőfegóver") gives way to a similar Tesseract word when only the
  latter is in the dictionary – with a looser similarity on lines the primary was unsure of;
- "l"/"1", "O"/"0" confusions go to Tesseract when it read a digit.

Anything else keeps the primary reading: Tesseract only ever *repairs*, never adds words.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Callable

import numpy as np

from .base import OcrLine, OcrResult

Known = Callable[[str], bool]
DASH_RE = re.compile(r"^[-–—]+$")
CORE_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")  # hyphenated compounds count as one
SIMILAR = 0.75  # min. letter similarity for a known secondary word to replace an unknown one
SIMILAR_LOW_CONF = 0.5  # …when the primary engine itself was unsure about the line
LOW_CONF = 0.85
DIGIT_LOOKALIKES = str.maketrans("lIioO", "11100")


def key(token: str) -> str:
    """Accent- and punctuation-free lowercase form used for alignment."""
    nfd = unicodedata.normalize("NFD", token.lower())
    return "".join(c for c in nfd if c.isalnum())


def _core(token: str) -> str | None:
    """The single letter run of a token ("mérget," → "mérget"), None if not exactly one."""
    runs = CORE_RE.findall(token)
    return runs[0] if len(runs) == 1 else None


HUNGARIAN = set("aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz-")


def _foreign(word: str) -> bool:
    """Has a letter outside the Hungarian alphabet (latin model output like ā, ô, ý)."""
    return any(c not in HUNGARIAN for c in word.lower())


def _undash(token: str) -> str:
    return re.sub(r"[-–—]", "", token)


def _clean_split(tokens: list[str], known: Known | None) -> bool:
    """Secondary pieces of a glued primary word: plain words (punctuation only at the very end),
    at least one of them in the dictionary – "Veszé: lyérzet" is noise, not a split."""
    if known is None:
        return False
    cores = [_core(t) for t in tokens]
    if any(c is None for c in cores) or any(t != c for t, c in zip(tokens[:-1], cores)):
        return False
    return tokens[-1].startswith(cores[-1]) and any(known(c) for c in cores)


def _all_known(tokens: list[str], known: Known | None) -> bool:
    if known is None:
        return False
    cores = [_core(t) for t in tokens]
    return all(c is not None and known(c) for c in cores)


def merge_words(primary: list[str], secondary: list[str], known: Known | None,
                low_conf: bool = False) -> list[str]:
    """Primary tokens of one line, repaired with the secondary engine's tokens.

    `low_conf`: the primary engine was unsure about this line – a known secondary word may
    replace an unknown primary one at a lower letter similarity.
    """
    similar = SIMILAR_LOW_CONF if low_conf else SIMILAR
    sec = [t for t in secondary if key(t)]
    # dash_before[k]: a standalone dash precedes secondary keyed token k (k == len → line end)
    dash_before: dict[int, bool] = {}
    n = 0
    for t in secondary:
        if key(t):
            n += 1
        elif DASH_RE.match(t):
            dash_before[n] = True
    pk, sk = [key(t) for t in primary], [key(t) for t in sec]
    sm = difflib.SequenceMatcher(None, pk, sk, autojunk=False)
    out: list[str] = []
    # primary index → secondary index for aligned tokens, to place dashes
    aligned: dict[int, int] = {}
    pieces: list[tuple[int, list[str]]] = []  # (first primary index, replacement tokens)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for d in range(i2 - i1):
                p, s = primary[i1 + d], sec[j1 + d]
                pc, sc = _core(p), _core(s)
                if known is not None and pc and sc and pc != sc and not known(pc) and known(sc):
                    p = p.replace(pc, sc, 1)
                elif pc and sc and _foreign(pc) and not _foreign(sc):
                    p = p.replace(pc, sc, 1)  # "Abrýss" vs "Abryss": no ý in Hungarian
                elif _undash(s) == _undash(p) and len(s) > len(p):
                    p = s  # dash the primary model cannot output: "89" vs "8-9"
                elif s.endswith("-") and not p.endswith(("-", "–", "—")) and pc and pc == sc:
                    p = p.rstrip(".,;:") + "-"  # line-end hyphen read as "." or lost
                aligned[i1 + d] = j1 + d
                pieces.append((i1 + d, [p]))
        elif op == "replace" and "".join(pk[i1:i2]) == "".join(sk[j1:j2]) \
                and (i2 - i1) < (j2 - j1) and _clean_split(sec[j1:j2], known) \
                and not _all_known(primary[i1:i2], known):
            pieces.append((i1, sec[j1:j2]))
            aligned[i1] = j1
            aligned[i2 - 1] = j2 - 1
        elif op == "replace" and i2 - i1 == j2 - j1:
            # same token count, different letters: misreads – a known secondary word may win
            for d in range(i2 - i1):
                p, s = primary[i1 + d], sec[j1 + d]
                pc, sc = _core(p), _core(s)
                if known is not None and pc and sc and len(pc) >= 3 and not known(pc) and known(sc) \
                        and difflib.SequenceMatcher(None, pc.lower(), sc.lower()).ratio() >= similar:
                    p = p.replace(pc, sc, 1)
                    aligned[i1 + d] = j1 + d
                elif any(c.isdigit() for c in s) and pk[i1 + d].translate(DIGIT_LOOKALIKES) == sk[j1 + d]:
                    p = s  # "l-szer" vs "1-szer"
                    aligned[i1 + d] = j1 + d
                pieces.append((i1 + d, [p]))
        else:
            pieces.extend((i, [primary[i]]) for i in range(i1, i2))
    for idx, (i, toks) in enumerate(pieces):
        j = aligned.get(i)
        prev_i = pieces[idx - 1][0] if idx else None
        # dash between this token and the previous one, both aligned and adjacent in secondary
        if j is not None and dash_before.get(j):
            first = idx == 0 and j == 0
            prev_j = aligned.get(prev_i) if prev_i is not None else None
            between = prev_j is not None and prev_j == j - 1 and not DASH_RE.match(out[-1] if out else "")
            if (first or between) and not toks[0].startswith(("-", "–", "—")):
                out.append("–")
        out.extend(toks)
    if primary and sec and aligned.get(len(primary) - 1) == len(sec) - 1 and dash_before.get(len(sec)):
        if not out[-1].endswith(("-", "–", "—")):
            out.append("–")
    return out


def _same_row(a: list[float], b: list[float]) -> bool:
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap >= 0.5 * min(a[3] - a[1], b[3] - b[1])


def merge_lines(primary: list[OcrLine], secondary: list[OcrLine], known: Known | None) -> list[OcrLine]:
    out = []
    for ln in primary:
        partners = sorted((s for s in secondary if len(s.box) == 4 and len(ln.box) == 4
                           and _same_row(ln.box, s.box)), key=lambda s: s.box[0])
        if partners:
            words = merge_words(ln.text.split(), " ".join(s.text for s in partners).split(), known,
                                low_conf=0 <= ln.conf < LOW_CONF)
            ln = OcrLine(" ".join(words), ln.box, ln.par, ln.conf)
        out.append(ln)
    return out


class EnsembleEngine:
    name = "ensemble"

    def __init__(self, primary, secondary, known: Known | None):
        self.primary, self.secondary, self.known = primary, secondary, known

    def recognize(self, image: np.ndarray, kind: str) -> OcrResult:
        return self.merge(self.primary.recognize(image, kind), self.secondary.recognize(image, kind))

    def merge(self, p: OcrResult, s: OcrResult) -> OcrResult:
        """Combine the two engines' results (the parallel pipeline runs them separately)."""
        return OcrResult(self.name, merge_lines(p.lines, s.lines, self.known))
