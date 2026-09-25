"""Turn OCR lines into paragraphs: drop scan line breaks, undo hyphenation.

Rule of thumb: a line break survives only at a paragraph boundary, and never inside a sentence
(the previous line must end with sentence-final punctuation). List items are the one exception.
"""

from __future__ import annotations

import re
import statistics

from .ocr.base import OcrLine

HYPHENS = ("-", "‐", "­", "¬")
DASHES = ("–", "—")
TERMINAL = set(".!?…:;")
CLOSERS = "\"'”’»)]"
# Hungarian suspended compounds: "kis- és nagybetű", "fel- vagy lemegy" – keep the hyphen + space
SUSPENDED_NEXT = {"és", "vagy", "s", "illetve", "valamint", "avagy", "meg", "sem", "ill."}
BULLET_RE = re.compile(r"^[•·▪■◆●○‣*]")  # always a new item, even without final punctuation
# numbered/lettered items and dialogue dashes: new paragraph only after a finished sentence
ITEM_RE = re.compile(r"^([-–—]\s|\d{1,3}[.)]\s|[a-z][.)]\s)")
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean(text: str) -> str:
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    return re.sub(r"[ \t]+", " ", text).strip()


def ends_sentence(text: str) -> bool:
    t = text.rstrip().rstrip(CLOSERS)
    return bool(t) and t[-1] in TERMINAL


def is_bullet(text: str) -> bool:
    return bool(BULLET_RE.match(text.lstrip()))


def is_item(text: str) -> bool:
    return bool(ITEM_RE.match(text.lstrip()))


def join_lines(a: str, b: str) -> str:
    """Join two consecutive text lines, resolving end-of-line hyphenation."""
    a, b = a.rstrip(), b.lstrip()
    if not a:
        return b
    if not b:
        return a
    if a.endswith(HYPHENS) and len(a) > 1 and a[-2].isalpha():
        first = b.split(" ", 1)[0]
        if first.lower() in SUSPENDED_NEXT:
            return f"{a[:-1]}- {b}"
        if b[0].islower():
            return a[:-1] + b
        return a[:-1] + "-" + b  # real compound hyphen: "Kard-" + "Mágia"
    return f"{a} {b}"


def _has_geometry(lines: list[OcrLine]) -> bool:
    return all(len(ln.box) == 4 for ln in lines)


def paragraph_breaks(lines: list[OcrLine]) -> list[bool]:
    """breaks[i] is True when a new paragraph starts at lines[i] (i ≥ 1)."""
    n = len(lines)
    breaks = [False] * n
    if n < 2:
        return breaks
    geo = _has_geometry(lines)
    if geo:
        heights = [ln.box[3] - ln.box[1] for ln in lines]
        H = statistics.median(heights) or 1.0
        gaps = [max(0.0, lines[i + 1].box[1] - lines[i].box[3]) for i in range(n - 1)]
        G = statistics.median(gaps) if gaps else 0.0
        xs0 = sorted(ln.box[0] for ln in lines)
        xs1 = sorted(ln.box[2] for ln in lines)
        left = xs0[len(xs0) // 10]
        right = xs1[(len(xs1) * 9) // 10]
    for i in range(1, n):
        prev, cur = lines[i - 1], lines[i]
        if is_bullet(cur.text):
            breaks[i] = True
            continue
        if not ends_sentence(prev.text):
            continue  # never break inside a sentence
        if is_item(cur.text):
            breaks[i] = True
            continue
        if geo:
            gap = cur.box[1] - prev.box[3]
            big_gap = gap > max(1.8 * G, G + 0.5 * H)
            indented = cur.box[0] - left > 0.9 * H
            short = right - prev.box[2] > 2.0 * H
            breaks[i] = big_gap or indented or short
        else:
            breaks[i] = cur.par != prev.par
    return breaks


def paragraphs(lines: list[OcrLine]) -> list[str]:
    lines = [OcrLine(clean(ln.text), ln.box, ln.par) for ln in lines if clean(ln.text)]
    if not lines:
        return []
    breaks = paragraph_breaks(lines)
    out: list[str] = []
    cur = ""
    for ln, brk in zip(lines, breaks):
        if brk and cur:
            out.append(cur)
            cur = ""
        cur = join_lines(cur, ln.text) if cur else ln.text
    if cur:
        out.append(cur)
    return out


def single_line(lines: list[OcrLine]) -> str:
    """Titles/captions: everything on one line."""
    text = ""
    for ln in lines:
        t = clean(ln.text)
        if t:
            text = join_lines(text, t) if text else t
    return text


def should_continue(prev_para: str, next_para: str) -> bool:
    """Does `next_para` (next column/page) continue the paragraph `prev_para`?"""
    if not prev_para or not next_para:
        return False
    if is_bullet(next_para):
        return False
    return not ends_sentence(prev_para)
