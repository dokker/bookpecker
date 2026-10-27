"""Hunspell-backed dictionary checks: diacritic repair, dehyphenation guard, quality metrics.

Talks to one long-running `hunspell -a` process (ispell pipe protocol) per dictionary; answers
are cached. OCR engines confuse the Hungarian long/short and umlaut vowels (PP-OCR reads ű/ő as
ú/ó, Tesseract sometimes drops accents), so an unknown word is replaced only when changing at
most two vowels within their confusion class yields exactly one dictionary word.
"""

from __future__ import annotations

import itertools
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Iterable

WORD_RE = re.compile(r"[^\W\d_]+")
TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
# vowels an OCR engine mixes up; a letter may only become another member of its class
CLASSES = ["aá", "eé", "ií", "oóöő", "uúüű"]
CONFUSABLE = {c: cls.replace(c, "") for cls in CLASSES for c in cls}
# accents a latin-script model may output that Hungarian never uses → any vowel of the class
FOREIGN = {"a": "àâäãāă", "e": "èêëēė", "i": "ìîïī", "o": "òôõōø", "u": "ùûūů", "y": "ýÿ"}
for base, letters in FOREIGN.items():
    cls = next((c for c in CLASSES if base in c), base)
    CONFUSABLE |= {f: cls for f in letters}
CONFUSABLE |= {c.upper(): v.upper() for c, v in CONFUSABLE.items()}
MAX_CHANGES = 2
MAX_SUFFIX = 6  # letters a book-specific word may be inflected with


class Speller:
    def __init__(self, dictionary: str = "hu_HU", words: Iterable[str] = ()):
        exe = shutil.which("hunspell")
        if exe is None:
            raise RuntimeError("spell step needs the hunspell CLI and dictionary "
                               "(Arch/Manjaro: sudo pacman -S hunspell hunspell-hu), "
                               "or set spell: {enabled: false}")
        self._proc = subprocess.Popen(
            [exe, "-a", "-i", "UTF-8", "-d", dictionary],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        banner = self._proc.stdout.readline()
        if not banner.startswith("@"):
            err = self._proc.stderr.read() if self._proc.poll() is not None else banner
            raise RuntimeError(f"hunspell -d {dictionary} failed: {err.strip()}")
        self._cache: dict[str, bool] = {}
        self.extra = {w.lower() for w in words}

    def known(self, word: str) -> bool:
        if not word or self._extra_known(word.lower()):
            return True
        hit = self._cache.get(word)
        if hit is None:
            self._proc.stdin.write("^" + word + "\n")  # ^ = never treat the line as a command
            answers = []
            while (line := self._proc.stdout.readline().rstrip("\n")) != "":
                answers.append(line)
            # *: in dictionary, +: via affix, -: compound; &/#: misspelled
            hit = bool(answers) and all(a[:1] in "*+-" for a in answers)
            self._cache[word] = hit
        return hit

    def _extra_known(self, lw: str) -> bool:
        """Book-specific words and their inflected forms ("mentalior" → "mentaliorban")."""
        return lw in self.extra or any(lw.startswith(e) and len(lw) - len(e) <= MAX_SUFFIX
                                       for e in self.extra if len(e) >= 4)

    def fix_word(self, word: str) -> str:
        """`word` with confused vowels repaired, or unchanged when unknown/ambiguous."""
        if len(word) < 3 or self.known(word):
            return word
        slots = [i for i, c in enumerate(word) if c in CONFUSABLE]
        for n in range(1, MAX_CHANGES + 1):
            found = set()
            for pos in itertools.combinations(slots, n):
                for repl in itertools.product(*(CONFUSABLE[word[i]] for i in pos)):
                    chars = list(word)
                    for i, c in zip(pos, repl):
                        chars[i] = c
                    cand = "".join(chars)
                    if self.known(cand):
                        found.add(cand)
                if len(found) > 1:
                    return word  # ambiguous: leave it to the reader
            if found:
                return found.pop()
        return word

    def fix_text(self, text: str) -> str:
        return WORD_RE.sub(lambda m: self.fix_word(m.group()), text)

    def unknown_words(self, text: str) -> tuple[int, list[str]]:
        """(number of checked words, the unknown ones) – words of 3+ letters only.

        Hyphenated tokens count as one word; a suffix after an abbreviation ("TÉ-jét", "KM-mel")
        is not checked.
        """
        n, unknown = 0, []
        for token in TOKEN_RE.findall(text):
            parts = token.split("-")
            if parts[0].isupper() and len(parts[0]) <= 4:
                parts = parts[:1]
            parts = [p for p in parts if len(p) >= 3]
            if not parts:
                continue
            n += 1
            if not all(self.known(p) for p in parts):
                unknown.append(token)
        return n, unknown


@lru_cache(maxsize=4)
def _speller(dictionary: str, words: tuple[str, ...]) -> Speller:
    return Speller(dictionary, words)


def get_speller(cfg) -> Speller | None:
    """Shared Speller for a SpellCfg, or None when the step is switched off."""
    if not cfg.enabled:
        return None
    return _speller(cfg.dictionary, tuple(sorted(cfg.words)))
