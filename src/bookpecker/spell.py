"""Hunspell-backed dictionary checks: diacritic repair, dehyphenation guard, quality metrics.

Looks words up in libhunspell directly (falls back to a `hunspell -a` pipe); answers are cached.
OCR engines confuse the Hungarian long/short and umlaut vowels (PP-OCR reads ű/ő as ú/ó, Tesseract sometimes drops accents), so an unknown word is replaced only when changing at
most two vowels within their confusion class yields exactly one dictionary word.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import itertools
import os
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Iterable

WORD_RE = re.compile(r"[^\W\d_]+")
TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
SUFFIX_AFTER = re.compile(r"[\d%°]-$")  # text up to a token: it is a number's suffix
# vowels an OCR engine mixes up; a letter may only become another member of its class
CLASSES = ["aá", "eé", "ií", "oóöő", "uúüű"]
CONFUSABLE = {c: cls.replace(c, "") for cls in CLASSES for c in cls}
# accents a latin-script model may output that Hungarian never uses → any vowel of the class
FOREIGN = {"a": "àâäãāă", "e": "èêëēė", "i": "ìîïī", "o": "òôõōø", "u": "ùûūů", "y": "ýÿ"}
for base, letters in FOREIGN.items():
    cls = next((c for c in CLASSES if base in c), base)
    CONFUSABLE |= {f: cls for f in letters}
CONFUSABLE |= {c.upper(): v.upper() for c, v in CONFUSABLE.items()}
# abbreviations the tokenizer sees without their full stop ("stb." → "stb")
ABBREVIATIONS = {"stb", "ill", "max", "min", "pl", "kb", "vö", "ún", "db", "uo", "ld", "ford", "jav"}
MAX_CHANGES = 2
MAX_SUFFIX = 6  # letters a book-specific word may be inflected with


class Speller:
    def __init__(self, dictionary: str = "hu_HU", words: Iterable[str] = ()):
        self._check = _LibHunspell.open(dictionary) or _PipeHunspell(dictionary)
        self._cache: dict[str, bool] = {}
        self.extra = {w.lower() for w in words} | ABBREVIATIONS

    def known(self, word: str) -> bool:
        if not word or self._extra_known(word.lower()):
            return True
        hit = self._cache.get(word)
        if hit is None:
            hit = self._cache[word] = self._check(word)
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
        or a number ("5%-kal", "3-nál") is not checked.
        """
        n, unknown = 0, []
        for m in TOKEN_RE.finditer(text):
            if SUFFIX_AFTER.search(text, max(0, m.start() - 2), m.start()):
                continue  # "10%-kal", "k10-zel": a suffix, not a word
            token = m.group()
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


DICT_DIRS = ["/usr/share/hunspell", "/usr/share/myspell/dicts", "/usr/local/share/hunspell",
             os.path.expanduser("~/.local/share/hunspell")]


class _LibHunspell:
    """libhunspell through ctypes: a yes/no lookup in microseconds.

    The `hunspell -a` pipe answers a misspelled word only after computing suggestions – with the
    Hungarian dictionary ~0.1 s per unknown word, and the vowel repair probes many unknown
    candidates. Not thread safe: one Speller is used from one thread at a time.
    """

    def __init__(self, lib, handle, encoding: str):
        self._lib, self._h, self._enc = lib, handle, encoding

    @classmethod
    def open(cls, dictionary: str) -> _LibHunspell | None:
        name = ctypes.util.find_library("hunspell") or ctypes.util.find_library("hunspell-1.7")
        base = dictionary if os.path.sep in dictionary else next(
            (os.path.join(d, dictionary) for d in DICT_DIRS
             if os.path.exists(os.path.join(d, dictionary + ".dic"))), None)
        if name is None or base is None:
            return None
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            return None
        lib.Hunspell_create.restype = ctypes.c_void_p
        lib.Hunspell_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.Hunspell_spell.restype = ctypes.c_int
        lib.Hunspell_spell.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.Hunspell_get_dic_encoding.restype = ctypes.c_char_p
        lib.Hunspell_get_dic_encoding.argtypes = [ctypes.c_void_p]
        handle = lib.Hunspell_create((base + ".aff").encode(), (base + ".dic").encode())
        if not handle:
            return None
        enc = (lib.Hunspell_get_dic_encoding(handle) or b"UTF-8").decode()
        return cls(lib, handle, enc)

    def __call__(self, word: str) -> bool:
        try:
            raw = word.encode(self._enc)
        except (UnicodeEncodeError, LookupError):
            return False
        return bool(self._lib.Hunspell_spell(self._h, raw))


class _PipeHunspell:
    """Fallback: one long-running `hunspell -a` process (ispell pipe protocol)."""

    def __init__(self, dictionary: str):
        exe = shutil.which("hunspell")
        if exe is None:
            raise RuntimeError("spell step needs libhunspell or the hunspell CLI, plus the dictionary "
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

    def __call__(self, word: str) -> bool:
        self._proc.stdin.write("^" + word + "\n")  # ^ = never treat the line as a command
        answers = []
        while (line := self._proc.stdout.readline().rstrip("\n")) != "":
            answers.append(line)
        # *: in dictionary, +: via affix, -: compound; &/#: misspelled
        return bool(answers) and all(a[:1] in "*+-" for a in answers)


@lru_cache(maxsize=4)
def _speller(dictionary: str, words: tuple[str, ...]) -> Speller:
    return Speller(dictionary, words)


def get_speller(cfg) -> Speller | None:
    """Shared Speller for a SpellCfg, or None when the step is switched off."""
    if not cfg.enabled:
        return None
    return _speller(cfg.dictionary, tuple(sorted(cfg.words)))
