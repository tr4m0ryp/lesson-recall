"""Turning text into terms, in any script, and the stop words that are dropped from them.

Terms are runs of Unicode word characters after casefolding, so accented and non-Latin words
survive. Scripts written without spaces (Chinese, Japanese kana, Thai, Lao, Khmer, Myanmar)
have no word boundaries to split on; there every run becomes overlapping character bigrams,
the usual cheap stand-in for a word segmenter.

Stop words are learned per user by `cli.py tune`: the terms that occur in a large share of the
user's own typed prompts, whatever their language. Until tune has run, the built-in list
below applies. It is English plus Dutch only, the author's two languages.
"""
import json
import os
import re

from . import paths

BUILTIN_STOP = frozenset("""a an the and or but if then else of to in on at by for with from into over under as is
are was were be been being it its this that these those i you he she we they me my your our their not no do does
did done have has had can could should would will shall may might must just also so than too very there here
what which who whom when where why how all any some each other more most such only own same s t now please make
use using get got want need like let one two also about up out off again still de het een en of maar als dan
van naar in op bij voor met uit over is zijn was waren ben bent het dit dat deze die ik jij je u hij zij we
wij ze mijn jouw niet geen wel ook nog al om te er hier daar wat wie waar hoe kan kun moet moeten wil zou
hebben heb heeft had doe doen gedaan maak maken even graag""".split())
NOSPACE = ("฀-໿က-႟ក-៿぀-ヿ㐀-䶿一-鿿"
           "豈-﫿ｦ-ﾟ")
_WORD = re.compile(r"\w+")
_NOSPACE_RUN = re.compile(f"[{NOSPACE}]+")
_HAS_NOSPACE = re.compile(f"[{NOSPACE}]")
_learned = {}


def _bigrams(run):
    return [run] if len(run) < 2 else [run[i:i + 2] for i in range(len(run) - 1)]


def words(text):
    """Every term of `text`, casefolded, before stop words and stemming."""
    out = []
    for word in _WORD.findall(text.casefold()):
        if not _HAS_NOSPACE.search(word):
            out.append(word)
            continue
        at = 0
        for run in _NOSPACE_RUN.finditer(word):
            if run.start() > at:
                out.append(word[at:run.start()])
            out.extend(_bigrams(run.group()))
            at = run.end()
        if at < len(word):
            out.append(word[at:])
    return out


def stem(tok):
    """Trim a few English suffixes. Cards and prompts go through the same trim, so it only merges."""
    if len(tok) > 4:
        for suffix in ("ing", "es", "ed", "s"):
            if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
                return tok[: -len(suffix)]
    return tok


def stopwords():
    """The learned stop words when tune has written them, else the built-in English and Dutch list."""
    try:
        mtime = os.stat(paths.STOPWORDS).st_mtime_ns
    except FileNotFoundError:
        return BUILTIN_STOP
    if _learned.get("mtime") != mtime:
        with open(paths.STOPWORDS, encoding="utf-8") as f:
            _learned.update(mtime=mtime, words=frozenset(json.load(f)["words"]))
    return _learned["words"]


def has_nospace(text):
    return bool(_HAS_NOSPACE.search(text))
