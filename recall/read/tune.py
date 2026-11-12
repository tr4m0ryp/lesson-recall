"""Adapt the keyword side to this user: learn stop words from their own typed prompts. No model call.

A stop word is a term that occurs in at least `stop_df` of the user's prompts (10% by default).
Such a term is matched by almost every prompt, so it cannot tell cards apart, whatever language
it belongs to. The built-in English and Dutch list stays the fallback until there are
`stop_min_prompts` prompts to learn from.

The language estimate is a heuristic, not a classifier: a prompt counts as English when at least
15% of its terms are common English function words. It exists so that setup can decide whether
the English-only encoder fits this user.
"""
import collections
import json
import os
import time

from ..core import paths, text
from ..write import select as stage0
from .replay import prompts

ENGLISH = frozenset("the to and is it this that of in for with you can not be on what how do are "
                    "was have if or but my we so".split())
ENGLISH_SHARE, ENGLISH_MIN = 0.15, 3   # per prompt: function-word share, and the fewest terms to judge
MOSTLY_ENGLISH = 0.6                   # share of judged prompts above which the English encoder is kept


def learn(texts, cfg):
    """Pure part of tune: statistics and the learned stop words for a list of prompt texts."""
    df, total, non_ascii, judged, english = collections.Counter(), 0, 0, 0, 0
    for t in texts:
        terms = [w for w in text.words(t) if len(w) > 1]
        total += len(terms)
        non_ascii += sum(1 for w in terms if not w.isascii())
        df.update(set(terms))
        if len(terms) >= ENGLISH_MIN:
            judged += 1
            english += sum(1 for w in terms if w in ENGLISH) >= ENGLISH_SHARE * len(terms)
    n = len(texts)
    floor = cfg["stop_df"] * n
    ranked = sorted((w for w, c in df.items() if c >= floor), key=lambda w: (-df[w], w))
    share = round(english / judged, 3) if judged else None
    return {
        "prompts": n,
        "learned": n >= cfg["stop_min_prompts"],
        "words": ranked[: cfg["stop_max"]],
        "df": {w: round(df[w] / n, 3) for w in ranked[:20]},
        "non_ascii_token_share": round(non_ascii / total, 3) if total else None,
        "english_prompt_share": share,
        "judged_prompts": judged,
    }


def _current():
    try:
        with open(paths.STOPWORDS, encoding="utf-8") as f:
            return json.load(f)["words"]
    except FileNotFoundError:
        return None


def suggestion(share):
    if share is None:
        return "none yet: no prompts to judge"
    return "encoder" if share >= MOSTLY_ENGLISH else "bm25: the encoder is English-only, run dense off"


def tune():
    """Learn and save this user's stop words. Idempotent: the same prompts give the same file."""
    cfg = paths.config()
    files, _ = stage0.candidates(cfg["extra_roots"])
    stats = learn([t for _, _, t in prompts(files)], cfg)
    before = _current()
    changed = {}
    if stats["learned"] and stats["words"] != before:
        paths.ensure()
        tmp = paths.STOPWORDS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"at": int(time.time()), "prompts": stats["prompts"], "stop_df": cfg["stop_df"],
                       "words": stats["words"]}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, paths.STOPWORDS)
        changed["stopwords"] = {"from": "built-in" if before is None else f"{len(before)} learned",
                                "to": f"{len(stats['words'])} learned"}
    share = stats["english_prompt_share"]
    out = {
        "prompts_read": stats["prompts"],
        "stopwords": ("learned" if stats["learned"] else
                      f"built-in English and Dutch list kept: fewer than {cfg['stop_min_prompts']} prompts"),
        "stopwords_learned": len(stats["words"]) if stats["learned"] else 0,
        "top_stopwords": stats["df"] if stats["learned"] else {},
        "non_ascii_token_share": stats["non_ascii_token_share"],
        "english_prompt_share": share,
        "language_estimate": ("unknown" if share is None else
                              "mostly English" if share >= MOSTLY_ENGLISH else "mostly not English"),
        "language_method": "heuristic: share of prompts whose terms are at least 15% common English function words",
        "gate_suggestion": suggestion(share),
        "card_language": cfg["card_language"],
        "changed": changed,
    }
    paths.log("tune", out)
    return out
