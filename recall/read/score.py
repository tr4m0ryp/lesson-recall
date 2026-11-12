"""Keyword ranking of cards (BM25) and the gate that decides whether a prompt fires.

A keyword scorer is enough here: the research found the right memory in the top 10 every
time, and BM25 no worse than a small dense encoder at top-1. The picking is left to the
model that reads the injected cards.
"""
import math
import re

STOP = set("""a an the and or but if then else of to in on at by for with from into over under as is are was
were be been being it its this that these those i you he she we they me my your our their not no do does did
done have has had can could should would will shall may might must just also so than too very there here what
which who whom when where why how all any some each other more most such only own same s t now please make
use using get got want need like let one two also about up out off again still de het een en of maar als dan
van naar in op bij voor met uit over is zijn was waren ben bent het dit dat deze die ik jij je u hij zij we
wij ze mijn jouw niet geen wel ook nog al om te er hier daar wat wie waar hoe kan kun moet moeten wil zou
hebben heb heeft had doe doen gedaan maak maken even graag""".split())
_TOKEN = re.compile(r"[a-z0-9_]+")
K1, B = 1.2, 0.75


def _stem(tok):
    if len(tok) > 4:
        for suffix in ("ing", "es", "ed", "s"):
            if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
                return tok[: -len(suffix)]
    return tok


def tokens(text):
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in STOP]


def card_tokens(card):
    """Trigger and referent are the retrieval key, so they count twice."""
    key = f"{card.get('trigger', '')} {card.get('referent', '')}"
    return tokens(key) * 2 + tokens(card.get("lesson", ""))


class Index:
    def __init__(self, cards):
        self.cards = list(cards)
        self.docs = []
        df = {}
        for card in self.cards:
            tf = {}
            for tok in card_tokens(card):
                tf[tok] = tf.get(tok, 0) + 1
            self.docs.append(tf)
            for tok in tf:
                df[tok] = df.get(tok, 0) + 1
        n = len(self.docs)
        self.avg = (sum(sum(tf.values()) for tf in self.docs) / n) if n else 1.0
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, text, k=10):
        """Return [(score, matched terms, card)] best first."""
        query = set(tokens(text))
        hits = []
        for tf, card in zip(self.docs, self.cards):
            length = sum(tf.values())
            score, matched = 0.0, []
            for tok in query:
                f = tf.get(tok)
                if f:
                    score += self.idf[tok] * f * (K1 + 1) / (f + K1 * (1 - B + B * length / self.avg))
                    matched.append(tok)
            if matched:
                hits.append((score, sorted(matched), card))
        hits.sort(key=lambda h: -h[0])
        return hits[:k]


def passes(hits, cfg):
    """The gate: the best card must score high enough on enough distinct terms."""
    return bool(hits) and hits[0][0] >= cfg["gate_score"] and len(hits[0][1]) >= cfg["gate_terms"]
