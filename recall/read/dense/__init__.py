"""Dense second stage of the gate: a small local sentence encoder reranks the BM25 shortlist."""
from .encoder import Unavailable, card_text, cosines, embed, ready, vectors

__all__ = ["Unavailable", "card_text", "cosines", "embed", "ready", "vectors"]
