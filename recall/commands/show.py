"""Inspection and the switch: query, stats, cards, replay, mode."""
import collections
import glob
import json
import os

from ..core import paths, store


def cmd_query(args):
    from ..read import hook
    cards, _ = store.load()
    fires, hits, info = hook.select(args.text, cards, hook.project_key(args.cwd), paths.config())
    print(f"fires: {fires}  {json.dumps(info)}")
    for score, matched, card in hits[:5]:
        print(f"{score:6.2f} {matched}\n       {hook.line(card, cards)}")


def cmd_mode(args):
    print(json.dumps(paths.set_config(mode=args.mode), indent=2))


def cmd_stats(_):
    cards, done = store.load()
    count = collections.Counter()
    for c in cards.values():
        count["status:" + c["status"]] += 1
        count["scope:" + c.get("scope", "")] += 1
        count["kind:" + c.get("kind", "")] += 1
        count["recurring"] += c["count"] > 1
    eps = [json.load(open(p)) for p in glob.glob(os.path.join(paths.CACHE, "*.json"))]
    pairs = collections.Counter()
    if os.path.exists(paths.PAIRS):
        pairs.update(json.loads(line)["class"] for line in open(paths.PAIRS))
    cfg = paths.config()
    print(json.dumps({
        "mode": cfg["mode"], "gate": "dense" if cfg["dense"] else "bm25", "store": paths.HOME,
        "cards": len(cards), "episodes_done": len(done),
        "episodes_extracted": len(eps), "episodes_no_lesson": sum(not e["cards"] for e in eps),
        "cards_proposed": sum(e["raw"] for e in eps), "cards_dropped_unverified": sum(e["dropped"] for e in eps),
        "extract_cost_usd": round(sum(e["cost"] for e in eps), 4),
        "by": dict(sorted(count.items())), "pairs": dict(pairs)}, indent=2))


def cmd_replay(_):
    from ..read import replay
    print(json.dumps(replay.replay(), indent=2, ensure_ascii=False))


def cmd_cards(args):
    """The newest active cards, one line each: what the store has learned so far."""
    cards, _ = store.load()
    live = sorted(store.active(cards), key=lambda c: (c.get("ts", ""), c["id"]), reverse=True)
    for card in live[: args.limit]:
        print(f"[{card.get('kind') or 'card'}, {card.get('scope', '')}] {card['referent']}: "
              f"{card['lesson']}\n    applies when: {card['trigger']}")
    print(f"{min(args.limit, len(live))} of {len(live)} active cards")
