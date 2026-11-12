"""Render the README figures from data.json.

Docs only: the plugin never imports this file, and matplotlib is not a runtime dependency.
Every plotted value is read from data.json, which names the source of each one.

    python3 docs/figures/plot.py              # PNG (light and dark, 2x) and SVG next to this file
    python3 docs/figures/plot.py --pdf DIR    # also a light PDF per figure into DIR
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "quiet": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926", "quiet": "#52514e"},
}


def style(t):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"], "axes.grid": True,
        "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.labelcolor": t["ink2"], "lines.linewidth": 2, "lines.markersize": 6,
    })


def recall_vs_k(d, t):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    k, rec, ci = d["k"], d["recall_at_k"], d["recall_ci"]
    ax.fill_between(k, [c[0] for c in ci], [c[1] for c in ci], color=t["s1"], alpha=0.15, lw=0)
    ax.plot(k, rec, marker="o", color=t["s1"], label="right memory in the top k (recall@k)")
    lo, hi = d["disambiguation_ci"]
    ax.axhspan(lo, hi, color=t["s2"], alpha=0.12, lw=0)
    ax.plot(k, d["disambiguation"], marker="o", color=t["s2"], label="right memory ranked first (disambiguation)")
    ax.set_xscale("log")
    ax.set_xticks(k, [str(x) for x in k])
    ax.minorticks_off()
    ax.set_ylim(0.45, 1.06)
    ax.set_xlabel("k, memories retrieved per query")
    ax.set_ylabel("share of 360 queries")
    ax.set_title("A larger k finds the right memory, it never ranks it first")
    ax.annotate(f"{rec[3]:.4f} at k=5", (5, rec[3]), (2.6, 0.985), color=t["ink2"], fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": t["muted"], "lw": 0.8})
    ax.annotate(f"{rec[4]:.4f} from k=10", (10, 1.0), (13, 0.955), color=t["ink2"], fontsize=8.5)
    ax.text(50, lo - 0.012, f"{d['disambiguation'][0]:.4f} at every k", ha="right", va="top",
            color=t["ink2"], fontsize=8.5)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55))
    return fig


def separation(d, t):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True)
    s = d["strength"]
    a.plot(s, d["sibling_cosine"], marker="o", color=t["s1"])
    a.set_ylim(0.2, 0.9)
    a.set_ylabel("cosine between lookalike memories")
    a.set_title("(a) The push moves the vectors apart")
    lo, hi = d["control_ci"]
    b.axhspan(lo, hi, color=t["quiet"], alpha=0.35, lw=0)
    b.axhline(d["control"], color=t["muted"], lw=1)
    b.text(0.8, hi + 0.008, f"no push {d['control']:.4f}, 95% interval", ha="right", va="bottom",
           color=t["ink2"], fontsize=8)
    b.plot(s, d["pair_accuracy_m2"], marker="o", color=t["s2"], label="pairs of two lookalikes")
    b.plot(s, d["disambiguation"], marker="o", color=t["s1"], label="all queries")
    b.set_ylim(0.4, 1.0)
    b.set_ylabel("disambiguation")
    b.set_title("(b) The ranking does not follow")
    b.legend(loc="upper right")
    for ax in (a, b):
        ax.set_xlabel("separation strength s")
    fig.tight_layout(w_pad=2.5)
    return fig


def pickers(d, t):
    rows = d["rows"][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    groups = list(dict.fromkeys(r["group"] for r in d["rows"]))
    color = {groups[0]: t["s1"], groups[1]: t["s2"]}
    seen = set()
    for y, r in enumerate(rows):
        lab = None if r["group"] in seen else r["group"]
        seen.add(r["group"])
        ax.errorbar(r["value"], y, xerr=[[r["value"] - r["ci"][0]], [r["ci"][1] - r["value"]]], fmt="o",
                    color=color[r["group"]], ecolor=color[r["group"]], elinewidth=1.5, capsize=0, label=lab)
        ax.text(r["value"], y + 0.22, f"{r['value']:.4f}", ha="center", va="bottom", color=t["ink2"],
                fontsize=8.5)
    c = d["ceiling"]
    ax.axvline(c["value"], color=t["muted"], lw=1, ls=(0, (3, 2)))
    ax.text(c["value"] - 0.005, len(rows) - 0.45, "right memory\nin the top 10", ha="right", va="top",
            color=t["ink2"], fontsize=8)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.45, 1.03)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("disambiguation: right memory picked first (95% interval)")
    ax.set_title("Only a comparison closes the gap")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center right", bbox_to_anchor=(0.93, 0.62), fontsize=8.5)
    fig.tight_layout()
    return fig


def gate(d, t):
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = range(len(rows))
    vals = [r["top_card_relevant"] for r in rows]
    cols = [t["quiet"]] * (len(rows) - 1) + [t["s1"]]
    ax.bar(x, vals, width=0.5, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", color=t["ink"], fontsize=9)
    ax.set_xticks(list(x), [f"{r['label']}\nfires: {r['fires']}" for r in rows])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("top card relevant, share of fires")
    diff = d["difference"]
    ax.annotate("", (2, 0.80), (1, 0.80), arrowprops={"arrowstyle": "-", "color": t["muted"], "lw": 0.8})
    ax.text(1.5, 0.81, f"+{diff['value']:.3f} [{diff['ci'][0]:.3f}, {diff['ci'][1]:.3f}]", ha="center",
            va="bottom", color=t["ink2"], fontsize=8.5)
    ax.set_title(f"Recall gate on {d['prompts']} real past prompts (labels by Haiku)")
    fig.tight_layout()
    return fig


def decision(d, t):
    names, counts = d["classes"][::-1], d["counts"][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.barh(range(len(names)), counts, height=0.55, color=t["s1"])
    notes = {"INTERFERENCE": f" ({d['interference_with_condition']:,} name a condition that tells them apart)",
             "SUPERSESSION": f" ({d['supersessions_confirmed']} confirmed by the second call)"}
    for i, (n, c) in enumerate(zip(names, counts)):
        ax.text(c + 15, i, f"{c:,}" + notes.get(n, ""), va="center", color=t["ink2"], fontsize=8.5)
    ax.set_yticks(range(len(names)), names)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 2900)
    ax.set_xlabel(f"judged pairs, of {d['pairs']:,}")
    ax.set_title("Write decision on real sessions: lookalikes are common, contradictions rare")
    fig.tight_layout()
    return fig


FIGURES = {"recall-vs-k": ("recall_vs_k", recall_vs_k), "separation": ("separation", separation),
           "pickers": ("pickers", pickers), "gate": ("gate", gate), "write-decision": ("decision", decision)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="also write a light PDF of each figure into this directory")
    args = ap.parse_args()
    with open(os.path.join(HERE, "data.json"), encoding="utf-8") as f:
        data = json.load(f)
    for sub in ("png", "svg"):
        os.makedirs(os.path.join(HERE, sub), exist_ok=True)
    if args.pdf:
        os.makedirs(args.pdf, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "lesson-recall"
    plt.rcParams["svg.fonttype"] = "none"
    for name, (key, draw) in FIGURES.items():
        for mode, t in THEMES.items():
            style(t)
            fig = draw(data[key], t)
            fig.savefig(os.path.join(HERE, "png", f"{name}-{mode}.png"), dpi=200, bbox_inches="tight")
            if mode == "light":
                fig.savefig(os.path.join(HERE, "svg", f"{name}.svg"), bbox_inches="tight",
                            metadata={"Date": None})
                if args.pdf:
                    fig.savefig(os.path.join(args.pdf, f"{name}.pdf"), bbox_inches="tight",
                                metadata={"CreationDate": None})
            plt.close(fig)
    print("wrote", ", ".join(FIGURES))


if __name__ == "__main__":
    main()
