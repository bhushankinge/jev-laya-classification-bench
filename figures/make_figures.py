"""Every README and report figure, drawn only from committed results.

    python3 -m figures.make_figures      # docs/figures/<name>-light.png, <name>-dark.png, social_preview.png

Colors are the dataviz reference palette, slots 1-3, validated all-pairs in light and dark.
Aqua sits below 3:1 on the light surface, so every chart names each series in text too.
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from pipeline.evaluate import wilson_lower

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"
SOURCES = ("jev", "qwen", "laya")
NAMES = {"jev": "Jev (jev-1.13.0)", "qwen": "Qwen3.5-35B-A3B", "laya": "Laya 421M"}
VARIANTS = [f"{q}-{s}" for q in "ABC" for s in ("S1", "S2", "S3")]
QUESTIONS = {"A": "12 presence nouls", "B": "one 12-way choice", "C": "hierarchical + nouls"}
STATES = {"S1": "title + description", "S2": "+ line items", "S3": "+ attachment excerpt"}
CLASSES = ["Hardware", "Software", "Maintenance & Support", "Services",
           "Installation & Integration", "Furniture / Facilities", "Other"]
SHORT = ["Hardware", "Software", "Maint. & Support", "Services", "Install. & Integ.", "Furniture", "Other"]
MIN_CURVE_N = 20         # curve points below this many accepted rows are noise (Laya's first point is n=1)
JEV_COST_12K = "$0.78"   # results/e5-cost.md: Jev list price for all 12,000 rows
THEMES = {
    "light": {"surface": "#ffffff", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e1",
              "jev": "#2a78d6", "qwen": "#eb6834", "laya": "#1baf7a",
              "seq": ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]},
    "dark": {"surface": "#0d1117", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#30302d",
             "jev": "#3987e5", "qwen": "#d95926", "laya": "#199e70",
             "seq": ["#0d1117", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]},
}


def load(results=RESULTS):
    """(e2, e1): the final three-source run and the nine Jev E1 variants' paired primary-class metrics."""
    results = Path(results)
    e2 = json.loads((results / "e2-full" / "metrics.json").read_text())
    e1 = {v: json.loads((results / f"e1-{v}" / "metrics.json").read_text())
          ["by_source"]["jev"]["paired"]["primary_vs_quote_gold"] for v in VARIANTS}
    return e2, e1


def paired(e2, src, key):
    return e2["by_source"][src]["paired"][key]


def hero_values(e2):
    """{source: (primary accuracy, auto-accept coverage at the 95% cutoff or None, fulfillment accuracy)}"""
    return {s: (paired(e2, s, "primary_vs_quote_gold")["accuracy"],
                paired(e2, s, "primary_vs_quote_gold")["coverage_at_cutoff"],
                paired(e2, s, "fulfillment_vs_quote_gold")["accuracy"]) for s in SOURCES}


def wilson(acc, n):
    k = round(acc * n)
    return wilson_lower(k, n), 1 - wilson_lower(n - k, n)


def figure(t, w, h, head, sub, adjust=None, **kw):
    fig, axes = plt.subplots(figsize=(w, h), facecolor=t["surface"], **kw)
    fig.text(0.015, 1 - 0.25 / h, head, color=t["text"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.015, 1 - 0.55 / h, sub, color=t["muted"], fontsize=9.5, va="top")
    fig.subplots_adjust(**{"top": 1 - 1.0 / h, "bottom": 0.14, "left": 0.08, "right": 0.97, **(adjust or {})})
    return fig, axes


def style(ax, t, grid="y"):
    ax.set_facecolor(t["surface"])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=t["muted"], labelcolor=t["muted"], length=0, labelsize=9)
    if grid:
        ax.grid(axis=grid, color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["muted"]); ax.yaxis.label.set_color(t["muted"])


def fig_legend(fig, ax, t):
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3,
               frameon=False, labelcolor=t["text"], fontsize=9)


def ink(rgba):
    """Text color that clears contrast on a filled cell."""
    r, g, b = rgba[:3]
    return "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"


def draw_hero(ax, t, e2, groups=3):
    vals = hero_values(e2)
    n_p = paired(e2, "jev", "primary_vs_quote_gold")["n"]
    n_f = paired(e2, "jev", "fulfillment_vs_quote_gold")["n"]
    labels = [f"Primary-class accuracy\n(n={n_p})", f"Auto-accepted at ≥95% precision\n(Wilson bound, n={n_p})",
              f"Fulfillment-mode accuracy\n(n={n_f})"][:groups]
    w = 0.26
    for i, s in enumerate(SOURCES):
        for g in range(groups):
            v, x = vals[s][g], g + (i - 1) * w
            if v is None:
                ax.text(x, 0.02, "not\nreached", ha="center", va="bottom", fontsize=7.5, color=t["muted"])
                continue
            ax.bar(x, v, w, color=t[s], edgecolor=t["surface"], lw=2, label=NAMES[s] if g == 0 else None)
            ax.text(x, v + 0.015, f"{v:.1%}", ha="center", va="bottom", fontsize=8.5, color=t["text"])
    ax.set_xticks(range(groups), labels)
    ax.set_xlim(-0.5, groups - 0.5); ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style(ax, t)


def hero(t, e2, e1):
    fig, ax = figure(t, 9, 4.8, "Same opportunities, three model paths, graded against real quotes",
                     "Only Jev's confidence supports a 95%-precision auto-accept gate. Fulfillment mode is hard for all three.",
                     adjust={"bottom": 0.22})
    draw_hero(ax, t, e2)
    fig_legend(fig, ax, t)
    return fig


def precision_coverage(t, e2, e1):
    m = paired(e2, "jev", "primary_vs_quote_gold")
    fig, ax = figure(t, 9, 4.8, "Precision vs share auto-accepted, primary class",
                     f"Paired rows with quote gold (n={m['n']}). Curve points backed by fewer than {MIN_CURVE_N} rows omitted.",
                     adjust={"right": 0.80, "bottom": 0.2})
    for s in SOURCES:
        c = [p for p in paired(e2, s, "primary_vs_quote_gold")["curve"] if p["n"] >= MIN_CURVE_N]
        xs, ys = [p["coverage"] for p in c], [p["precision"] for p in c]
        ax.plot(xs, ys, color=t[s], lw=2, solid_capstyle="round", solid_joinstyle="round", label=NAMES[s],
                marker="o" if len(c) < 10 else None, ms=8, mec=t["surface"], mew=2)
        ax.text(xs[-1] + 0.015, ys[-1], NAMES[s], color=t["text"], fontsize=9, va="center", clip_on=False)
    ax.axhline(0.95, color=t["muted"], lw=1)
    ax.text(0.01, 0.947, "95% precision target", color=t["muted"], fontsize=8.5, va="top")
    pt = next(p for p in m["curve"] if p["cutoff"] == m["cutoff_95"])
    ax.plot(pt["coverage"], pt["precision"], "o", ms=10, color=t["jev"], mec=t["surface"], mew=2, zorder=5)
    ax.annotate(f"Jev cutoff {m['cutoff_95']}: {pt['coverage']:.1%} auto-accepted,\n"
                f"{pt['precision']:.1%} precise, Wilson lower bound ≥ 95%",
                (pt["coverage"], pt["precision"]), xytext=(0.30, 0.985), textcoords="data",
                color=t["text"], fontsize=8.5, va="top",
                arrowprops={"arrowstyle": "-", "color": t["muted"], "lw": 1})
    ax.set_xlim(0, 1); ax.set_ylim(0.6, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("share of rows auto-accepted (coverage)"); ax.set_ylabel("precision")
    style(ax, t, grid="both")
    fig_legend(fig, ax, t)
    return fig


def e1_variants(t, e2, e1):
    n = e1["A-S2"]["n"]
    fig, ax = figure(t, 9, 4.8, "Nine Jev prompt variants, one result",
                     f"Primary-class accuracy with 95% Wilson intervals, n={n} each. "
                     "The pre-registered rule selected the cheapest adequate variant.",
                     adjust={"left": 0.41, "right": 0.95, "bottom": 0.1})
    for i, v in enumerate(VARIANTS):
        acc, y = e1[v]["accuracy"], len(VARIANTS) - 1 - i
        lo, hi = wilson(acc, n)
        ax.plot([lo, hi], [y, y], color=t["jev"], lw=2, solid_capstyle="round")
        win = v == "A-S2"
        ax.plot(acc, y, "o", ms=9, mfc=t["jev"] if win else t["surface"], mec=t["jev"], mew=2, zorder=5)
        ax.text(hi + 0.004, y, f"{acc:.1%}" + ("   selected" if win else ""), va="center", fontsize=8.5, color=t["text"])
    ax.set_yticks(range(len(VARIANTS)),
                  [f"{v}   {QUESTIONS[v[0]]} · {STATES[v[2:]]}" for v in reversed(VARIANTS)])
    for s in ("qwen", "laya"):
        a = paired(e2, s, "primary_vs_quote_gold")["accuracy"]
        ax.axvline(a, color=t[s], lw=2)
        ax.text(a - 0.002, len(VARIANTS) - 0.4, f"{NAMES[s]}\n{a:.1%}", color=t["text"], fontsize=8.5, ha="right", va="bottom")
    ax.set_xlim(0.74, 0.97); ax.set_ylim(-0.6, len(VARIANTS) + 0.6)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style(ax, t, grid="x")
    return fig


def agreement(t, e2, e1):
    m = e2["jev_vs_qwen_primary"]
    counts = [[m.get(r, {}).get(c, 0) for c in CLASSES] for r in CLASSES]
    total, agree = sum(map(sum, counts)), sum(counts[i][i] for i in range(len(CLASSES)))
    share = [[x / max(sum(row), 1) for x in row] for row in counts]
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    fig, ax = figure(t, 9, 6.4, "Where Jev and Qwen disagree",
                     f"Primary class on {total:,} rows, {agree / total:.1%} agreement. "
                     "Numbers are row counts; shade is the share of Jev's row.",
                     adjust={"left": 0.2, "bottom": 0.2, "right": 0.95})
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(counts):
        for j, x in enumerate(row):
            ax.text(j, i, f"{x:,}", ha="center", va="center", fontsize=8.5, color=ink(cmap(share[i][j])))
    ax.set_xticks(range(len(SHORT)), SHORT, rotation=30, ha="right")
    ax.set_yticks(range(len(SHORT)), SHORT)
    ax.set_xlabel("Qwen3.5-35B-A3B says"); ax.set_ylabel("Jev says")
    style(ax, t, grid=None)
    ax.set_xticks([x + 0.5 for x in range(len(SHORT) - 1)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(SHORT) - 1)], minor=True)
    ax.grid(which="minor", color=t["surface"], lw=2)            # 2px surface gap between cells
    ax.tick_params(which="minor", length=0)
    return fig


def per_class_f1(t, e2, e1):
    fig, axes = figure(t, 10, 4.8, "Per-class F1: hardware is easy, configured builds are not",
                       "Paired rows with quote gold. Classes with fewer than 40 rows are indicative only.",
                       adjust={"left": 0.19, "wspace": 0.75, "bottom": 0.2}, ncols=2)
    for ax, key, title in ((axes[0], "primary_vs_quote_gold", "Primary class"),
                           (axes[1], "fulfillment_vs_quote_gold", "Fulfillment mode (hardware rows)")):
        gold = paired(e2, "jev", key)["per_class"]
        classes = [c for c, v in gold.items() if v["support"]]
        h = 0.26
        for i, s in enumerate(SOURCES):
            pcs = paired(e2, s, key)["per_class"]
            for j, c in enumerate(classes):
                y, f1 = len(classes) - 1 - j + (1 - i) * h, pcs[c]["f1"]
                ax.barh(y, f1, h, color=t[s], edgecolor=t["surface"], lw=2, label=NAMES[s] if j == 0 else None)
                ax.text(f1 + 0.02, y, f"{f1:.2f}", va="center", fontsize=7.5, color=t["text"])
        ax.set_yticks(range(len(classes)),
                      [f"{SHORT[CLASSES.index(c)] if c in CLASSES else c} (n={gold[c]['support']})" for c in reversed(classes)])
        ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_title(title, loc="left", color=t["text"], fontsize=10)
        style(ax, t, grid="x")
    fig_legend(fig, axes[0], t)
    return fig


def gate(t, e2, e1):
    g = e2["gate"]
    total = g["flags_not_queued"]["n_accepted"] + g["flags_not_queued"]["n_queued"]
    policies = [("flags_not_queued", "Flags recorded,\nnot queued"),
                ("flags_always_queue", "Spec rule: any flag\nsends row to review")]
    fig, axes = figure(t, 9, 3.8, "One queueing rule decides how much gets automated",
                       f"Gate simulation on {total:,} rows at Jev cutoff {g['cutoff']}, Qwen agreement required.",
                       adjust={"left": 0.19, "wspace": 0.12, "bottom": 0.12}, ncols=2, sharey=True)
    for ax, key, title in ((axes[0], "coverage", "Share auto-accepted"),
                           (axes[1], "precision_primary", "Primary precision on scored rows")):
        for y, (p, _) in enumerate(policies):
            v = g[p][key]
            note = f"  n={g[p]['n_primary_scored']}" if key == "precision_primary" else f"  {g[p]['n_accepted']:,} rows"
            ax.barh(y, v, 0.5, color=t["jev"])
            ax.text(v + 0.02, y, f"{v:.1%}{note}", va="center", fontsize=8.5, color=t["text"])
        ax.set_xlim(0, 1.45); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(title, loc="left", color=t["text"], fontsize=10)
        style(ax, t, grid="x")
    axes[0].set_yticks(range(len(policies)), [label for _, label in policies])
    return fig


def social_preview(e2, out):
    t = THEMES["light"]
    v = hero_values(e2)["jev"]
    n = paired(e2, "jev", "primary_vs_quote_gold")["n"]
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=t["surface"])
    fig.text(0.05, 0.88, "jev-laya-classification-bench", fontsize=15, color=t["muted"], family="monospace")
    fig.text(0.05, 0.64, "Typed-decision models vs a 35B LLM\non 12,000 federal IT solicitations",
             fontsize=22, fontweight="bold", color=t["text"], va="bottom", linespacing=1.25)
    fig.text(0.05, 0.56, "Graded against what a reseller actually quoted.", fontsize=15, color=t["muted"])
    for i, line in enumerate([f"{v[0]:.1%} primary-class accuracy (Jev, n={n})",
                              f"{v[1]:.1%} auto-accepted at ≥95% precision",
                              f"{JEV_COST_12K} to label all 12,000 rows"]):
        fig.text(0.05, 0.40 - i * 0.08, line, fontsize=15, color=t["text"])
    ax = fig.add_axes([0.62, 0.24, 0.34, 0.58])
    draw_hero(ax, t, e2, groups=2)
    ax.legend(frameon=False, labelcolor=t["text"], fontsize=10, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=3)
    path = Path(out) / "social_preview.png"
    fig.savefig(path, dpi=100, facecolor=t["surface"])
    plt.close(fig)
    return path


FIGURES = {"hero": hero, "precision_coverage": precision_coverage, "e1_variants": e1_variants,
           "agreement": agreement, "per_class_f1": per_class_f1, "gate": gate}


def main(out=OUT, results=RESULTS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    e2, e1 = load(results)
    written = []
    for theme, t in THEMES.items():
        for name, draw in FIGURES.items():
            fig = draw(t, e2, e1)
            path = out / f"{name}-{theme}.png"
            fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
            plt.close(fig)
            written.append(path)
    written.append(social_preview(e2, out))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render README and report figures from committed results.")
    ap.add_argument("--out", default=OUT, type=Path)
    for p in main(ap.parse_args().out):
        print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
