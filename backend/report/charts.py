"""Report charts drawn with matplotlib from the SAME data the page plots
(training/curves.py, explanations, curves, clustering, bibliometrics), so the PDF
and the screen can't disagree. Each function returns draw(ax) for report.layout."""

import numpy as np

PALETTE = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2", "#dc2626", "#4b5563", "#be185d",
           "#65a30d", "#0d9488"]


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=9, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def confusion(c):
    def draw(ax):
        m = np.array(c["confusion"], float)
        share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)          # seaborn's whitegrid style (set by eda) would draw lines over the cells
        labels = [str(x)[:12] for x in c["classes"]]
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        if len(labels) <= 12:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if share[i, j] > 0.5 else "black")
        _style(ax, "Confusion matrix (rows = actual)", "predicted", "actual")
    return draw


def curve(series, xk, yk, score, title, xl, yl, diagonal=False):
    def draw(ax):
        for k, s in enumerate(series):
            ax.plot(s[xk], s[yk], color=PALETTE[k % 10], lw=1.4, label=f"{s['label']} ({score} {s[score]:.3f})")
        if diagonal:
            ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6.5, frameon=False)
        _style(ax, title, xl, yl)
    return draw


def calibration(series):
    def draw(ax):
        for k, s in enumerate(series):
            ax.plot(s["predicted"], s["observed"], marker="o", ms=3, color=PALETTE[k % 10], lw=1.2,
                    label=f"{s['label']} (Brier {s['brier']:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.8)
        ax.legend(fontsize=6.5, frameon=False)
        _style(ax, "Calibration", "mean predicted probability", "observed frequency")
    return draw


def actual_vs_predicted(c):
    def draw(ax):
        ax.scatter(c["actual"], c["predicted"], s=5, alpha=0.5, color=PALETTE[0])
        lo, hi = min(min(c["actual"]), min(c["predicted"])), max(max(c["actual"]), max(c["predicted"]))
        ax.plot([lo, hi], [lo, hi], ls="--", color="#999999", lw=0.8)
        _style(ax, "Actual vs predicted (test set)", "actual", "predicted")
    return draw


def residuals(c):
    def draw(ax):
        ax.scatter(c["predicted"], c["residual"], s=5, alpha=0.5, color=PALETTE[2])
        ax.axhline(0, ls="--", color="#999999", lw=0.8)
        _style(ax, "Residuals vs predicted", "predicted", "residual (actual - predicted)")
    return draw


def histogram(c):
    def draw(ax):
        e = c["histogram"]["edges"]
        ax.bar(e[:-1], c["histogram"]["counts"], width=np.diff(e), align="edge", color=PALETTE[3], alpha=0.8)
        _style(ax, "Error distribution", "residual", "rows")
    return draw


def qq(q, title="Normal Q-Q plot of residuals"):
    def draw(ax):
        ax.scatter(q["theoretical"], q["sample"], s=5, color=PALETTE[0])
        x = np.array([min(q["theoretical"]), max(q["theoretical"])])
        ax.plot(x, q["slope"] * x + q["intercept"], ls="--", color="#999999", lw=0.8)
        _style(ax, title, "normal quantiles", "sample quantiles")
    return draw


def bars(labels, values, title, xlabel, errors=None, color=0, top=15):
    def draw(ax):
        lab, val = list(labels)[:top][::-1], list(values)[:top][::-1]
        err = list(errors)[:top][::-1] if errors is not None else None
        ax.barh(range(len(lab)), val, xerr=err, color=PALETTE[color % 10], alpha=0.85)
        ax.set_yticks(range(len(lab)), [str(x)[:30] for x in lab])
        ax.grid(axis="y", visible=False)
        _style(ax, title, xlabel)
    return draw


def lines(x, series, title, xlabel, ylabel):
    """series: [(label, values, sd or None)]"""
    def draw(ax):
        for k, (label, ys, sd) in enumerate(series):
            ys = np.array([np.nan if v is None else v for v in ys], float)
            ax.plot(x, ys, marker="o", ms=3, lw=1.3, color=PALETTE[k % 10], label=label)
            if sd is not None:
                s = np.array([np.nan if v is None else v for v in sd], float)
                ax.fill_between(x, ys - s, ys + s, color=PALETTE[k % 10], alpha=0.15)
        ax.legend(fontsize=6.5, frameon=False)
        _style(ax, title, xlabel, ylabel)
    return draw


def clusters(scatter, title):
    def draw(ax):
        lab = np.array(scatter["label"])
        for k, c in enumerate(sorted(set(lab))):
            m = lab == c
            ax.scatter(np.array(scatter["x"])[m], np.array(scatter["y"])[m], s=6, alpha=0.7,
                       color="#bbbbbb" if c == -1 else PALETTE[k % 10], label="noise" if c == -1 else f"cluster {c}")
        ax.legend(fontsize=6, frameon=False, markerscale=2)
        _style(ax, title, f"component 1 ({scatter['method']})", "component 2")
    return draw


def network(g, title):
    def draw(ax):
        nodes = g["nodes"]
        w = max([e["weight"] for e in g["edges"]] or [1])
        for e in g["edges"]:
            a, b = nodes[e["source"]], nodes[e["target"]]
            ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#999999", lw=0.3 + 1.5 * e["weight"] / w, alpha=0.5)
        smax = max([n["size"] for n in nodes] or [1]) or 1
        for n in nodes:
            ax.scatter(n["x"], n["y"], s=15 + 120 * n["size"] / smax, color=PALETTE[n["community"] % 10], zorder=3)
        for n in sorted(nodes, key=lambda n: -n["size"])[:25]:
            ax.text(n["x"], n["y"], str(n["label"])[:22], fontsize=5.5, ha="center", va="bottom", zorder=4)
        ax.set_xticks([])
        ax.set_yticks([])
        _style(ax, title)
    return draw


def thematic(tm):
    def draw(ax):
        for k, t in enumerate(tm["themes"]):
            ax.scatter(t["centrality"], t["density"], s=30 + t["frequency"] * 3, color=PALETTE[k % 10], alpha=0.7)
            ax.text(t["centrality"], t["density"], t["label"][:30], fontsize=6, ha="center", va="bottom")
        ax.axvline(tm["median_centrality"], ls="--", color="#999999", lw=0.7)
        ax.axhline(tm["median_density"], ls="--", color="#999999", lw=0.7)
        _style(ax, "Thematic map (motor: top right, niche: top left, emerging/declining: bottom left, basic: bottom right)",
               "centrality (links to other themes)", "density (internal links)")
    return draw
