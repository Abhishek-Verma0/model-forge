"""Server-rendered EDA charts (matplotlib/seaborn) as base64 PNG data URIs.

The frontend just shows <img src=...>. seaborn does hist/box/heatmap/pairplot in
one line each -- far less code than hand-rolled SVG, and PNG export is free.

ponytail: renders a bounded default set. Target-aware charts (scatter/violin by
the chosen target) are deferred -- they'd need the target passed per request.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")            # headless backend -- REQUIRED off the main thread (web server)
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns            # noqa: E402

from profiler import is_numeric  # noqa: E402

sns.set_theme(style="whitegrid")

MAX_NUMERIC = 8       # cap per-column charts so a wide dataset doesn't render 100 images
MAX_MISSMAP_COLS = 40 # missing-value map gets unreadable past this many columns
MAX_PAIRPLOT = 5      # pairplot is O(k^2) panels
SCATTER_SAMPLE = 5000 # sample rows before plotting points


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)               # MUST close every figure or matplotlib leaks memory per request
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_charts(df):
    """Return [{title, image(dataURI)}] for a bounded default EDA set."""
    numeric = [c for c in df.columns if is_numeric(df[c])][:MAX_NUMERIC]
    charts = []

    # distribution + box, per numeric column
    for col in numeric:
        clean = df[col].dropna()
        if clean.empty:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.4))
        sns.histplot(clean, kde=True, ax=ax1)
        ax1.set_title(f"{col} — distribution")
        sns.boxplot(x=clean, ax=ax2)
        ax2.set_title(f"{col} — box")
        charts.append({"title": col, "image": _png(fig)})

    # missing-value map -- where the gaps are, not just how many (skip if none)
    na = df.isna()
    if na.to_numpy().any():
        sub = df.loc[:, na.any()] if na.any().sum() > MAX_MISSMAP_COLS else df
        sub = sub.iloc[:, :MAX_MISSMAP_COLS]
        rows = sub.sample(SCATTER_SAMPLE, random_state=0).sort_index() if len(sub) > SCATTER_SAMPLE else sub
        fig, ax = plt.subplots(figsize=(min(2 + 0.4 * sub.shape[1], 10), 3))
        sns.heatmap(rows.isna(), cbar=False, cmap="viridis", yticklabels=False, ax=ax)
        ax.set_title("Missing values (light = missing)")
        charts.append({"title": "Missing-value map", "image": _png(fig)})

    # correlation heatmap (>= 2 numeric)
    if len(numeric) >= 2:
        side = min(1 + len(numeric), 8)
        fig, ax = plt.subplots(figsize=(side, side * 0.8))
        sns.heatmap(df[numeric].corr(numeric_only=True), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
        ax.set_title("Correlation")
        charts.append({"title": "Correlation heatmap", "image": _png(fig)})

    # pair plot (few numeric cols; sampled rows)
    if 2 <= len(numeric) <= MAX_PAIRPLOT:
        sample = df[numeric].dropna()
        if len(sample) > SCATTER_SAMPLE:
            sample = sample.sample(SCATTER_SAMPLE, random_state=0)
        if not sample.empty:
            grid = sns.pairplot(sample, corner=True, plot_kws={"s": 8, "alpha": 0.4})
            grid.figure.suptitle("Pair plot (sampled)", y=1.02)
            charts.append({"title": "Pair plot", "image": _png(grid.figure)})

    return charts


if __name__ == "__main__":
    import pandas as pd
    d = pd.DataFrame({"a": range(50), "b": [x * 2 for x in range(50)], "c": list("xy") * 25})
    out = render_charts(d)
    assert out and all(c["image"].startswith("data:image/png;base64,") for c in out)
    titles = [c["title"] for c in out]
    assert "Missing-value map" not in titles, "no gaps -> no map"
    d.loc[0, "a"] = None
    assert "Missing-value map" in [c["title"] for c in render_charts(d)]
    print(f"eda self-check passed ({len(out)} charts)")