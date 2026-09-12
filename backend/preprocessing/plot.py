"""On-demand chart rendering for the frontend's chart builder.

One seaborn call per chart type, driven by (kind, x, y, hue), returned as PNG
bytes. Same matplotlib gotchas as eda.py -- Agg backend, close every figure.
"""

import io

import matplotlib
matplotlib.use("Agg")            # headless -- required off the main thread (web server)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402

sns.set_theme(style="whitegrid")

SAMPLE = 5000        # point-heavy charts sample this many rows
MAX_BAR_CATS = 50    # a bar chart of 745k unique ids would melt -- refuse it

# For box/violin the user's chosen column is the numeric distribution (seaborn y)
# and hue is the group (seaborn x).
_KINDS = {
    "histogram": lambda df, x, y, hue, ax: sns.histplot(data=df, x=x, hue=hue, ax=ax),
    "density":   lambda df, x, y, hue, ax: sns.kdeplot(data=df, x=x, hue=hue, fill=True, ax=ax),
    "box":       lambda df, x, y, hue, ax: sns.boxplot(data=df, x=hue, y=x, ax=ax),
    "violin":    lambda df, x, y, hue, ax: sns.violinplot(data=df, x=hue, y=x, ax=ax),
    "scatter":   lambda df, x, y, hue, ax: sns.scatterplot(data=df, x=x, y=y, hue=hue, s=12, alpha=0.5, ax=ax),
    "line":      lambda df, x, y, hue, ax: sns.lineplot(data=df.sort_values(x), x=x, y=y, hue=hue, ax=ax),
    "bar":       lambda df, x, y, hue, ax: sns.countplot(data=df, x=x, hue=hue, ax=ax),
    "pie":       lambda df, x, y, hue, ax: _pie(df, x, ax),   # class distribution / category share
    "tsne":      lambda df, x, y, hue, ax: _tsne(df, hue, ax),  # 2-D map of all numeric cols
}
KINDS = list(_KINDS)
_MEDIA = {"png": "image/png", "svg": "image/svg+xml"}
_NO_X = {"tsne"}     # kinds that use the whole numeric matrix, not a chosen column
_NEEDS_Y = {"scatter", "line"}  # kinds plotting one column against another


def _pie(df, x, ax):
    counts = df[x].value_counts().head(MAX_BAR_CATS)
    ax.pie(counts.values, labels=[str(v) for v in counts.index], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")


def _tsne(df, hue, ax):
    """2-D t-SNE of all numeric columns, optionally colored by `hue`. For
    visualization only (transductive -- no train/test transform)."""
    from sklearn.manifold import TSNE
    num = df.select_dtypes("number").dropna()
    if num.shape[1] < 2:
        raise ValueError("t-SNE needs at least 2 numeric columns.")
    perp = min(30, max(5, len(num) // 4))
    emb = TSNE(n_components=2, random_state=0, perplexity=perp).fit_transform(num)
    hue_vals = df.loc[num.index, hue] if hue else None
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=hue_vals, s=14, alpha=0.6, ax=ax)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")


def render(df, kind, x=None, y=None, hue=None, fmt="png"):
    """Render one chart -> (bytes, media_type). Validates inputs; samples big/point
    charts. fmt = png | svg (SVG is vector, for print/publication)."""
    if kind not in _KINDS:
        raise ValueError(f"Unknown chart '{kind}'. One of: {', '.join(KINDS)}")
    if fmt not in _MEDIA:
        raise ValueError(f"Unknown format '{fmt}'. Use png or svg.")
    for col in (x, y, hue):
        if col and col not in df.columns:
            raise ValueError(f"No column '{col}' in this dataset.")
    if kind in _NEEDS_Y:
        if not (x and y):
            raise ValueError(f"{kind.capitalize()} needs both X and Y columns.")
    elif kind not in _NO_X and not x:
        raise ValueError(f"'{kind}' needs a column.")
    if kind in ("bar", "pie") and x and df[x].nunique(dropna=True) > MAX_BAR_CATS:
        raise ValueError(f"'{x}' has too many categories for a {kind} chart (>{MAX_BAR_CATS}).")

    data = df
    if kind == "tsne" and len(df) > 1500:
        data = df.sample(1500, random_state=0)  # t-SNE is O(n^2)-ish -- cap hard
    elif kind in ("scatter", "density", "line") and len(df) > SAMPLE:
        data = df.sample(SAMPLE, random_state=0)  # ponytail: sample so 745k points don't hang the render

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        _KINDS[kind](data, x, y, hue, ax)
        ax.set_title(_title(kind, x, y, hue))
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=90, bbox_inches="tight")
        return buf.getvalue(), _MEDIA[fmt]
    finally:
        plt.close(fig)   # always close, even if seaborn raised


def _title(kind, x, y, hue):
    t = f"{kind}: {y} vs {x}" if kind in _NEEDS_Y else f"{kind}: {x}"
    return t + (f" by {hue}" if hue else "")


if __name__ == "__main__":
    import pandas as pd
    d = pd.DataFrame({"a": range(40), "b": [i % 5 for i in range(40)], "g": list("xy") * 20})
    for k in KINDS:
        col = "g" if k in ("bar", "pie") else "a"     # pie/bar want a categorical
        img, media = render(d, k, x=col, y=("b" if k in _NEEDS_Y else None), hue="g")
        assert img[:4] == b"\x89PNG" and media == "image/png", k
    for k in _NEEDS_Y:                                # both axes are required
        try:
            render(d, k, x="a")
            raise AssertionError(f"{k} without y should have been rejected")
        except ValueError:
            pass
    svg, media = render(d, "bar", x="g", fmt="svg")
    assert b"<svg" in svg[:200] and media == "image/svg+xml"
    print(f"plot self-check passed ({len(KINDS)} kinds, png+svg)")