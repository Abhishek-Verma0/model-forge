"""Server-rendered EDA charts (matplotlib/seaborn) as base64 PNG data URIs.

The frontend just shows <img src=...>. seaborn does hist/box/heatmap/pairplot in
one line each -- far less code than hand-rolled SVG, and PNG export is free.
Each chart has a key, so one chart can be drawn again on its own for download:
PNG at print resolution or SVG (vector) -- plan §8.3 / §15 "PNG or SVG".

ponytail: renders a bounded default set. Target-aware charts (scatter/violin by
the chosen target) are deferred -- they'd need the target passed per request.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")            # headless backend -- REQUIRED off the main thread (web server)
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns            # noqa: E402

from preprocessing.profiler import is_numeric  # noqa: E402

sns.set_theme(style="whitegrid")

# Chart caps -- all our starting points (display limits, they never change the data):
MAX_NUMERIC = 8       # cap per-column charts so a wide dataset doesn't render 100 images
MAX_MISSMAP_COLS = 40 # missing-value map gets unreadable past this many columns
MAX_PAIRPLOT = 5      # pairplot is O(k^2) panels
SCATTER_SAMPLE = 5000 # sample rows before plotting points
SCREEN_DPI = 90       # our starting point: on-screen preview (small page payload)
DOWNLOAD_DPI = 300    # published: the print resolution journal artwork guidelines commonly ask of figures
MEDIA = {"png": "image/png", "svg": "image/svg+xml"}


def _save(fig, fmt="png", dpi=SCREEN_DPI):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)               # MUST close every figure or matplotlib leaks memory per request
    return buf.getvalue()


def _png(fig):
    return "data:image/png;base64," + base64.b64encode(_save(fig)).decode()


def _figures(df, only=None):
    """Yield (key, title, figure) for the default set, in page order. With `only`,
    just that chart is drawn (the others are skipped before any drawing)."""
    numeric = [c for c in df.columns if is_numeric(df[c])][:MAX_NUMERIC]

    def wanted(key):
        return only is None or only == key

    # distribution + box, per numeric column
    for col in numeric:
        clean = df[col].dropna()
        if clean.empty or not wanted(f"col:{col}"):
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.4))
        sns.histplot(clean, kde=True, ax=ax1)
        ax1.set_title(f"{col} — distribution")
        sns.boxplot(x=clean, ax=ax2)
        ax2.set_title(f"{col} — box")
        yield f"col:{col}", str(col), fig

    # missing-value map -- where the gaps are, not just how many (skip if none)
    na = df.isna()
    if wanted("missing") and na.to_numpy().any():
        sub = df.loc[:, na.any()] if na.any().sum() > MAX_MISSMAP_COLS else df
        sub = sub.iloc[:, :MAX_MISSMAP_COLS]
        rows = sub.sample(SCATTER_SAMPLE, random_state=0).sort_index() if len(sub) > SCATTER_SAMPLE else sub
        fig, ax = plt.subplots(figsize=(min(2 + 0.4 * sub.shape[1], 10), 3))
        sns.heatmap(rows.isna(), cbar=False, cmap="viridis", yticklabels=False, ax=ax)
        ax.set_title("Missing values (light = missing)")
        yield "missing", "Missing-value map", fig

    # correlation heatmap (>= 2 numeric)
    if wanted("correlation") and len(numeric) >= 2:
        side = min(1 + len(numeric), 8)
        fig, ax = plt.subplots(figsize=(side, side * 0.8))
        sns.heatmap(df[numeric].corr(numeric_only=True), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
        ax.set_title("Correlation")
        yield "correlation", "Correlation heatmap", fig

    # pair plot (few numeric cols; sampled rows)
    if wanted("pairplot") and 2 <= len(numeric) <= MAX_PAIRPLOT:
        sample = df[numeric].dropna()
        if len(sample) > SCATTER_SAMPLE:
            sample = sample.sample(SCATTER_SAMPLE, random_state=0)
        if not sample.empty:
            grid = sns.pairplot(sample, corner=True, plot_kws={"s": 8, "alpha": 0.4})
            grid.figure.suptitle("Pair plot (sampled)", y=1.02)
            yield "pairplot", "Pair plot", grid.figure


def render_charts(df):
    """Return [{key, title, image(dataURI)}] for a bounded default EDA set."""
    return [{"key": key, "title": title, "image": _png(fig)} for key, title, fig in _figures(df)]


def render_one(df, key, fmt="png"):
    """One chart of the default set for download -> (bytes, media type, title):
    PNG at DOWNLOAD_DPI, or SVG."""
    if fmt not in MEDIA:
        raise ValueError(f"Unknown format '{fmt}'. Use png or svg.")
    for _key, title, fig in _figures(df, only=key):
        return _save(fig, fmt, DOWNLOAD_DPI), MEDIA[fmt], title
    raise ValueError(f"No chart '{key}' for this table.")


if __name__ == "__main__":
    import pandas as pd
    d = pd.DataFrame({"a": range(50), "b": [x * 2 for x in range(50)], "c": list("xy") * 25})
    out = render_charts(d)
    assert out and all(c["image"].startswith("data:image/png;base64,") for c in out)
    titles = [c["title"] for c in out]
    assert "Missing-value map" not in titles, "no gaps -> no map"
    assert [c["key"] for c in out] == ["col:a", "col:b", "correlation", "pairplot"], [c["key"] for c in out]
    d.loc[0, "a"] = None
    assert "Missing-value map" in [c["title"] for c in render_charts(d)]
    # download copies: SVG is vector text; PNG is drawn at print resolution (larger than the preview)
    svg, media, title = render_one(d, "correlation", "svg")
    assert media == "image/svg+xml" and svg.lstrip().startswith(b"<?xml") and b"<svg" in svg and title == "Correlation heatmap"
    png = render_one(d, "col:a", "png")[0]
    preview = base64.b64decode(render_charts(d)[0]["image"].split(",", 1)[1])
    from PIL import Image
    big, small = Image.open(io.BytesIO(png)).size, Image.open(io.BytesIO(preview)).size
    assert big[0] > 3 * small[0], (big, small)        # 300 vs 90 dpi
    for bad in (("nope", "png"), ("missing", "gif")):
        try:
            render_one(d, *bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    print(f"eda self-check passed ({len(out)} charts)")
