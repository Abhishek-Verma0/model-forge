"""On-demand chart rendering for the frontend's chart builder.

One seaborn call per chart type, driven by (kind, x, y, hue), returned as PNG
bytes. Same matplotlib gotchas as eda.py -- Agg backend, close every figure.

Each chart takes only the kinds of column it can show (FITS), read from the same
checks the Quality audit runs (preprocessing/detector.py). As in ydata-profiling,
Sweetviz and DataPrep, a many-valued column is never drawn one bar per value:
  number    -> histogram, density, box, violin, scatter / line axes -- a number ID (a row counter,
               a year axis) too: a scale draws fast, only one bar per value is slow
  coded     -> a number with at most 50 values (class 1/2/3, Likert 1-5): number charts + bar, pie, group
  date      -> histogram, scatter / line x -- as real dates (text dates are read first)
  category  -> histogram, bar, pie, colour / group (at most 50 values: detector.MAX_CATEGORICAL_UNIQUE)
  text      -> bar of the TOP_VALUES most common values + "Other" (many-valued labels: cities, codes)
  free text -> text length, common words (reviews, notes)
  id        -> (a text ID) nothing: every value differs, there is nothing to count
Measured 2026-10-21 before this rule: one bar per id of viral_videos.csv (12,000) took 152 s, text
dates of nvda_stock_clean.csv 73 s -- and 0.1 s once read as dates.
"""

import io
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")            # headless -- required off the main thread (web server)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402

from core import logs            # noqa: E402
from preprocessing.clean import to_datetime                                # noqa: E402
from preprocessing.detector import (DATETIME_SHARE, MAX_CATEGORICAL_UNIQUE,  # noqa: E402
                                    detect_column_types, detect_id_columns)

sns.set_theme(style="whitegrid")

SAMPLE = 5000        # our starting point: point-heavy charts sample this many rows
MAX_BAR_CATS = MAX_CATEGORICAL_UNIQUE  # up to this many values every value gets its own bar
TOP_VALUES = 10      # our starting point: bars above that -- the same 10 the profile and audit list
TOP_WORDS = 20       # our starting point: words in a "common words" chart
LABEL_ANGLE = 45     # our starting point: x names that don't fit side by side are turned this far
LABEL_GAP = 1.2      # our starting point: turned names keep 20% of a text line clear of the next
MAX_WIDTH_IN = 20    # our starting point: the widest a chart grows (inches) so its names fit
log = logs.get("chart")

# For box/violin the user's chosen column is the numeric distribution (seaborn y)
# and hue is the group (seaborn x).
_KINDS = {
    "histogram": lambda df, x, y, hue, ax: sns.histplot(data=df, x=x, hue=hue, ax=ax),
    "density":   lambda df, x, y, hue, ax: sns.kdeplot(data=df, x=x, hue=hue, fill=True, ax=ax),
    "box":       lambda df, x, y, hue, ax: sns.boxplot(data=df, x=hue, y=x, ax=ax),
    "violin":    lambda df, x, y, hue, ax: sns.violinplot(data=df, x=hue, y=x, ax=ax),
    "scatter":   lambda df, x, y, hue, ax: sns.scatterplot(data=df, x=x, y=y, hue=hue, s=12, alpha=0.5, ax=ax),
    "line":      lambda df, x, y, hue, ax: sns.lineplot(data=df.sort_values(x), x=x, y=y, hue=hue, ax=ax),
    "bar":       lambda df, x, y, hue, ax: _bar(df, x, hue, ax),
    "pie":       lambda df, x, y, hue, ax: _pie(df, x, ax),   # class distribution / category share
    "length":    lambda df, x, y, hue, ax: _length(df, x, ax),
    "words":     lambda df, x, y, hue, ax: _words(df, x, ax),
    "tsne":      lambda df, x, y, hue, ax: _tsne(df, hue, ax),  # 2-D map of all numeric cols
}
KINDS = list(_KINDS)
_MEDIA = {"png": "image/png", "svg": "image/svg+xml"}
_NO_X = {"tsne"}     # kinds that use the whole numeric matrix, not a chosen column
_NEEDS_Y = {"scatter", "line"}  # kinds plotting one column against another
_NO_HUE = {"pie", "length", "words"}  # kinds that draw one column only
_SAMPLED = {"scatter", "density", "line", "length", "words"}

# What each chart can show, by column role (see the module docstring).
_NUM = {"number", "coded"}
FITS = {"histogram": _NUM | {"date", "category"}, "density": _NUM, "box": _NUM, "violin": _NUM,
        "scatter": _NUM | {"date"}, "line": _NUM | {"date"}, "bar": {"coded", "category", "text"},
        "pie": {"coded", "category"}, "length": {"free_text"}, "words": {"free_text"}, "tsne": set()}
Y_FITS, HUE_FITS = _NUM, {"coded", "category"}
_ROLE = {"datetime": "date", "boolean": "category", "categorical": "category"}
_SAYS = {"number": "a number column with many different values", "coded": "a number column",
         "date": "a date column", "category": "a category column",
         "text": "a text column with many different values", "free_text": "a free-text column",
         "id": "an ID column (every value is different, so there is nothing to count)"}


def roles(checks):
    """Column -> number | coded | date | category | text | free_text | id, from the audit's
    column-type and ID checks (detector.run_quality_report). Only a text ID is "id"."""
    ids = {c["column"] for c in checks["id_columns"]["columns"]}
    out = {}
    for c in checks["column_types"]["columns"]:
        col, kind = c["column"], c["detected_type"]
        if kind == "numeric":
            out[col] = "coded" if c["distinct_values"] <= MAX_CATEGORICAL_UNIQUE else "number"
        elif col in ids:
            out[col] = "id"
        else:
            out[col] = _ROLE.get(kind) or ("free_text" if c["text_kind"] == "free_text" else "text")
    return out


def builder_options(checks):
    """The columns each chart can draw -- the builder lists only these."""
    role = roles(checks)

    def pick(fits):
        return [c for c, r in role.items() if r in fits]
    return {"x": {k: pick(f) for k, f in FITS.items()}, "y": pick(Y_FITS), "hue": pick(HUE_FITS)}


def _refusal(col, role, what):
    fit = [k for k, f in FITS.items() if role in f]
    hint = f" It fits: {', '.join(fit)}." if fit else ""
    return f"'{col}' is {_SAYS[role]} - a {what} can't show it.{hint}"


def _bar(df, x, hue, ax):
    counts = df[x].value_counts()
    if len(counts) <= MAX_BAR_CATS:
        sns.countplot(data=df, x=x, hue=hue, ax=ax)
        return
    if hue:
        raise ValueError(f"'{x}' has {len(counts):,} different values - its chart shows the {TOP_VALUES} most "
                         "common and can't also be split by a group.")
    top = counts.head(TOP_VALUES)
    labels = [str(v) for v in top.index] + [f"Other ({len(counts) - TOP_VALUES:,} values)"]
    values = list(top.values) + [int(counts.iloc[TOP_VALUES:].sum())]
    ax.barh(labels[::-1], values[::-1])   # horizontal: long labels stay readable
    ax.set_xlabel("rows")


def _pie(df, x, ax):
    counts = df[x].value_counts().head(MAX_BAR_CATS)
    ax.pie(counts.values, labels=[str(v) for v in counts.index], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")


def _length(df, x, ax):
    sns.histplot(x=df[x].dropna().astype(str).str.split().str.len(), ax=ax)
    ax.set_xlabel("words per value")


def _words(df, x, ax):
    """The most used words (3+ letters, scikit-learn's English stop words left out -- the
    rule bibliometrics/analysis.py uses for titles)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    found = df[x].dropna().astype(str).str.lower().str.findall(r"[a-z][a-z-]{2,}")
    top = Counter(w for ws in found for w in ws if w not in ENGLISH_STOP_WORDS).most_common(TOP_WORDS)
    if not top:
        raise ValueError(f"'{x}' has no words to count.")
    ax.barh([w for w, _ in top][::-1], [n for _, n in top][::-1])
    ax.set_xlabel("times used")


def _fit_xlabels(fig, ax):
    """Turn the x names when they would overlap side by side (15 category names on the 6-inch chart
    did: 14 of 14 neighbouring pairs), and widen the chart when even turned they would touch --
    two names turned by an angle sit (tick spacing x sin angle) apart. Measured on the drawn labels;
    names that already fit (numbers, dates, a few short categories) are left as they are."""
    fig.canvas.draw()                                    # lay out once so the label sizes are known
    labels = [t for t in ax.get_xticklabels() if t.get_text()]
    boxes = sorted((t.get_window_extent() for t in labels), key=lambda b: b.x0)
    if all(a.x1 <= b.x0 for a, b in zip(boxes, boxes[1:])):
        return
    step = max(1.0, min((b.x0 + b.x1 - a.x0 - a.x1) / 2 for a, b in zip(boxes, boxes[1:])))  # px between ticks
    line = max(b.height for b in boxes)                  # one line of text, px
    plt.setp(labels, rotation=LABEL_ANGLE, ha="right", rotation_mode="anchor")
    need = LABEL_GAP * line / (step * math.sin(math.radians(LABEL_ANGLE)))
    if need > 1:   # the axes widen with the figure, so the ticks spread by the same factor
        fig.set_figwidth(min(MAX_WIDTH_IN, fig.get_figwidth() * need))


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
    """Render one chart -> (bytes, media_type). Validates inputs -- every column must
    fit the chart (FITS) -- and samples big/point charts. fmt = png | svg (SVG is
    vector, for print/publication)."""
    if kind not in _KINDS:
        raise ValueError(f"Unknown chart '{kind}'. One of: {', '.join(KINDS)}")
    if fmt not in _MEDIA:
        raise ValueError(f"Unknown format '{fmt}'. Use png or svg.")
    x = None if kind in _NO_X else x
    y = y if kind in _NEEDS_Y else None
    hue = None if kind in _NO_HUE else hue
    for col in (x, y, hue):
        if col and col not in df.columns:
            raise ValueError(f"No column '{col}' in this dataset.")
    if kind in _NEEDS_Y:
        if not (x and y):
            raise ValueError(f"{kind.capitalize()} needs both X and Y columns.")
    elif kind not in _NO_X and not x:
        raise ValueError(f"'{kind}' needs a column.")

    used = list(dict.fromkeys(c for c in (x, y, hue) if c))
    sub = df[used]
    role = roles({"column_types": detect_column_types(sub), "id_columns": detect_id_columns(sub)}) if used else {}
    for col, fits, what in ((x, FITS[kind], f"{kind} chart"), (y, Y_FITS, f"{kind} chart's Y axis"),
                            (hue, HUE_FITS, "colour / group")):
        if col and role[col] not in fits:
            raise ValueError(_refusal(col, role[col], what))

    data = df if kind in _NO_X else sub
    if kind == "tsne" and len(df) > 1500:
        data = df.sample(1500, random_state=0)  # t-SNE is O(n^2)-ish -- cap hard
    elif kind in _SAMPLED and len(df) > SAMPLE:
        data = data.sample(SAMPLE, random_state=0)  # ponytail: sample so 745k points don't hang the render
    if x and role[x] == "date" and not pd.api.types.is_datetime64_any_dtype(data[x]):
        try:   # day / month order from the column's own values, as cleaning does
            dates = to_datetime(data[x])[0]
        except (ValueError, TypeError, OverflowError) as exc:   # OutOfBoundsDatetime is a ValueError
            raise ValueError(f"'{x}' looks like dates but can't be read as dates ({exc}).") from None
        if dates.notna().sum() < DATETIME_SHARE * data[x].notna().sum():
            raise ValueError(f"Only {int(dates.notna().sum()):,} of {int(data[x].notna().sum()):,} values of "
                             f"'{x}' read as dates - check its format on the Cleaning step.")
        data = data.copy()
        data[x] = dates

    distinct = int(df[x].nunique(dropna=True)) if x else 0
    what = f"draw {kind} of {x} ({role[x]}, {distinct:,} distinct)" if x else f"draw {kind}"
    with logs.step(log, f"{what}{f' vs {y}' if y else ''}{f' by {hue}' if hue else ''}, {len(data):,} rows"):
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            _KINDS[kind](data, x, y, hue, ax)
            sample = f" (random {len(data):,} of {len(df):,} rows)" if len(data) < len(df) else ""
            ax.set_title(_title(kind, x, y, hue) + sample)
            _fit_xlabels(fig, ax)
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=90, bbox_inches="tight")
            return buf.getvalue(), _MEDIA[fmt]
        finally:
            plt.close(fig)   # always close, even if seaborn raised


def _title(kind, x, y, hue):
    t = f"{kind}: {y} vs {x}" if kind in _NEEDS_Y else f"{kind}: {x}" if x else kind
    return t + (f" by {hue}" if hue else "")


if __name__ == "__main__":
    import time
    from preprocessing.detector import run_quality_report
    n = 400
    d = pd.DataFrame({"a": range(n), "b": [i % 5 for i in range(n)], "g": list("xy") * (n // 2),
                      "num": [float(i * 37 % 101) for i in range(n)],
                      "day": pd.date_range("2020-01-01", periods=n).strftime("%d/%m/%Y"),  # text dates, day first
                      "city": [f"city {i % 120}" for i in range(n)],                   # 120 short labels
                      "note": [f"the parcel {i} arrived late and broken again" for i in range(n)],
                      "vid": [f"vid_{i:06d}" for i in range(n)]})
    opts = builder_options(run_quality_report(d)["checks"])
    role = roles(run_quality_report(d)["checks"])
    assert role == {"a": "number", "b": "coded", "g": "category", "num": "number", "day": "date", "city": "text",
                    "note": "free_text", "vid": "id"}, role
    assert opts["x"]["histogram"] == ["a", "b", "g", "num", "day"] and opts["x"]["bar"] == ["b", "g", "city"], opts["x"]
    assert opts["x"]["length"] == ["note"] and opts["y"] == ["a", "b", "num"] and opts["hue"] == ["b", "g"], opts
    assert not any("vid" in cols for cols in opts["x"].values())   # a text ID is never offered

    # every chart the builder offers renders, png and svg
    for k in KINDS:
        cols = opts["x"][k] or [None]
        img, media = render(d, k, x=cols[0], y=("num" if k in _NEEDS_Y else None), hue=("g" if k not in _NO_HUE else None))
        assert img[:4] == b"\x89PNG" and media == "image/png", k
    svg, media = render(d, "bar", x="g", fmt="svg")
    assert b"<svg" in svg[:200] and media == "image/svg+xml"

    # what is refused, with the reason -- instantly, before anything is drawn
    for kind, x, y, hue, msg in (("histogram", "vid", None, None, "ID column"),
                                 ("histogram", "city", None, None, "many different values"),
                                 ("bar", "vid", None, None, "ID column"),
                                 ("pie", "city", None, None, "It fits: bar"),
                                 ("density", "g", None, None, "category column"),
                                 ("bar", "num", None, None, "number column with many different values"),
                                 ("scatter", "num", "city", None, "Y axis"),
                                 ("scatter", "num", "b", "city", "colour / group"),
                                 ("scatter", "a", None, None, "both X and Y")):
        t = time.perf_counter()
        try:
            render(d, kind, x=x, y=y, hue=hue)
            raise AssertionError((kind, x, y, hue))
        except ValueError as e:
            assert msg in str(e) and time.perf_counter() - t < 2, (kind, x, str(e))

    # many values: the 10 most common + one "Other" bar; text dates are read as dates (day first)
    fig, ax = plt.subplots()
    _bar(d, "city", None, ax)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == TOP_VALUES + 1 and labels[0] == "Other (110 values)", labels
    plt.close(fig)
    # x names that would overlap side by side are turned, and the chart widens when even turned they
    # would touch: afterwards neighbouring names sit at least one text line apart; short names stay flat
    long = pd.DataFrame({"code": [f"department-{i % 45:02d}" for i in range(900)]})
    for frame, col, turned in ((d, "g", False), (long, "code", True)):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=frame, x=col, ax=ax)
        _fit_xlabels(fig, ax)
        fig.canvas.draw()
        lab = ax.get_xticklabels()[0]
        assert (lab.get_rotation() == LABEL_ANGLE) == turned and (fig.get_figwidth() > 6) == turned, (col, fig.get_figwidth())
        if turned:
            ticks = ax.transData.transform([(t, 0) for t in ax.get_xticks()])[:, 0]
            line = fig.canvas.get_renderer().get_text_width_height_descent(lab.get_text(), lab.get_fontproperties(),
                                                                            ismath=False)[1]
            assert min(b - a for a, b in zip(ticks, ticks[1:])) * math.sin(math.radians(LABEL_ANGLE)) >= line
        plt.close(fig)
    days = to_datetime(d["day"])[0]
    assert days.iloc[12] == pd.Timestamp("2020-01-13") and days.notna().all()

    # still drawn: a year axis (a number counter), class / Likert codes as bars and groups, month names
    # as categories, timestamps with a zone as dates (each of these was refused or failed in review)
    yrs = pd.DataFrame({"year": range(1960, 2021), "gdp": [1.0 + i * 0.1 for i in range(61)]})
    titanic = pd.DataFrame({"pclass": [1, 2, 3] * 100, "fare": [i * 1.5 for i in range(300)],
                            "month": ["Jan", "July", "March"] * 100,
                            "stamp": [f"2020-01-{1 + i % 28:02d}T10:00:00+05:30" for i in range(300)]})
    for frame, kind, x, y, hue in ((yrs, "line", "year", "gdp", None), (titanic, "bar", "pclass", None, None),
                                   (titanic, "box", "fare", None, "pclass"), (titanic, "pie", "month", None, None),
                                   (titanic, "histogram", "stamp", None, None)):
        assert render(frame, kind, x=x, y=y, hue=hue)[0][:4] == b"\x89PNG", (kind, x)
    print(f"plot self-check passed ({len(KINDS)} kinds, png+svg, refusals with reasons)")
