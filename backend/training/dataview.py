"""A readable look at the data the models learn from (spec 2026-09-15).

The model matrix itself (e.g. 110,062 sparse TF-IDF columns) is never shown as
rows -- only summarised per input column. The table shows per-column preprocessed
values (text stays text, categories become their one-hot / ordinal columns,
numbers show their scaled / clipped values) and is built on request for just the
rows of one page from the saved pipeline, so no extra copy of the data is stored.
"""

import pandas as pd

from core import store
from preprocessing import execute
from training import train

PAGE_MAX = 200  # our starting point: most rows per page


def summary(fitted, X_one):
    """Per input column: the steps applied and how many model features it becomes."""
    columns = []
    for col, replay in fitted["columns"]:
        steps, width = [], 1
        for op, st, _intlike in replay:
            if op["op"] == "outliers" and op.get("method") == "remove_rows":
                steps.append("outlier rows removed during training")
                continue
            method = op.get("method")
            if op["op"] == "encode" and method == "text":
                width = sum(len(v.vocabulary_) for v in st)
            elif op["op"] == "encode" and method == "onehot":
                width = len(st)
            steps.append(execute._label(op["op"], op, width, st))
        columns.append({"column": str(col), "steps": steps or ["kept as-is"], "features": width})
    notes = []
    if fitted["keep"] is not None:
        notes.append(f"feature selection kept {len(fitted['keep'])} columns")
    if fitted["pca"]:
        notes.append(f"PCA replaced {len(fitted['pca']['num'])} numeric columns with {fitted['pca']['pca'].n_components_}")
    total = execute.to_matrix(fitted, execute.transform(fitted, X_one)).shape[1]
    return {"columns": columns, "notes": notes, "model_features": int(total)}


def data_page(ds_id, split="all", offset=0, limit=50):
    ctx = train.load_context(ds_id)
    fitted = {**store.load_prep(ds_id)["fitted"], "clean_ops": []}  # these rows are already cleaned
    parts = [("train", ctx["X_tr"], ctx["y_tr"]), ("test", ctx["X_te"], ctx["y_te"])]
    if split in ("train", "test"):
        parts = [p for p in parts if p[0] == split]
    elif split != "all":
        raise ValueError("split must be all, train or test.")
    total = sum(len(X) for _, X, _ in parts)
    offset, limit = max(0, int(offset)), max(1, min(int(limit), PAGE_MAX))
    pages, skip, take = [], offset, limit
    for name, X, y in parts:
        if take == 0:
            break
        if skip >= len(X):
            skip -= len(X)
            continue
        rows = X.iloc[skip:skip + take]
        out = execute.transform(fitted, rows)  # only this page's rows
        out.insert(0, "row", rows.index)
        labels = y[skip:skip + take]
        out[ctx["target"]] = [ctx["classes"][int(v)] for v in labels] if ctx["classes"] else labels
        out["split"] = name
        pages.append(out)
        take -= len(rows)
        skip = 0
    frame = pd.concat(pages) if pages else None
    return {"total": total, "offset": offset, "limit": limit, "rows_train": len(ctx["X_tr"]), "rows_test": len(ctx["X_te"]),
            "columns": [] if frame is None else list(map(str, frame.columns)),
            "rows": [] if frame is None else frame.astype(object).where(frame.notna(), None).to_dict(orient="records"),
            "summary": summary(fitted, ctx["X_tr"].iloc[:1])}


if __name__ == "__main__":
    import tempfile
    import warnings
    from pathlib import Path
    import numpy as np
    import pandas as pd
    warnings.filterwarnings("ignore")
    store.ROOT = Path(tempfile.mkdtemp())
    n = 100
    df = pd.DataFrame({"msg": [f"hello world {i}" for i in range(n)], "city": ["a", "b", "c", "d"] * 25,
                       "age": [20 + i % 30 for i in range(n)], "y": ["x", "z"] * 50})
    ds = store.create(df, "t.csv")
    plan = {"msg": [{"op": "encode", "method": "text"}], "city": [{"op": "encode", "method": "onehot"}],
            "age": [{"op": "scale", "method": "standard"}]}
    r = execute.apply_plan(df, "y", plan)
    r["fitted"]["clean_ops"] = store.clean_ops(ds)
    store.save_prep(ds, r["csv"], r["fitted"])

    page = data_page(ds, "all", 75, 10)
    assert page["total"] == 100 and page["rows_train"] == 80 and len(page["rows"]) == 10
    assert [row["split"] for row in page["rows"]] == ["train"] * 5 + ["test"] * 5, "paging crosses train -> test"
    assert set(page["columns"]) == {"row", "msg", "city_a", "city_b", "city_c", "city_d", "age", "y", "split"}, page["columns"]
    assert isinstance(page["rows"][0]["msg"], str) and page["rows"][0]["y"] in ("x", "z"), "text and labels stay readable"
    test_only = data_page(ds, "test", 0, 500)
    assert test_only["total"] == 20 and test_only["limit"] == PAGE_MAX and len(test_only["rows"]) == 20
    # the page matches the saved preprocessed export for those rows (same pipeline, no re-fitting)
    export = pd.read_csv(pd.io.common.StringIO(r["csv"]))
    first = data_page(ds, "train", 0, 3)["rows"]
    assert np.allclose([row["age"] for row in first], export["age"].head(3)), "page values differ from the export"

    s = page["summary"]
    by_col = {c["column"]: c for c in s["columns"]}
    assert by_col["city"]["features"] == 4 and by_col["age"]["features"] == 1
    assert by_col["msg"]["features"] > 50 and s["model_features"] == by_col["msg"]["features"] + 5, s
    assert s["model_features"] == train.facts(train.load_context(ds))["features"], "summary width != what models receive"
    try:
        data_page(ds, "everything")
        raise AssertionError("bad split accepted")
    except ValueError:
        pass
    print("dataview self-check passed")
