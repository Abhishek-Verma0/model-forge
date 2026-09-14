"""The fold-safe model pipeline (imbalanced-learn), so steps that drop or add rows
only touch fold-training rows and are skipped when predicting:
    [IQR row removal on the raw table] -> Prep -> [isolation forest] -> [resampling] -> model
"""

import numpy as np
import pandas as pd
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from preprocessing import advanced, execute
from training import epochs

SMOTE_K = 5                # published default (imbalanced-learn SMOTE k_neighbors)


class Prep(BaseEstimator, TransformerMixin):
    """Our preprocessing (execute.fit_prep / transform / to_matrix) as a
    scikit-learn step, so every CV fold refits it on its own training rows.
    Output: sparse matrix when there are text columns, dense array otherwise."""

    def __init__(self, columns=None, pipeline=None, task="classification"):
        self.columns = columns
        self.pipeline = pipeline
        self.task = task

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.seen_index_ = X.index  # which rows fitting saw -- the leakage check reads this
        intlike = {c for c in X.columns if execute._is_intlike(X[c])}
        self.fitted_, tr_out, *_ = execute.fit_prep(
            X, np.asarray(y), self.columns or {}, self.pipeline or {}, self.task, intlike)
        return self._matrix(tr_out)

    def transform(self, X):
        return self._matrix(execute.transform(self.fitted_, X))

    def _matrix(self, frame):
        m = execute.to_matrix(self.fitted_, frame)
        self.n_features_out_ = m.shape[1]
        return m if self.fitted_["text"] else m.toarray()


def _drop_iqr_rows(X, y, cols_k):
    """Per-column 'remove rows' outliers, bounds from the rows it is given (fold train)."""
    y = np.asarray(y)
    for col, k in cols_k:
        lo, hi = execute._iqr_bounds(X[col], k)
        num = pd.to_numeric(X[col], errors="coerce")
        keep = (num.between(lo, hi) | num.isna()).to_numpy()
        X, y = X[keep], y[keep]
    return X, y


def _drop_iforest_rows(X, y, contamination, random_state):
    keep = IsolationForest(contamination=contamination, random_state=random_state).fit_predict(X) == 1
    return X[keep], np.asarray(y)[keep]


def build_pipeline(ctx, key, row, facts, folds, kwargs=None):
    """One model's fold-safe pipeline. The baseline gets the same row filters and
    preprocessing but no resampling or class weights (they would change what
    'most frequent class' means)."""
    columns, pipeline, seed = ctx["prep"]["columns"], ctx["prep"]["pipeline"] or {}, ctx["seed"]
    steps = []
    cols_k = [(c, float(o.get("k", execute.IQR_K))) for c, ops in columns.items() for o in ops
              if o["op"] == "outliers" and o.get("method") == "remove_rows" and c in ctx["X_tr"].columns]
    if cols_k:
        steps.append(("iqr_rows", FunctionSampler(func=_drop_iqr_rows, validate=False,
                                                  kw_args={"cols_k": cols_k})))
    steps.append(("prep", Prep(columns, pipeline, ctx["task"])))
    if advanced._m("outlier_removal", pipeline):
        c = (pipeline.get("outlier_removal") or {}).get("contamination")
        steps.append(("iforest_rows", FunctionSampler(func=_drop_iforest_rows, validate=False, kw_args={
            "contamination": advanced.CONTAMINATION if c is None else float(c), "random_state": seed})))
    model = row["make"](seed).set_params(**(kwargs or {}))
    imb = advanced._m("imbalance", pipeline) if ctx["task"] == "classification" and key != "baseline" else ""
    if imb == "oversample":
        steps.append(("resample", RandomOverSampler(random_state=seed)))
    elif imb == "undersample":
        steps.append(("resample", RandomUnderSampler(random_state=seed)))
    elif imb == "smote":
        smallest = min(facts["class_counts"].values())
        k = max(1, min(SMOTE_K, smallest * (folds - 1) // folds - 1))  # computed: a fold must hold k+1 of each class
        steps.append(("resample", SMOTE(random_state=seed, k_neighbors=k)))
    elif imb == "class_weights" and row["balanced"]:
        model.set_params(**row["balanced"])
    steps.append(("model", model))
    return Pipeline(steps)


def fit_steps(pipe, X, y):
    """Fit every step before the model exactly as imbalanced-learn's Pipeline.fit does
    (samplers resample, transformers transform). Returns (fitted steps, Xt, yt)."""
    steps = [(name, clone(step)) for name, step in pipe.steps[:-1]]
    Xt, yt = X, y
    for _name, step in steps:
        if hasattr(step, "fit_resample"):
            Xt, yt = step.fit_resample(Xt, yt)
        else:
            Xt = step.fit_transform(Xt, yt)
    return steps, Xt, np.asarray(yt)


def transform_steps(steps, X):
    """Prediction-time path: transformers only -- samplers never touch rows being scored."""
    for _name, step in steps:
        if not hasattr(step, "fit_resample"):
            X = step.transform(X)
    return X


def fit_fold(pipe, row, fit, X, y, share, seed, task, on_epoch, check, classes=None):
    """Fit one pipeline on rows (X, y) and return (fitted Pipeline, epoch info).

    Epoch-based models with share > 0 first hold back `share` of these RAW rows --
    before preprocessing and resampling -- and report validation loss / early-stop
    on them, so held-back rows never shape the preprocessing or contain resampled
    copies (spec 2026-09-14). Other models fit on all rows."""
    X_val = y_val = None
    y = np.asarray(y)
    if row["epochs"] and share > 0:
        idx = np.arange(len(X))
        try:
            fit_idx, val_idx = train_test_split(idx, test_size=share, random_state=seed,
                                                stratify=y if task == "classification" else None)
        except ValueError:  # too few rows per class to stratify the held-back share
            fit_idx, val_idx = train_test_split(idx, test_size=share, random_state=seed)
        X_raw_val, y_val = X.iloc[val_idx], y[val_idx]
        X, y = X.iloc[fit_idx], y[fit_idx]
    steps, Xt, yt = fit_steps(pipe, X, y)
    if y_val is not None:
        X_val = transform_steps(steps, X_raw_val)
    model, info = epochs.fit_model(row, clone(pipe.steps[-1][1]), Xt, yt, X_val, y_val, fit, on_epoch, check,
                                   classes)
    return Pipeline(steps + [("model", model)]), info


def input_schema(pipe, X):
    """What one prediction row must contain, read from the model's OWN fitted
    preprocessing (the categories it learned, which columns are text) plus the
    training rows it was fitted on (number ranges). This -- not a re-detection of
    column types -- is what the manual prediction form uses."""
    fitted = pipe.named_steps["prep"].fitted_
    imputed_block = set((fitted["impute"] or {}).get("num", []))
    fields = []
    for col, replay in fitted["columns"]:
        name = str(col)
        encode = next(((o, st) for o, st, _ in replay if o["op"] == "encode"), (None, None))
        method = encode[0].get("method") if encode[0] else None
        imputed = col in imputed_block or any(o["op"] == "impute" for o, _, _ in replay)
        s = X[col]
        if method == "text":
            field = {"kind": "text", "missing_ok": True}
        elif method == "onehot":
            field = {"kind": "category", "choices": [c[len(name) + 1:] for c in encode[1]], "missing_ok": True}
        elif method == "ordinal":
            field = {"kind": "category", "choices": [str(c) for c in encode[1]], "missing_ok": True}
        elif pd.api.types.is_bool_dtype(s):
            field = {"kind": "boolean", "missing_ok": imputed}
        elif pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce").dropna()
            field = {"kind": "number", "integer": bool(execute._is_intlike(s)), "missing_ok": imputed,
                     "min": float(vals.min()) if len(vals) else None, "max": float(vals.max()) if len(vals) else None}
        else:  # an unencoded text column cannot reach a model (to_matrix refuses); kept for completeness
            field = {"kind": "text", "missing_ok": True}
        fields.append({"name": name, **field})
    return fields

