"""Dataset-level preprocessing stage -- the multivariate steps that don't fit the
per-column executor in execute.py: advanced imputation, multivariate outlier
removal, feature selection, class-imbalance handling, and dimensionality
reduction.

Every step is fit on TRAIN only (leakage-safe) and chosen from a fixed, validated
allowlist -- the user/LLM SELECTS a method, Python runs the tested function; no
arbitrary code. Runs after execute.py's per-column impute/encode/scale, on the
assembled *numeric* feature matrix (imputation is the exception -- it runs early,
before encoding, on the numeric columns).

pipeline = {
  "imputation":      {"method": "knn"|"iterative"},          # numeric block, train-fit
  "outlier_removal": {"method": "isolation_forest"},         # drops TRAIN rows only
  "feature_selection":{"method": "correlation"|"chi2"|"anova"|"mutual_info"|"rfe", "k": 10},
  "imbalance":       {"method": "oversample"|"undersample"|"smote"|"class_weights"},
  "reduction":       {"method": "pca", "n": 5},
}
Any sub-key omitted / {"method":"none"} = that step is skipped.

ponytail: t-SNE/UMAP (spec's "reduction for visualization") are deferred -- they're
transductive (no train->test transform) and UMAP is a heavy extra dep; the EDA tab's
scatter/pairplot already cover 2-D visualization.
"""

import numpy as np
import pandas as pd

_IMPUTE = {"knn", "iterative"}
_OUTREM = {"isolation_forest"}
_FS = {"correlation", "chi2", "anova", "mutual_info", "rfe"}
_IMB = {"oversample", "undersample", "smote", "class_weights"}
_RED = {"pca"}


def _m(section, p):
    """method string for a pipeline section, '' if absent/none."""
    v = (p.get(section) or {}).get("method", "")
    return "" if v in ("", "none", None) else v


def validate_pipeline(p):
    checks = [("imputation", _IMPUTE), ("outlier_removal", _OUTREM),
              ("feature_selection", _FS), ("imbalance", _IMB), ("reduction", _RED)]
    for section, allowed in checks:
        m = _m(section, p)
        if m and m not in allowed:
            raise ValueError(f"{section}.method must be one of {sorted(allowed)} (got '{m}')")


# --- advanced imputation (numeric block, before encoding) -------------------

def advanced_impute(X_tr, X_te, method, intlike_cols):
    """KNN / iterative imputation of ALL numeric feature columns at once, fit on
    train. Returns (X_tr, X_te, note). intlike_cols are rounded back to integers."""
    from sklearn.impute import KNNImputer
    num = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    if not num:
        return X_tr, X_te, "no numeric columns to impute"
    if method == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(random_state=0, max_iter=10)
    else:
        imp = KNNImputer()
    imp.fit(X_tr[num])
    X_tr = X_tr.copy(); X_te = X_te.copy()
    X_tr[num] = imp.transform(X_tr[num])
    X_te[num] = imp.transform(X_te[num])
    for c in num:
        if c in intlike_cols:  # keep integer semantics (age stays whole)
            X_tr[c] = X_tr[c].round().astype("Int64")
            X_te[c] = X_te[c].round().astype("Int64")
    return X_tr, X_te, f"{method} imputation on {len(num)} numeric column(s)"


# --- multivariate outlier removal (train rows only) -------------------------

def remove_outliers(X_tr, y_tr, method, contamination=0.05):
    from sklearn.ensemble import IsolationForest
    num = X_tr.select_dtypes("number")
    if num.shape[1] == 0:
        return X_tr, y_tr, "no numeric columns for outlier detection"
    keep = IsolationForest(contamination=contamination, random_state=0).fit_predict(num) == 1
    return X_tr[keep], y_tr[keep], f"isolation forest dropped {int((~keep).sum())} outlier train row(s)"


# --- feature selection (fit on train) ---------------------------------------

def select_features(X_tr, X_te, y_tr, method, k, task):
    from sklearn.feature_selection import (
        SelectKBest, chi2, f_classif, f_regression,
        mutual_info_classif, mutual_info_regression, RFE,
    )
    num = X_tr.select_dtypes("number")
    k = max(1, min(int(k or 10), num.shape[1]))
    if num.shape[1] <= k:
        return X_tr, X_te, "fewer features than k -- nothing removed"
    y = pd.factorize(y_tr)[0] if task == "classification" else pd.to_numeric(y_tr, errors="coerce")

    if method == "correlation":
        corr = num.apply(lambda col: np.abs(np.corrcoef(col, y)[0, 1]) if col.std() else 0.0)
        keep = corr.sort_values(ascending=False).head(k).index.tolist()
    elif method == "rfe":
        from sklearn.linear_model import LogisticRegression, LinearRegression
        est = LogisticRegression(max_iter=200) if task == "classification" else LinearRegression()
        sel = RFE(est, n_features_to_select=k).fit(num, y)
        keep = num.columns[sel.support_].tolist()
    else:
        if method == "chi2":
            # chi2 needs non-negative -> shift each column to [0, .] using train min
            score = chi2
            data = num - num.min()
        elif method == "anova":
            score = f_classif if task == "classification" else f_regression
            data = num
        else:  # mutual_info
            score = mutual_info_classif if task == "classification" else mutual_info_regression
            data = num
        sel = SelectKBest(score, k=k).fit(data, y)
        keep = num.columns[sel.get_support()].tolist()

    dropped = [c for c in X_tr.columns if c not in keep]
    return X_tr[keep], X_te[keep], f"{method}: kept {len(keep)} of {num.shape[1]} feature(s), dropped {len(dropped)}"


# --- class imbalance (train rows only) --------------------------------------

def balance(X_tr, y_tr, method, task):
    """Returns (X_tr, y_tr, note, class_weights). class_weights is a dict only for
    the 'class_weights' method (no resampling), else None."""
    if task != "classification":
        return X_tr, y_tr, "imbalance handling skipped (not a classification task)", None
    counts = pd.Series(y_tr).value_counts()

    if method == "class_weights":
        from sklearn.utils.class_weight import compute_class_weight
        classes = counts.index.to_numpy()
        w = compute_class_weight("balanced", classes=classes, y=y_tr)
        weights = {str(c): round(float(x), 3) for c, x in zip(classes, w)}
        return X_tr, y_tr, "computed balanced class weights (data unchanged)", weights

    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ValueError("SMOTE needs imbalanced-learn: pip install imbalanced-learn")
        num = X_tr.select_dtypes("number")
        if num.shape[1] != X_tr.shape[1]:
            raise ValueError("SMOTE needs an all-numeric feature matrix (encode categoricals first).")
        k = max(1, min(5, int(counts.min()) - 1))
        Xr, yr = SMOTE(random_state=0, k_neighbors=k).fit_resample(X_tr, y_tr)
        return Xr, pd.Series(yr), f"SMOTE resampled {len(X_tr)} -> {len(Xr)} train rows", None

    # random over/under sampling via pandas (no extra dependency)
    target_n = counts.max() if method == "oversample" else counts.min()
    replace = method == "oversample"
    idx = []
    y_ser = pd.Series(y_tr).reset_index(drop=True)
    X_res = X_tr.reset_index(drop=True)
    for cls, grp in y_ser.groupby(y_ser):
        idx += list(grp.sample(target_n, replace=replace, random_state=0).index)
    idx = np.array(idx)
    note = f"{method} resampled {len(X_tr)} -> {len(idx)} train rows"
    return X_res.iloc[idx].reset_index(drop=True), y_ser.iloc[idx].reset_index(drop=True), note, None


# --- dimensionality reduction (fit on train) --------------------------------

def reduce(X_tr, X_te, method, n):
    from sklearn.decomposition import PCA
    num = X_tr.select_dtypes("number")
    if num.shape[1] < 2:
        return X_tr, X_te, "too few numeric features for reduction"
    n = max(1, min(int(n or 2), num.shape[1]))
    pca = PCA(n_components=n, random_state=0).fit(num)
    cols = [f"pc{i + 1}" for i in range(n)]
    tr = pd.DataFrame(pca.transform(num), columns=cols, index=X_tr.index)
    te = pd.DataFrame(pca.transform(X_te.select_dtypes("number")[num.columns]), columns=cols, index=X_te.index)
    var = round(float(pca.explained_variance_ratio_.sum()) * 100, 1)
    return tr, te, f"PCA -> {n} component(s), {var}% variance retained"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 1, n), "age": rng.integers(20, 60, n),
    })
    y = pd.Series((X["a"] + rng.normal(0, 0.3, n) > 0).astype(int))          # imbalanced-ish
    Xtr, Xte, ytr = X.iloc[:150], X.iloc[150:], y.iloc[:150]

    # advanced impute keeps age integer
    Xi = Xtr.copy(); Xi.loc[Xi.index[:5], "age"] = np.nan
    a_tr, a_te, note = advanced_impute(Xi, Xte.copy(), "knn", {"age"})
    assert str(a_tr["age"].dtype) == "Int64" and a_tr["age"].notna().all(), note

    # isolation forest drops some train rows
    o_tr, o_y, _ = remove_outliers(Xtr, ytr, "isolation_forest")
    assert len(o_tr) < len(Xtr) and len(o_tr) == len(o_y)

    # feature selection keeps k
    for meth in ("correlation", "chi2", "anova", "mutual_info", "rfe"):
        s_tr, s_te, note = select_features(Xtr, Xte, ytr, meth, 2, "classification")
        assert s_tr.shape[1] == 2, (meth, note)

    # over/under sampling balance the classes
    for meth in ("oversample", "undersample"):
        b_tr, b_y, note, _ = balance(Xtr, ytr, meth, "classification")
        vc = b_y.value_counts()
        assert vc.min() == vc.max(), (meth, vc.to_dict())
    _, _, _, weights = balance(Xtr, ytr, "class_weights", "classification")
    assert weights and len(weights) == 2

    # PCA reduces to n components on train + test
    r_tr, r_te, note = reduce(Xtr, Xte, "pca", 2)
    assert list(r_tr.columns) == ["pc1", "pc2"] and len(r_te) == len(Xte), note

    validate_pipeline({"feature_selection": {"method": "chi2", "k": 3}})
    try:
        validate_pipeline({"reduction": {"method": "nope"}}); raise AssertionError("should reject")
    except ValueError:
        pass
    print("advanced self-check passed")
