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
  "imputation":      {"method": "knn"|"iterative", "n_neighbors": 5},  # numeric block, train-fit
  "outlier_removal": {"method": "isolation_forest", "contamination": 0.05},  # drops TRAIN rows only
  "feature_selection":{"method": "correlation"|"chi2"|"anova"|"mutual_info"|"rfe", "k": 10},  # k < 1: share kept
  "imbalance":       {"method": "oversample"|"undersample"|"smote"|"class_weights"},
  "reduction":       {"method": "pca"|"autoencoder", "n": 5, "scale": False},  # n < 1: variance kept
}
Any sub-key omitted / {"method":"none"} = that step is skipped.

ponytail: t-SNE/UMAP (spec's "reduction for visualization") are deferred -- they're
transductive (no train->test transform) and UMAP is a heavy extra dep; the EDA tab's
scatter/pairplot already cover 2-D visualization.
"""

import math

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

CONTAMINATION = 0.05      # our starting point: IsolationForest expected outlier share (scikit-learn's is "auto")
KNN_NEIGHBORS = 5         # published default: scikit-learn KNNImputer n_neighbors=5
ITERATIVE_MAX_ITER = 10   # published default: scikit-learn IterativeImputer max_iter=10

# Ordered: the Preprocessing tab lists them in this order (execute.options).
_IMPUTE = ("knn", "iterative")
_OUTREM = ("isolation_forest",)
_FS = ("correlation", "chi2", "anova", "mutual_info", "rfe")
_IMB = ("oversample", "undersample", "smote", "class_weights")
_RED = ("pca", "autoencoder")


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
    c = (p.get("outlier_removal") or {}).get("contamination")
    if c is not None and not 0 < float(c) < 0.5:
        raise ValueError(f"outlier_removal.contamination must be between 0 and 0.5 (got {c}).")
    k = (p.get("imputation") or {}).get("n_neighbors")
    if k is not None and int(k) < 1:
        raise ValueError(f"imputation.n_neighbors must be at least 1 (got {k}).")


# --- advanced imputation (numeric block, before encoding) -------------------

def fit_impute(X_tr, method, intlike_cols, n_neighbors=None):
    """Fit KNN / iterative imputation on ALL numeric train columns. Returns
    (state, note); state is None when there is nothing numeric."""
    from sklearn.impute import KNNImputer
    num = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    if not num:
        return None, "no numeric columns to impute"
    if method == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(random_state=0, max_iter=ITERATIVE_MAX_ITER)
    else:
        imp = KNNImputer(n_neighbors=int(n_neighbors or KNN_NEIGHBORS))
    state = {"imp": imp.fit(X_tr[num]), "num": num, "int": [c for c in num if c in intlike_cols]}
    return state, f"{method} imputation on {len(num)} numeric column(s)"


def advanced_impute(X_tr, X_te, method, intlike_cols, n_neighbors=None):
    """Fit on train, apply to train + test. Returns (X_tr, X_te, note, state);
    impute_apply(state, X) replays it on new rows."""
    state, note = fit_impute(X_tr, method, intlike_cols, n_neighbors)
    if state is None:
        return X_tr, X_te, note, None
    return impute_apply(state, X_tr), impute_apply(state, X_te), note, state


def impute_apply(state, X):
    X = X.copy()
    X[state["num"]] = state["imp"].transform(X[state["num"]])
    for c in state["int"]:  # keep integer semantics (age stays whole)
        X[c] = X[c].round().astype("Int64")
    return X


# --- multivariate outlier removal (train rows only) -------------------------

def remove_outliers(X_tr, y_tr, method, contamination=None):
    from sklearn.ensemble import IsolationForest
    contamination = CONTAMINATION if contamination is None else float(contamination)
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
    from functools import partial
    num = X_tr.select_dtypes("number")  # raw text columns pass through untouched
    k = float(k or 10)
    k = max(1, min(math.ceil(k * num.shape[1]) if 0 < k < 1 else int(k), num.shape[1]))  # k < 1: a share
    if num.shape[1] <= k:
        return X_tr, X_te, "fewer features than k -- nothing removed", None
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
            # seeded: unseeded mutual info picks different columns on every run
            score = partial(mutual_info_classif if task == "classification" else mutual_info_regression,
                            random_state=0)
            data = num
        sel = SelectKBest(score, k=k).fit(data, y)
        keep = num.columns[sel.get_support()].tolist()

    note = f"{method}: kept {len(keep)} of {num.shape[1]} feature(s), dropped {num.shape[1] - len(keep)}"
    keep += [c for c in X_tr.columns if c not in num.columns]
    return X_tr[keep], X_te[keep], note, keep


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
            raise ValueError("SMOTE needs an all-numeric feature matrix: encode categoricals first, and it "
                             "can't invent text rows -- use oversample or class_weights with text columns.")
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

def fit_reduce(X_tr, n, method="pca", scale=False):
    """Numeric columns -> n components, fitted on train. pca: principal components.
    n below 1 is a share of variance: as many components as keep it (for the
    autoencoder: as many units). scale=True standardises the columns first (fitted
    here, replayed by reduce_apply) -- the tested hybrid pipelines use it.
    autoencoder (plan §9.6 "autoencoder + classifier"): a one-hidden-layer network
    (scikit-learn MLPRegressor, its default ReLU) trained to rebuild its input from
    n units; the n unit activations become the features. Scale the numeric columns
    first -- a network learns poorly from unscaled numbers."""
    num = X_tr.select_dtypes("number")
    if num.shape[1] < 2:
        return None, "too few numeric features for reduction"
    scaler = None
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(num)
        num = pd.DataFrame(scaler.transform(num), columns=num.columns, index=num.index)
    share = None
    if n is not None and 0 < float(n) < 1:
        share = float(n)
        n = int(PCA(n_components=share, svd_solver="full").fit(num).n_components_)
    n = max(1, min(int(n or 2), num.shape[1]))
    kept = f" (keeps {share:.0%} of the variance)" if share else ""
    if method == "autoencoder":
        import warnings
        from sklearn.neural_network import MLPRegressor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # ConvergenceWarning at the default 200 epochs is reported below
            ae = MLPRegressor(hidden_layer_sizes=(n,), random_state=0).fit(num, num)
        state = {"ae": ae, "num": list(num.columns), "n": n, "prefix": "ae", "scaler": scaler}
        return state, (f"autoencoder -> {n} feature(s){kept}, rebuilds the training columns with "
                       f"R² {ae.score(num, num):.2f}")
    state = {"pca": PCA(n_components=n, random_state=0).fit(num), "num": list(num.columns), "n": n, "prefix": "pc",
             "scaler": scaler}
    var = round(float(state["pca"].explained_variance_ratio_.sum()) * 100, 1)
    return state, f"PCA -> {n} component(s), {var}% variance retained" + (" (standardised first)" if scale else "")


def reduce(X_tr, X_te, method, n, scale=False):
    state, note = fit_reduce(X_tr, n, method or "pca", scale)
    if state is None:
        return X_tr, X_te, note, None
    return reduce_apply(state, X_tr), reduce_apply(state, X_te), note, state


def reduce_apply(state, X):
    """Numeric columns -> components; anything else (raw text) passes through."""
    cols = X[state["num"]]
    if state.get("scaler") is not None:       # pipelines saved before "scale" existed have no scaler
        cols = pd.DataFrame(state["scaler"].transform(cols), columns=cols.columns, index=cols.index)
    num = cols.to_numpy(dtype="float64")
    if "ae" in state:   # encoder half: ReLU(X W + b) of the hidden layer
        Z = np.maximum(num @ state["ae"].coefs_[0] + state["ae"].intercepts_[0], 0)
    else:
        Z = state["pca"].transform(cols)
    prefix = state.get("prefix", "pc")      # pipelines saved before autoencoders have no prefix
    pcs = pd.DataFrame(Z, columns=[f"{prefix}{i + 1}" for i in range(Z.shape[1])], index=X.index)
    return pd.concat([pcs, X.drop(columns=state["num"])], axis=1)


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
    a_tr, a_te, note, _ = advanced_impute(Xi, Xte.copy(), "knn", {"age"})
    assert str(a_tr["age"].dtype) == "Int64" and a_tr["age"].notna().all(), note

    # isolation forest drops some train rows
    o_tr, o_y, _ = remove_outliers(Xtr, ytr, "isolation_forest")
    assert len(o_tr) < len(Xtr) and len(o_tr) == len(o_y)

    # feature selection keeps k
    for meth in ("correlation", "chi2", "anova", "mutual_info", "rfe"):
        s_tr, s_te, note, _ = select_features(Xtr, Xte, ytr, meth, 2, "classification")
        assert s_tr.shape[1] == 2, (meth, note)

    # over/under sampling balance the classes
    for meth in ("oversample", "undersample"):
        b_tr, b_y, note, _ = balance(Xtr, ytr, meth, "classification")
        vc = b_y.value_counts()
        assert vc.min() == vc.max(), (meth, vc.to_dict())
    _, _, _, weights = balance(Xtr, ytr, "class_weights", "classification")
    assert weights and len(weights) == 2

    # PCA reduces to n components on train + test
    r_tr, r_te, note, _ = reduce(Xtr, Xte, "pca", 2)
    assert list(r_tr.columns) == ["pc1", "pc2"] and len(r_te) == len(Xte), note
    # autoencoder: n learned features; the encoder is the network's own first layer
    from sklearn.preprocessing import StandardScaler
    Xs = pd.DataFrame(StandardScaler().fit_transform(Xtr), columns=Xtr.columns)
    a_tr, a_te, note, st = reduce(Xs, Xs.iloc[:10], "autoencoder", 2)
    assert list(a_tr.columns) == ["ae1", "ae2"] and len(a_te) == 10 and "autoencoder" in note, note
    hidden = st["ae"].coefs_[0].shape[1]
    assert hidden == 2 and (a_tr.to_numpy() >= 0).all()      # ReLU activations of the 2 hidden units

    # n < 1 = variance share; scale=True standardises first and is replayed on new rows
    wide = pd.DataFrame(rng.normal(size=(150, 6)) * [1, 10, 100, 1, 1, 1], columns=list("uvwxyz"))
    st, note = fit_reduce(wide, 0.95, "pca", scale=True)
    from sklearn.pipeline import make_pipeline
    ref = make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full")).fit(wide)
    assert st["n"] == ref[-1].n_components_ and "standardised" in note, (st["n"], note)
    assert np.allclose(np.abs(reduce_apply(st, wide).to_numpy()), np.abs(ref.transform(wide)), atol=1e-6)
    ae_st, note = fit_reduce(wide, 0.95, "autoencoder", scale=True)
    assert ae_st["n"] == st["n"] and "95%" in note and reduce_apply(ae_st, wide.iloc[:3]).shape == (3, st["n"])
    unscaled = fit_reduce(wide, 2, "pca")[0]                         # saved pipelines: no scaler, same as before
    assert unscaled["scaler"] is None and np.allclose(reduce_apply(unscaled, wide).to_numpy(),
                                                      PCA(n_components=2, random_state=0).fit(wide).transform(wide))
    # feature selection k < 1 = share of the numeric features kept
    yy = (wide["u"] > 0).astype(int)
    assert select_features(wide, wide, yy, "mutual_info", 0.5, "classification")[0].shape[1] == 3

    validate_pipeline({"feature_selection": {"method": "chi2", "k": 3}})
    try:
        validate_pipeline({"reduction": {"method": "nope"}}); raise AssertionError("should reject")
    except ValueError:
        pass
    print("advanced self-check passed")
