"""Exact SHAP values for scikit-learn's own tree models (decision tree, random forest,
extra trees, gradient boosting, histogram gradient boosting) -- the `shap` package is
not installed, and XGBoost / LightGBM / CatBoost bring their own TreeSHAP.

What is computed: path-dependent TreeSHAP (Lundberg, Erion & Lee 2018, "Consistent
individualized feature attribution for tree ensembles", Algorithm 2) -- the same
quantity the shap library's TreeExplainer and the boosters' pred_contribs give. The
value of a set S of features for one leaf L is v_L * prod over L's path features of
(row satisfies the path's splits on it ? 1 : share of training rows that went that
way); features outside the path are null players. Its Shapley value for feature f is

    v_L (o_f - z_f) * sum over S in P\\{f} of |S|!(d-|S|-1)!/d! prod_S o_k prod_rest z_k

(o = 1/0 the row follows the path's splits on that feature, z = training share, d =
features on the path). The sum is computed with Algorithm 2's EXTEND and UNWOUND-
PATH-SUM recurrences, whose path weights already carry the subset-size factors and
stay between 0 and 1 (raw polynomial coefficients lost all precision on 60-feature
text paths -- measured), vectorised here over leaves and rows. The self-check
compares it with brute-force Shapley values (every subset) and checks additivity
(contributions + expected value = the model's output) for every supported model,
with missing values and with deep paths over wide text features.

Output space: probability for decision tree / forest classifiers (what predict_proba
averages), log-odds (raw score) for gradient boosting classifiers, target units for
regressors.
"""

import math

import numpy as np

CHUNK_CELLS = 2_000_000   # our starting point: leaves x rows x path length held at once (~16 MB per array)


def _sk_tree(est, classifier):
    """A fitted sklearn tree -> plain arrays. Leaf values: class shares (classifier
    trees) or the leaf output (regression trees, also inside gradient boosting)."""
    t = est.tree_
    value = t.value[:, 0, :] if t.value.shape[1] == 1 else t.value[:, :, 0]
    if classifier:
        value = value / np.maximum(value.sum(axis=1, keepdims=True), 1e-300)
    nan_left = getattr(t, "missing_go_to_left", None)
    return {"left": t.children_left, "right": t.children_right, "feature": t.feature,
            "threshold": t.threshold, "cover": t.weighted_n_node_samples.astype(float),
            "nan_left": np.ones(t.node_count, bool) if nan_left is None else np.asarray(nan_left, bool),
            "value": np.asarray(value, float), "dtype": np.float32}   # sklearn trees compare float32 inputs


def _hgb_tree(pred):
    n = pred.nodes
    if n["is_categorical"].any():
        raise ValueError("categorical splits are not supported")
    leaf = n["is_leaf"].astype(bool)
    return {"left": np.where(leaf, -1, n["left"].astype(np.intp)), "right": np.where(leaf, -1, n["right"].astype(np.intp)),
            "feature": n["feature_idx"].astype(np.intp), "threshold": n["num_threshold"].astype(float),
            "cover": n["count"].astype(float), "nan_left": n["missing_go_to_left"].astype(bool),
            "value": n["value"].astype(float)[:, None], "dtype": np.float64}


def ensemble(model):
    """model -> (trees, output, units): trees = [(tree arrays, output index, weight)],
    output = number of outputs (1 for regression and binary classification, one per
    class otherwise), plus the constant added to the trees' sum. None = unsupported."""
    from sklearn import ensemble as E, tree as T
    name = type(model).__name__
    clf = name.endswith("Classifier")
    if isinstance(model, (T.DecisionTreeClassifier, T.DecisionTreeRegressor, T.ExtraTreeClassifier,
                          T.ExtraTreeRegressor)):
        ests = [model]
    elif isinstance(model, (E.RandomForestClassifier, E.RandomForestRegressor, E.ExtraTreesClassifier,
                            E.ExtraTreesRegressor)):
        ests = model.estimators_
    else:
        ests = None
    if ests is not None:
        if clf and len(model.classes_) == 2:      # one output: the share of class index 1
            trees = [_sk_tree(e, True) for e in ests]
            return [({**t, "value": t["value"][:, 1:]}, 0, 1 / len(ests)) for t in trees], 1, 0.0, "probability"
        trees = [(_sk_tree(e, clf), 0, 1 / len(ests)) for e in ests]
        return trees, (len(model.classes_) if clf else 1), 0.0, "probability" if clf else "target units"
    if isinstance(model, (E.GradientBoostingClassifier, E.GradientBoostingRegressor)):
        k = model.estimators_.shape[1]
        trees = [(_sk_tree(model.estimators_[i, j], False), j, model.learning_rate)
                 for i in range(model.estimators_.shape[0]) for j in range(k)]
        return trees, k, None, "log-odds" if clf else "target units"   # None: init offset read per model
    if isinstance(model, (E.HistGradientBoostingClassifier, E.HistGradientBoostingRegressor)):
        trees = [(_hgb_tree(p), j, 1.0) for it in model._predictors for j, p in enumerate(it)]
        k = len(model._predictors[0]) if model._predictors else 1
        return trees, k, np.asarray(model._baseline_prediction, float).ravel(), "log-odds" if clf else "target units"
    return None


def supports(model):
    try:
        return ensemble(model) is not None
    except (ValueError, AttributeError):
        return False


def _leaf_paths(T):
    """[(leaf, {feature: (lo, hi, nan_follows, z)})]: a row follows the leaf's splits on
    feature f when lo < x <= hi (a missing x: when every split sent missing values this
    way); z = the share of training rows (cover) that went this way on f's splits."""
    out, stack = [], [(0, {})]
    left, right, feat, thr, cover, nan_left = (T[k] for k in ("left", "right", "feature", "threshold",
                                                              "cover", "nan_left"))
    while stack:
        j, cond = stack.pop()
        if left[j] == -1:
            out.append((j, cond))
            continue
        f, t = int(feat[j]), float(thr[j])
        for child, goes_left in ((left[j], True), (right[j], False)):
            lo, hi, nan_ok, z = cond.get(f, (-np.inf, np.inf, True, 1.0))
            c = dict(cond)
            c[f] = (lo, min(hi, t), nan_ok and bool(nan_left[j]), z * cover[child] / cover[j]) if goes_left else \
                   (max(lo, t), hi, nan_ok and not nan_left[j], z * cover[child] / cover[j])
            stack.append((child, c))
    return out


def tree_shap(T, X, n_features):
    """Contributions of one tree: rows x outputs-of-the-leaf x n_features, and its
    expected value. X: dense rows (columns = the model's features)."""
    paths = _leaf_paths(T)
    R, K = X.shape[0], T["value"].shape[1]
    phi = np.zeros((n_features, R, K))
    root = T["cover"][0]
    expected = sum(T["value"][leaf] * T["cover"][leaf] / root for leaf, _ in paths)
    Xc = X.astype(T["dtype"], copy=False)
    nan = np.isnan(Xc)
    by_depth = {}
    for leaf, cond in paths:
        if cond:
            by_depth.setdefault(len(cond), []).append((leaf, cond))
    for d, group in by_depth.items():
        chunk = max(1, CHUNK_CELLS // (max(R, 1) * (d + 1)))
        idx = np.arange(d)
        for start in range(0, len(group), chunk):
            part = group[start:start + chunk]
            F = np.array([list(c) for _, c in part])                                        # L x d
            lo, hi, nok, Z = (np.array([[c[f][i] for f in c] for _, c in part]) for i in range(4))
            V = T["value"][[leaf for leaf, _ in part]]                                       # L x K
            x, xn = Xc[:, F], nan[:, F]                                                       # R x L x d
            O = np.where(xn, nok[None].astype(bool), (x > lo[None]) & (x <= hi[None]))
            O = O.transpose(1, 0, 2).astype(float)                                           # L x R x d
            Zb = Z[:, None, :]                                                                # L x 1 x d
            # EXTEND: element 0 is the root's placeholder (z = o = 1), then the d path features
            pw = np.zeros((len(part), R, d + 1))
            pw[..., 0] = 1.0
            for u in range(1, d + 1):
                z, o = Zb[..., u - 1], O[..., u - 1]
                for i in range(u - 1, -1, -1):
                    pw[..., i + 1] += o * pw[..., i] * ((i + 1) / (u + 1))
                    pw[..., i] *= z * ((u - i) / (u + 1))
            # UNWOUND-PATH-SUM: the path weights without feature k (summed), times (o - z) and the leaf
            # value. A row that doesn't follow k's splits (o = 0): divide k's z out, no recursion. One
            # that does (o = 1): undo  pw[i] = z*w[i]*(d-i)/(d+1) + w[i-1]*i/(d+1).  Algorithm 2 runs it
            # from the top, which multiplies rounding errors by z*(d-i)/i per step and blew up past
            # ~45 path features (measured: error 1e7 at 94). From the bottom the factor is
            # i/(z*(d-i)): each direction shrinks errors on its side of i = z*d/(1+z), so each side
            # is taken from the direction that is stable there -- the same numbers, exactly.
            off_sum = (pw[..., :d] * ((d + 1) / (d - idx))).sum(-1)        # o = 0 total, before / z
            up, down = np.empty((len(part), R, d)), np.empty((len(part), R, d))
            for k in range(d):
                z, o = Zb[..., k], O[..., k]                                 # L x 1, L x R
                cross = z * d / (1 + z)                                      # <= d/2 (z <= 1)
                n_up, i_lo = int(np.ceil(cross.max())), int(np.floor(cross.min()))
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    up[..., 0] = pw[..., 0] * (d + 1) / (z * d)              # only below the crossover
                    for i in range(1, n_up):
                        up[..., i] = (pw[..., i] - up[..., i - 1] * (i / (d + 1))) * (d + 1) / (z * (d - i))
                    down[..., d - 1] = pw[..., d] * ((d + 1) / d)            # only above it
                    for i in range(d - 1, max(i_lo, 0), -1):
                        down[..., i - 1] = (pw[..., i] - z * down[..., i] * ((d - i) / (d + 1))) * ((d + 1) / i)
                    on_sum = np.where(idx < cross[..., None], up, down).sum(-1)
                total = np.where(o > 0, on_sum, off_sum / z)
                np.add.at(phi, F[:, k], ((o - z) * total)[:, :, None] * V[:, None, :])
    return phi.transpose(1, 2, 0), np.asarray(expected, float)


def shap_values(model, X, check=lambda: None, budget_s=None, min_rows=1):
    """rows x outputs x features contributions and the expected value per output, for a
    supported scikit-learn tree model. X: the model's input rows (dense or sparse);
    only the columns its splits use are read. Time grows with leaves x rows x path
    length squared (measured 2026-10-31: a 100-tree forest on 20,000 rows, 258k
    leaves, took 227 s for 300 rows); with budget_s, once the first trees show the
    whole forest would take longer, only the first rows (at least min_rows) are kept
    -- the result then has fewer rows than X."""
    import time

    from scipy.sparse import issparse
    trees, k, offset, _units = ensemble(model)
    used = sorted({int(f) for T, _, _ in trees for f, l in zip(T["feature"], T["left"]) if l != -1})
    pos = {f: i for i, f in enumerate(used)}
    Xu = X[:, used]
    Xu = Xu.toarray() if issparse(Xu) else np.asarray(Xu, float)
    phi_u = np.zeros((X.shape[0], k, len(used)))
    base = np.zeros(k)
    remap = np.array([pos.get(int(f), 0) for f in range(max(used, default=0) + 1)])
    t0, cut = time.time(), False
    for i, (T, out, weight) in enumerate(trees):
        check()
        spent = time.time() - t0
        if budget_s and not cut and i >= 2 and spent / i * len(trees) > budget_s and len(Xu) > min_rows:
            keep = max(min_rows, int(len(Xu) * budget_s / (spent / i * len(trees))))
            Xu, phi_u, cut = Xu[:keep], phi_u[:keep], True
        T = {**T, "feature": np.where(T["left"] == -1, 0, remap[np.maximum(T["feature"], 0)])}
        p, e = tree_shap(T, Xu, len(used))
        if p.shape[1] == 1:
            phi_u[:, out] += weight * p[:, 0]
            base[out] += weight * e[0]
        else:
            phi_u += weight * p
            base += weight * e
    if offset is None:   # gradient boosting: the initial raw score (its init estimator's prior)
        offset = np.asarray(model._raw_predict_init(X[:1]), float).ravel()
    return phi_u, used, base + offset


if __name__ == "__main__":
    import itertools
    import warnings

    from sklearn import ensemble as E, tree as Tr
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    n, m = 400, 5
    X = rng.normal(size=(n, m))
    y_reg = X[:, 0] * 2 + X[:, 1] * X[:, 2] + rng.normal(0, .1, n)
    y_bin = (X[:, 0] + X[:, 3] > 0).astype(int)
    y_multi = np.digitize(X[:, 0] + X[:, 1], [-0.7, 0.7])

    # 1. brute force: every subset, the path-dependent value function walked through the tree
    def value(T, x, S):
        def go(j):
            if T["left"][j] == -1:
                return T["value"][j]
            f = T["feature"][j]
            if f in S:
                return go(T["left"][j] if x[f] <= T["threshold"][j] else T["right"][j])
            c = T["cover"]
            return (c[T["left"][j]] * go(T["left"][j]) + c[T["right"][j]] * go(T["right"][j])) / c[j]
        return go(0)

    def brute(T, x):
        out = np.zeros((m,) + T["value"].shape[1:])
        for f in range(m):
            others = [g for g in range(m) if g != f]
            for s in range(m):
                for S in itertools.combinations(others, s):
                    w = math.factorial(s) * math.factorial(m - s - 1) / math.factorial(m)
                    out[f] += w * (value(T, x, set(S) | {f}) - value(T, x, set(S)))
        return out

    dt = Tr.DecisionTreeRegressor(max_depth=6, random_state=0).fit(X, y_reg)
    T = _sk_tree(dt, False)
    T["dtype"] = np.float64
    phi, e = tree_shap(T, X[:12], m)
    for r in range(12):
        assert np.allclose(phi[r, 0], brute(T, X[r])[:, 0], atol=1e-9), r

    # 2. additivity for every supported model: contributions + expected value = the model's output
    Xn = X.copy()
    Xn[rng.random(Xn.shape) < 0.08] = np.nan                       # missing values follow the trees' routing
    cases = [
        (Tr.DecisionTreeClassifier(random_state=0).fit(X, y_multi), X, lambda mdl, Z: mdl.predict_proba(Z)),
        (E.RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y_bin), X,
         lambda mdl, Z: mdl.predict_proba(Z)[:, 1:]),
        (E.RandomForestClassifier(n_estimators=15, random_state=0).fit(Xn, y_multi), Xn,
         lambda mdl, Z: mdl.predict_proba(Z)),
        (E.ExtraTreesRegressor(n_estimators=15, random_state=0).fit(X, y_reg), X, lambda mdl, Z: mdl.predict(Z)[:, None]),
        (E.GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X, y_bin), X,
         lambda mdl, Z: mdl.decision_function(Z)[:, None]),
        (E.GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X, y_multi), X,
         lambda mdl, Z: mdl.decision_function(Z)),
        (E.GradientBoostingRegressor(n_estimators=30, random_state=0).fit(X, y_reg), X,
         lambda mdl, Z: mdl.predict(Z)[:, None]),
        (E.HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(Xn, y_multi), Xn,
         lambda mdl, Z: mdl.decision_function(Z)),
        (E.HistGradientBoostingRegressor(max_iter=30, random_state=0).fit(Xn, y_reg), Xn,
         lambda mdl, Z: mdl.predict(Z)[:, None]),
    ]
    for mdl, data, out in cases:
        rows = data[:60]
        phi, used, base = shap_values(mdl, rows)
        assert np.allclose(phi.sum(axis=2) + base, out(mdl, rows), atol=1e-6), type(mdl).__name__
    # the signal column gets the largest mean |SHAP| (y_bin = x0 + x3)
    phi, used, _ = shap_values(cases[1][0], X[:200])
    top = [used[i] for i in np.argsort(-np.abs(phi).mean(axis=(0, 1)))[:2]]
    assert set(top) == {0, 3}, top
    # sparse input (text features): only the split columns are read
    from scipy.sparse import csr_matrix, hstack
    sp = csr_matrix(np.abs(X))
    rf = E.RandomForestRegressor(n_estimators=10, random_state=0).fit(sp, y_reg)
    phi, used, base = shap_values(rf, sp[:30])
    assert np.allclose(phi.sum(axis=2)[:, 0] + base[0], rf.predict(sp[:30]), atol=1e-6)
    # deep paths over wide text features (60+ features on one path): still exact
    from sklearn.feature_extraction.text import TfidfVectorizer
    words = [f"{'urgent' if i % 4 == 0 else 'fine'} item {i}" for i in range(n)]
    tx = hstack([csr_matrix(X[:, :2]), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(words)]).tocsr()
    rft = E.RandomForestClassifier(n_estimators=20, random_state=0).fit(tx, y_bin)
    assert max(len(c) for t in rft.estimators_ for _, c in _leaf_paths(_sk_tree(t, True))) > 30
    phi, used, base = shap_values(rft, tx[:40])
    assert np.allclose(phi.sum(axis=2) + base, rft.predict_proba(tx[:40])[:, 1:], atol=1e-6)
    assert supports(cases[0][0]) and not supports(object())
    # a time budget keeps the first rows only - still exact for them
    big = E.RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y_reg)
    phi, used, base = shap_values(big, X[:300], budget_s=1e-4, min_rows=20)
    assert 20 <= len(phi) < 300 and np.allclose(phi.sum(axis=2)[:, 0] + base[0], big.predict(X[:len(phi)]), atol=1e-6)
    print("treeshap self-check passed (brute force + additivity on", len(cases), "models)")
