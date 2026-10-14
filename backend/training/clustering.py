"""Clustering (plan §9.3, §11.3; spec 2026-09-13 §7 piece 4): K-Means,
hierarchical (agglomerative), DBSCAN and Gaussian mixture on the preprocessed
rows. No target and no train/test split: every row gets a cluster.

Metrics measure compactness / separation only -- they can't say the groups are
meaningful (said on the page):
  silhouette        -1..1, higher = better separated (sampled on large data)
  Davies-Bouldin    >= 0, lower = better
  Calinski-Harabasz >= 0, higher = better
  (Davies-Bouldin and Calinski-Harabasz need dense data: n/a with text columns.)
Charts: elbow + silhouette per k (K-Means scan), silhouette plot, 2-D view (PCA, or
truncated SVD for sparse text data), dendrogram (agglomerative), k-distance curve
(for DBSCAN's eps).

Defaults: the number of clusters is COMPUTED -- the k with the best silhouette in
the elbow scan -- because the libraries' own defaults (K-Means 8, agglomerative 2,
Gaussian mixture 1) are arbitrary. DBSCAN keeps scikit-learn's published defaults.
"""

import ctypes
import hashlib
import json
import os
import sys
import time

import joblib
import numpy as np
from scipy.cluster import hierarchy
from scipy.sparse import issparse
from sklearn import cluster, mixture
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from core import store
from preprocessing import execute
from training import params as P
from training import train
from training.params import spec
from training.pipeline import Prep

ELBOW_MAX_K = 10           # spec §8: k from 2 to min(10, rows - 1), our starting point
SILHOUETTE_SAMPLE = 10_000  # spec §8: rows sampled for the silhouette score, our starting point
PLOT_ROWS = 2000           # our starting point: points in the 2-D view / silhouette plot
DENDRO_ROWS = 300          # our starting point: rows a dendrogram is drawn from (readable)
DENDRO_LEAVES = 30         # our starting point: merges shown at the bottom (truncated)
RAM_SHARE = 0.5            # spec §8: share of free RAM a fit may plan to use
AGGLO_FACTOR = 1.2         # spec §7: measured 1.13-1.17x the condensed distance matrix (tracemalloc)
METRICS = {"silhouette": ("Silhouette", True), "davies_bouldin": ("Davies-Bouldin", False),
           "calinski_harabasz": ("Calinski-Harabasz", True)}


def free_ram():
    """Available physical memory in bytes, or None when the OS doesn't say (stdlib only)."""
    try:
        if sys.platform == "win32":
            class _Mem(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _Mem()
            m.dwLength = ctypes.sizeof(_Mem)
            return int(m.ullAvailPhys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) else None
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


def agglomerative_cap(ram=None):
    """Largest row count whose distance matrix fits the RAM share (spec §7):
    1.2 x n(n-1)/2 x 8 bytes <= share of free RAM. None when RAM is unknown."""
    ram = free_ram() if ram is None else ram
    if not ram:
        return None
    budget = RAM_SHARE * ram / (AGGLO_FACTOR * 8)
    return int((1 + (1 + 8 * budget) ** 0.5) / 2)


def table(k):
    """Clustering rows (training/models.py shape). k = the computed number of clusters."""
    nk = spec("n_clusters", "Clusters (k)", "int", 2, 100, help="how many groups to form")
    return {
        "kmeans": {"label": "K-Means", "family": "Centroid", "sparse": True,
                   "make": lambda seed: cluster.KMeans(n_clusters=k, n_init="auto", random_state=seed),
                   "params": [nk], "predicts": True,
                   "about": "Puts each row with the nearest of k centres; centres move to their rows' mean. "
                            "Expects round, similar-sized groups; scale numeric columns first."},
        "agglomerative": {"label": "Hierarchical (agglomerative)", "family": "Hierarchical", "sparse": False,
                          "make": lambda seed: cluster.AgglomerativeClustering(n_clusters=k),
                          "params": [nk, spec("linkage", "Linkage", "choice",
                                              choices=["ward", "complete", "average", "single"])],
                          "predicts": False,
                          "about": "Starts with every row alone and repeatedly merges the two closest groups; "
                                   "the dendrogram shows the merges. Memory grows with rows squared."},
        "dbscan": {"label": "DBSCAN", "family": "Density", "sparse": True,
                   "make": lambda seed: cluster.DBSCAN(),
                   "params": [spec("eps", "Neighbourhood radius (eps)", "float", 1e-6, 1e6,
                                   help="pick it where the k-distance curve bends"),
                              spec("min_samples", "Min rows per dense area", "int", 1, 1000)],
                   "predicts": False,
                   "about": "Groups rows in dense areas and marks isolated rows as noise (-1); finds any "
                            "shape and decides the number of clusters itself."},
        "gaussian_mixture": {"label": "Gaussian mixture", "family": "Probabilistic", "sparse": False,
                             "make": lambda seed: mixture.GaussianMixture(n_components=k, random_state=seed),
                             "params": [spec("n_components", "Components (k)", "int", 1, 100),
                                        spec("covariance_type", "Covariance", "choice",
                                             choices=["full", "tied", "diag", "spherical"])],
                             "predicts": True,
                             "about": "Fits k overlapping bell-shaped groups; each row gets the most likely one "
                                      "(soft membership). Needs dense data."},
    }


def _context(ds_id):
    df, prep = store.load_frame(ds_id), store.load_prep(ds_id)
    if df is None:
        raise LookupError("Dataset not found. Re-upload.")
    if prep is None:
        raise LookupError("Run preprocessing first.")
    fitted = prep["fitted"]
    if fitted["clean_ops"] != store.clean_ops(ds_id):
        raise ValueError("The data was cleaned again after preprocessing - run preprocessing again.")
    s = fitted["settings"]
    if s["task"] != "clustering":
        raise ValueError("This data was preprocessed for a target - choose 'clustering' on the Preprocessing tab.")
    df, feats, _, notes = execute.prepare_rows(df, None, s["columns"], "clustering")
    X = df[feats]
    prep_step = Prep(s["columns"], s["pipeline"], "clustering")
    M = prep_step.fit_transform(X)
    return {"X": X, "M": M, "prep": prep_step, "settings": s, "seed": s["random_state"], "notes": notes,
            "clean_ops": fitted["clean_ops"], "features": feats}


def _sample(n, cap, seed):
    return np.arange(n) if n <= cap else np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def _metrics(M, labels, seed):
    """{silhouette, davies_bouldin, calinski_harabasz} on clustered (non-noise) rows; None when not computable."""
    keep = labels != -1
    Mk, lk = M[keep], labels[keep]
    k = len(set(lk))
    out = {m: None for m in METRICS}
    if k < 2 or k >= len(lk):
        return out, "fewer than 2 clusters (or one row per cluster): scores need 2 or more groups"
    out["silhouette"] = float(silhouette_score(Mk, lk, sample_size=min(len(lk), SILHOUETTE_SAMPLE),
                                               random_state=seed))
    if issparse(Mk):
        return out, "Davies-Bouldin and Calinski-Harabasz need dense data (text columns are sparse)"
    out["davies_bouldin"] = float(davies_bouldin_score(Mk, lk))
    out["calinski_harabasz"] = float(calinski_harabasz_score(Mk, lk))
    return out, None


def elbow(M, seed, n):
    """K-Means inertia and silhouette for k = 2 .. min(ELBOW_MAX_K, n - 1)."""
    ks = list(range(2, min(ELBOW_MAX_K, n - 1) + 1))
    out = {"k": ks, "inertia": [], "silhouette": []}
    for k in ks:
        km = cluster.KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(M)
        out["inertia"].append(float(km.inertia_))
        out["silhouette"].append(float(silhouette_score(M, km.labels_, sample_size=min(n, SILHOUETTE_SAMPLE),
                                                        random_state=seed)))
    out["best_k"] = ks[int(np.argmax(out["silhouette"]))] if ks else 2
    return out


def k_distance(M, min_samples, seed):
    """Sorted distance from each (sampled) row to its min_samples-th neighbour --
    the bend of this curve is a usual choice for DBSCAN's eps (Ester et al. 1996)."""
    idx = _sample(M.shape[0], PLOT_ROWS, seed)
    nn = NearestNeighbors(n_neighbors=min(min_samples, len(idx) - 1)).fit(M[idx])
    d = np.sort(nn.kneighbors(M[idx])[0][:, -1])
    pts = np.unique(np.linspace(0, len(d) - 1, min(len(d), 200)).round().astype(int))
    return {"min_samples": min_samples, "rank": pts.tolist(), "distance": [round(float(d[i]), 6) for i in pts]}


def facts(ctx):
    M = ctx["M"]
    values = M.data if issparse(M) else np.asarray(M)
    return {"rows": int(M.shape[0]), "features": int(M.shape[1]), "sparse": bool(issparse(M)),
            "nan": bool(np.isnan(values).any())}


def _incompatible(key, row, f, cap):
    if f["nan"]:
        return "can't handle missing values - add an impute step on the Preprocessing tab"
    if f["sparse"] and not row["sparse"]:
        return "needs dense data - your data has text columns (or add a PCA reduction step)"
    if key == "agglomerative" and cap is not None and f["rows"] > cap:
        return f"needs about {AGGLO_FACTOR * 8 * f['rows'] ** 2 / 2 / 1e9:.1f} GB for {f['rows']:,} rows " \
               f"(fits up to {cap:,} rows in half the free memory now)"
    return None


def fingerprint(ctx):
    s = ctx["settings"]
    key = {"clean_ops": ctx["clean_ops"], "prep": {k: s.get(k) for k in ("columns", "pipeline", "task", "random_state")}}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]


def options(ds_id):
    ctx = _context(ds_id)
    f = facts(ctx)
    if f["rows"] < 3:
        raise ValueError("Clustering needs at least 3 rows.")
    scan = None if f["nan"] else elbow(ctx["M"], ctx["seed"], f["rows"])
    k = scan["best_k"] if scan else 2
    cap = agglomerative_cap()
    listed = []
    for key, row in table(k).items():
        defaults = P.defaults(row)
        reason = _incompatible(key, row, f, cap)
        listed.append({"key": key, "label": row["label"], "family": row["family"], "compatible": reason is None,
                       "reason": reason, "about": row["about"], "predicts": row["predicts"],
                       "params": [{**p, "default": defaults[p["name"]], "help": p["help"]} for p in row["params"]]})
    return {"task": "clustering", "target": None, "notes": ctx["notes"], "facts": f, "fingerprint": fingerprint(ctx),
            "models": listed, "elbow": scan, "k_distance": None if f["nan"] else k_distance(ctx["M"], 5, ctx["seed"]),
            "k_note": f"default k = {k}: the best silhouette in the K-Means scan (computed)",
            "metrics": [{"key": m, "label": v[0], "higher_is_better": v[1]} for m, v in METRICS.items()],
            "defaults": {"primary_metric": "silhouette", "seed": ctx["seed"], "time_limit_min": 60},
            "agglomerative_cap": cap}


def validate(ds_id, payload, kind="cluster"):
    opt = options(ds_id)
    by_key = {m["key"]: m for m in opt["models"]}
    chosen = list(dict.fromkeys(payload.get("models") or []))
    if not chosen:
        raise ValueError("Pick at least one clustering method.")
    k = opt["elbow"]["best_k"] if opt["elbow"] else 2
    rows = table(k)
    params = {}
    for key in chosen:
        if key not in by_key:
            raise ValueError(f"Unknown clustering method '{key}'.")
        if not by_key[key]["compatible"]:
            raise ValueError(f"{by_key[key]['label']} can't run on this data: {by_key[key]['reason']}.")
        params[key] = P.resolve(rows[key], (payload.get("params") or {}).get(key))[2]
        n = params[key].get("n_clusters") or params[key].get("n_components")
        if n is not None and n >= opt["facts"]["rows"]:
            raise ValueError(f"{rows[key]['label']}: k must be smaller than the number of rows.")
    primary = payload.get("primary_metric") or "silhouette"
    if primary not in METRICS:
        raise ValueError(f"Unknown metric '{primary}'.")
    minutes = float(payload.get("time_limit_min") or 60)
    return {"kind": kind, "task": "clustering", "models": chosen, "params": params, "k": k,
            "primary_metric": primary, "time_limit_s": minutes * 60}


def _scatter(M, labels, seed):
    idx = _sample(M.shape[0], PLOT_ROWS, seed)
    reducer = TruncatedSVD(2, random_state=seed) if issparse(M) else PCA(2, random_state=seed)
    if M.shape[1] < 2:
        xy = np.c_[np.asarray(M[idx].todense() if issparse(M) else M[idx]).ravel(), np.zeros(len(idx))]
        how = "the only feature (no second axis)"
    else:
        xy = reducer.fit_transform(M)[idx]
        how = "truncated SVD (text data)" if issparse(M) else "PCA"
    return {"x": np.round(xy[:, 0], 5).tolist(), "y": np.round(xy[:, 1], 5).tolist(),
            "label": labels[idx].tolist(), "method": how}


def _silhouette_plot(M, labels, seed):
    keep = labels != -1
    if len(set(labels[keep])) < 2:
        return None
    idx = _sample(int(keep.sum()), PLOT_ROWS, seed)
    Mk, lk = M[keep][idx], labels[keep][idx]
    s = silhouette_samples(Mk, lk)
    out = []
    for c in sorted(set(lk)):
        v = np.sort(s[lk == c])[::-1]
        pick = np.unique(np.linspace(0, len(v) - 1, min(len(v), 200)).round().astype(int))
        out.append({"cluster": int(c), "size": int(len(v)), "values": np.round(v[pick], 4).tolist()})
    return out


def _dendrogram(M, method, seed):
    """scipy linkage on a sample -> the line segments of a truncated dendrogram."""
    idx = _sample(M.shape[0], DENDRO_ROWS, seed)
    D = np.asarray(M[idx].todense()) if issparse(M) else np.asarray(M[idx])
    Z = hierarchy.linkage(D, method=method)
    d = hierarchy.dendrogram(Z, truncate_mode="lastp", p=DENDRO_LEAVES, no_plot=True)
    return {"icoord": d["icoord"], "dcoord": d["dcoord"], "leaves": d["ivl"], "rows": int(len(idx)),
            "method": method}


def run(job, ds_id, settings):
    ctx = _context(ds_id)
    f = facts(ctx)
    M, seed = ctx["M"], ctx["seed"]
    rows = table(settings["k"])
    primary = settings["primary_metric"]
    higher = METRICS[primary][1]
    results = {"task": "clustering", "primary_metric": primary, "seed": seed, "fingerprint": fingerprint(ctx),
               "facts": f, "notes": ctx["notes"], "features": ctx["features"], "clean_ops": ctx["clean_ops"],
               "elbow": None if f["nan"] else elbow(M, seed, f["rows"]), "models": {}, "best": None,
               "versions": train.versions()}
    (job.dir / "models").mkdir(exist_ok=True)
    best = None
    for i, key in enumerate(settings["models"]):
        job.check()
        job.progress(model=rows[key]["label"], model_index=i + 1, models=len(settings["models"]))
        row = rows[key]
        rec = {"label": row["label"], "family": row["family"], "params": settings["params"][key],
               "predicts": row["predicts"]}
        try:
            kwargs, _fit, _ = P.resolve(row, settings["params"][key])
            model = row["make"](seed).set_params(**kwargs)
            t = time.time()
            labels = np.asarray(model.fit_predict(np.asarray(M) if not issparse(M) else M), dtype=int)
            rec["fit_seconds"] = round(time.time() - t, 3)
            scores, why = _metrics(M, labels, seed)
            sizes = {int(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))}
            rec.update(scores=scores, scores_note=why, clusters=len([c for c in sizes if c != -1]),
                       noise=sizes.get(-1, 0), sizes=sizes, scatter=_scatter(M, labels, seed),
                       silhouette_plot=_silhouette_plot(M, labels, seed))
            if key == "agglomerative":
                rec["dendrogram"] = _dendrogram(M, kwargs.get("linkage", "ward"), seed)
            store.write_json(job.dir / f"labels_{key}.json", labels.tolist())
            joblib.dump(Pipeline([("prep", ctx["prep"]), ("model", model)]), job.dir / "models" / f"{key}.joblib")
            v = scores.get(primary)
            if v is not None and (best is None or (v > best[0] if higher else v < best[0])):
                best = (v, key)
                results["best"] = key
        except Exception as exc:  # noqa: BLE001 -- one method failing must not end the run
            rec["error"] = f"{type(exc).__name__}: {exc}"
        results["models"][key] = rec
        store.write_json(job.dir / "results.json", results)


def assignments(ds_id, run_id, key):
    """The rows that were clustered, with their cluster (-1 = DBSCAN noise), as a DataFrame."""
    labels = store.read_json(store.run_dir(ds_id, run_id) / f"labels_{key}.json")
    if labels is None:
        raise LookupError("No cluster labels for this method in this run.")
    ctx = _context(ds_id)
    if len(labels) != len(ctx["X"]):
        raise ValueError("The data changed since this run - cluster again.")
    return ctx["X"].assign(cluster=labels)


if __name__ == "__main__":
    import tempfile
    import warnings
    from pathlib import Path

    import pandas as pd
    from core import jobs
    warnings.filterwarnings("ignore")
    store.ROOT = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [6, 6], [0, 8]])
    pts = np.vstack([c + rng.normal(0, .7, (80, 2)) for c in centers])
    df = pd.DataFrame(pts, columns=["x1", "x2"]).assign(kind=np.repeat(["a", "b", "c"], 80))

    ds = store.create(df, "blobs.csv")
    r = execute.apply_plan(df, None, {"x1": [{"op": "scale", "method": "standard"}],
                                      "x2": [{"op": "scale", "method": "standard"}],
                                      "kind": [{"op": "drop_column"}]}, task="clustering")
    r["fitted"]["clean_ops"] = []
    store.save_prep(ds, r["csv"], r["fitted"])

    o = options(ds)
    assert o["elbow"]["best_k"] == 3, o["elbow"]                    # three blobs -> k = 3 computed
    by = {m["key"]: m for m in o["models"]}
    assert all(by[k]["compatible"] for k in by), [(k, by[k]["reason"]) for k in by]
    assert {p["name"]: p["default"] for p in by["kmeans"]["params"]}["n_clusters"] == 3
    assert {p["name"]: p["default"] for p in by["dbscan"]["params"]} == {"eps": 0.5, "min_samples": 5}
    s = validate(ds, {"models": ["kmeans", "agglomerative", "dbscan", "gaussian_mixture"]})
    run_id = jobs.start(ds, lambda job: run(job, ds, s), s, 300)
    jobs.wait(run_id)
    assert jobs.status(ds, run_id)["state"] == "done", jobs.status(ds, run_id)
    res = store.read_json(store.run_dir(ds, run_id) / "results.json")
    km = res["models"]["kmeans"]
    assert km["clusters"] == 3 and sorted(km["sizes"].values()) == [80, 80, 80], km["sizes"]
    assert km["scores"]["silhouette"] > 0.7 and km["scores"]["davies_bouldin"] < 0.5
    # the reported silhouette is scikit-learn's on the same rows and labels
    ctx = _context(ds)
    labels = np.array(store.read_json(store.run_dir(ds, run_id) / "labels_kmeans.json"))
    assert abs(km["scores"]["silhouette"] - silhouette_score(ctx["M"], labels)) < 1e-9
    assert res["models"]["agglomerative"]["dendrogram"]["leaves"] and res["best"] in res["models"]
    assert len(km["scatter"]["x"]) == 240 and km["silhouette_plot"][0]["values"]
    assigned = assignments(ds, run_id, "kmeans")
    assert len(assigned) == 240 and assigned["cluster"].nunique() == 3
    # a fitted K-Means pipeline places raw rows like the training rows
    pipe = joblib.load(store.run_dir(ds, run_id) / "models" / "kmeans.joblib")
    assert (pipe.predict(df[["x1", "x2"]].head(5)) == labels[:5]).all()

    # DBSCAN with a tiny eps marks everything as noise: scores n/a, explained -- no crash
    s2 = validate(ds, {"models": ["dbscan"], "params": {"dbscan": {"eps": 1e-4}}})
    r2 = jobs.start(ds, lambda job: run(job, ds, s2), s2, 300)
    jobs.wait(r2)
    db = store.read_json(store.run_dir(ds, r2) / "results.json")["models"]["dbscan"]
    assert db["noise"] == 240 and db["scores"]["silhouette"] is None and "fewer than 2" in db["scores_note"]

    # memory cap formula (spec §7): 1.2 x n(n-1)/2 x 8 bytes must fit half the RAM
    cap = agglomerative_cap(ram=8e9)
    assert AGGLO_FACTOR * cap * (cap - 1) / 2 * 8 <= RAM_SHARE * 8e9 < AGGLO_FACTOR * (cap + 1) * cap / 2 * 8
    assert free_ram() is None or free_ram() > 0
    try:
        validate(ds, {"models": ["kmeans"], "params": {"kmeans": {"n_clusters": 500}}})
        raise AssertionError("k above the spec range")
    except ValueError:
        pass
    print("clustering self-check passed")
