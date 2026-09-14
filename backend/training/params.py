"""Editable training parameters. A model row lists specs; defaults are read from the
library (never typed in here), user values are validated against each spec's range
before a run starts. Ranges are "our starting point" -- the library accepts wider.

kinds: int, float, choice, bool, layers (hidden layer sizes, e.g. "64,32").
flags: none=True allows "no limit" (None); auto=True allows "auto" (library decides).
fit=True marks options our epoch loop uses (early stopping) -- not constructor args.
"""

import math

import numpy as np

AUTO = "auto"
EARLY_STOPPING_DEFAULT = False  # published default (scikit-learn MLP early_stopping=False)
PATIENCE_DEFAULT = 10           # published default (scikit-learn MLP / HistGB n_iter_no_change=10)
MAX_LAYERS = 10                 # our starting point
MAX_LAYER_UNITS = 4096          # our starting point


def spec(name, label, kind, lo=None, hi=None, *, choices=None, none=False, auto=False, log=False, help=""):
    return {"name": name, "label": label, "kind": kind, "min": lo, "max": hi, "choices": choices,
            "none": none, "auto": auto, "log": log, "help": help, "fit": False}


def fit_specs():
    """Early-stopping options shared by every epoch-based model."""
    return [
        {**spec("early_stopping", "Early stopping", "bool",
                help="stop when validation loss stops improving (needs validation share > 0)"), "fit": True},
        {**spec("patience", "Patience (epochs)", "int", 1, 1000,
                help="epochs without improvement before stopping"), "fit": True},
    ]


def _plain(v):
    """JSON-safe copy of a library default."""
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def defaults(row):
    """Library defaults for this row's parameters (row['resolve'] fills values the
    library only reveals after a fit, e.g. XGBoost / CatBoost)."""
    got = row["make"](0).get_params()
    out = {}
    for p in row["params"]:
        if p["fit"]:
            out[p["name"]] = EARLY_STOPPING_DEFAULT if p["name"] == "early_stopping" else PATIENCE_DEFAULT
        else:
            out[p["name"]] = _plain(got.get(p["name"]))
    if row.get("resolve"):
        for k, v in row["resolve"]().items():
            if out.get(k) is None:
                out[k] = v
    return out


def _check(label, p, v):
    name = f"{label} · {p['label']}"
    if p["none"] and v in (None, "none", ""):
        return None
    if p["auto"] and v == AUTO:
        return AUTO
    kind = p["kind"]
    try:
        if kind == "bool":
            if not isinstance(v, bool):
                raise ValueError
            return v
        if kind == "choice":
            if v not in p["choices"]:
                raise ValueError
            return v
        if kind == "layers":
            parts = v if isinstance(v, (list, tuple)) else [x for x in str(v).replace(" ", "").split(",") if x]
            sizes = [int(x) for x in parts]
            if not 1 <= len(sizes) <= MAX_LAYERS or any(not 1 <= s <= MAX_LAYER_UNITS for s in sizes) \
                    or any(float(x) != int(x) for x in parts):
                raise ValueError
            return sizes
        num = float(v)
        if not math.isfinite(num):
            raise ValueError
        if kind == "int":
            if num != int(num):
                raise ValueError
            num = int(num)
        if (p["min"] is not None and num < p["min"]) or (p["max"] is not None and num > p["max"]):
            raise ValueError
        return num
    except (TypeError, ValueError):
        allowed = (f"one of {p['choices']}" if kind == "choice" else
                   f"{MAX_LAYERS} or fewer layer sizes between 1 and {MAX_LAYER_UNITS}, e.g. 64,32" if kind == "layers"
                   else "true or false" if kind == "bool" else f"a {kind} between {p['min']} and {p['max']}")
        extra = (" or none" if p["none"] else "") + (" or auto" if p["auto"] else "")
        raise ValueError(f"{name} must be {allowed}{extra} (got {v!r}).") from None


def resolve(row, user):
    """Validated values -> (constructor kwargs, fit options, record). 'auto' values
    are left to the library (not passed); the record keeps every value used."""
    user = user or {}
    unknown = sorted(set(user) - {p["name"] for p in row["params"]})
    if unknown:
        raise ValueError(f"{row['label']}: unknown parameter(s) {', '.join(unknown)}.")
    base = defaults(row)
    kwargs, fit, record = {}, {}, {}
    for p in row["params"]:
        v = _check(row["label"], p, user[p["name"]]) if p["name"] in user else base[p["name"]]
        record[p["name"]] = v
        if p["fit"]:
            fit[p["name"]] = v
        elif v != AUTO:
            kwargs[p["name"]] = tuple(v) if p["kind"] == "layers" else v
    return kwargs, fit, record


if __name__ == "__main__":
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    mlp = {"label": "MLP", "make": lambda seed: MLPClassifier(random_state=seed),
           "params": [spec("hidden_layer_sizes", "Hidden layers", "layers"),
                      spec("max_iter", "Epochs", "int", 1, 10000),
                      spec("learning_rate_init", "Learning rate", "float", 1e-6, 1.0, log=True)] + fit_specs()}
    d = defaults(mlp)
    assert d == {"hidden_layer_sizes": [100], "max_iter": 200, "learning_rate_init": 0.001,
                 "early_stopping": False, "patience": 10}, d
    kw, fit, rec = resolve(mlp, {"hidden_layer_sizes": "64, 32", "max_iter": 50, "early_stopping": True})
    assert kw == {"hidden_layer_sizes": (64, 32), "max_iter": 50, "learning_rate_init": 0.001}, kw
    assert fit == {"early_stopping": True, "patience": 10} and rec["hidden_layer_sizes"] == [64, 32]
    MLPClassifier(**kw)  # the kwargs are valid constructor arguments
    for bad, msg in (({"max_iter": 0}, "between 1 and 10000"), ({"max_iter": 2.5}, "int"),
                     ({"hidden_layer_sizes": "64,x"}, "layer sizes"), ({"early_stopping": "yes"}, "true or false"),
                     ({"learning_rate_init": float("nan")}, "float"), ({"nope": 1}, "unknown parameter")):
        try:
            resolve(mlp, bad)
            raise AssertionError(bad)
        except ValueError as e:
            assert msg in str(e), (bad, str(e))
    tree = {"label": "Tree", "make": lambda seed: DecisionTreeClassifier(random_state=seed),
            "params": [spec("max_depth", "Max depth", "int", 1, 200, none=True)]}
    assert resolve(tree, {})[0] == {"max_depth": None} and resolve(tree, {"max_depth": "none"})[0] == {"max_depth": None}
    print("params self-check passed")
