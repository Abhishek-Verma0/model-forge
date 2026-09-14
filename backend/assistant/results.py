"""What the chat AI sees of a training run, and a check on the numbers it answers with.

view(results, question, metrics) -> the part of a run's results.json the question needs:
  always   task, target, main metric (and which way is better), best model, and for every
           model its main score: train / CV mean ± sd / test
  details  every score, epochs, parameters and fit time -- only for the models the question
           names (key or label) or, when none is named, for the best model and the baseline
unverified(answer, sent) -> numbers in the AI's answer that appear nowhere in what it was sent.
"""

import json
import re

SMALL_INT = 10  # our starting point: 0-10 are list numbers / counts ("3 models", "1.") -- not checked


def _name_hits(question, models):
    q = question.lower()
    return [k for k, m in models.items()
            if k.lower() in q or k.replace("_", " ") in q or (m.get("label") or "").lower() in q]


def view(results, question, metrics):
    """`metrics` = {key: (label, higher_is_better)} (training.metrics.METRICS)."""
    if not results or not results.get("models"):
        return None
    primary = results.get("primary_metric")
    models = results["models"]

    def main(m):
        if m.get("error"):
            return {"error": m["error"]}
        cv = (m.get("cv") or {}).get(primary) or {}
        return {"train": (m.get("train") or {}).get(primary), "cv_mean": cv.get("mean"), "cv_sd": cv.get("sd"),
                "test": (m.get("test") or {}).get(primary)}

    named = _name_hits(question or "", models)
    detail_keys = named or [k for k in (results.get("best"), "baseline") if k in models]
    details = {}
    for k in detail_keys:
        m = models[k]
        details[k] = {"label": m.get("label"), "cv_mean": {n: (s or {}).get("mean") for n, s in (m.get("cv") or {}).items()},
                      "test": m.get("test"), "train_cv": m.get("train_cv"), "epochs": m.get("epochs"),
                      "params": m.get("params"), "fit_seconds": m.get("fit_seconds"), "note": m.get("note"),
                      "error": m.get("error")}
    label, higher = metrics.get(primary, (primary, True))
    return {"task": results.get("task"), "target": results.get("target"), "folds": results.get("folds"),
            "positive_class": results.get("positive_class"),
            "main_metric": {"key": primary, "label": label, "higher_is_better": higher},
            "best_by_cv_mean": results.get("best"),
            "all_models_main_metric": {k: {"label": m.get("label"), **main(m)} for k, m in models.items()},
            "details": details}


_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:[eE]-?\d+)?\s*([kKmMbB%])?(?![\w])")
_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9}


def _numbers_in(text):
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(0).rstrip("kKmMbB% ").strip()
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        digits = len(raw.split(".")[1]) if "." in raw and "e" not in raw.lower() else 0
        out.append((m.group(0).strip(), value, digits, (m.group(1) or "").lower()))
    return out


def unverified(answer, sent):
    """Numbers the AI wrote that don't match any number it was given, allowing for its
    rounding (0.9512 -> "0.95"), percentages (0.9512 -> "95.1%") and k / M / B suffixes."""
    given = [v for _, v, _, _ in _numbers_in(json.dumps(sent, default=str))]
    missing = []
    for text, value, digits, suffix in _numbers_in(answer):
        if suffix == "" and value.is_integer() and 0 <= value <= SMALL_INT:
            continue
        tol = 0.5 * 10 ** -digits
        candidates = [value]
        if suffix == "%":
            candidates = [value, value / 100]
        elif suffix in _SCALE:
            candidates = [value * _SCALE[suffix]]
            tol *= _SCALE[suffix]
        ok = any(abs(g - c) <= (tol / 100 if suffix == "%" and c == value / 100 else tol) + 1e-12
                 for g in given for c in candidates)
        if not ok:
            missing.append(text)
    return missing


if __name__ == "__main__":
    metrics = {"f1": ("F1", True), "rmse": ("RMSE", False)}
    res = {"task": "classification", "target": "Category", "primary_metric": "f1", "folds": 5, "best": "lightgbm",
           "models": {
               "baseline": {"label": "Baseline", "cv": {"f1": {"mean": 0.0, "sd": 0.0}}, "test": {"f1": 0.0},
                            "train": {"f1": 0.0}},
               "lightgbm": {"label": "LightGBM", "cv": {"f1": {"mean": 0.9412, "sd": 0.012}, "accuracy": {"mean": 0.98}},
                            "test": {"f1": 0.9512, "accuracy": 0.985}, "train": {"f1": 0.9987},
                            "train_cv": {"f1": {"mean": 0.9991}}, "epochs": {"epochs_trained": 87, "best_epoch": 77}},
               "xgboost": {"label": "XGBoost", "cv": {"f1": {"mean": 0.8745, "sd": 0.02}}, "test": {"f1": 0.8801},
                           "train": {"f1": 0.9123}},
               "svm": {"label": "Support vector machine", "error": "ValueError: boom"}}}

    # every model's main score is always there; details only for what the question names
    v = view(res, "Which model worked best?", metrics)
    assert set(v["all_models_main_metric"]) == set(res["models"]) and v["main_metric"]["label"] == "F1", v
    assert set(v["details"]) == {"lightgbm", "baseline"}, v["details"].keys()   # none named: best + baseline
    assert v["all_models_main_metric"]["lightgbm"] == {"label": "LightGBM", "train": 0.9987, "cv_mean": 0.9412,
                                                       "cv_sd": 0.012, "test": 0.9512}
    assert v["all_models_main_metric"]["svm"] == {"label": "Support vector machine", "error": "ValueError: boom"}
    assert set(view(res, "how well does XGBoost work?", metrics)["details"]) == {"xgboost"}
    assert set(view(res, "did the support vector machine fail?", metrics)["details"]) == {"svm"}
    assert view({}, "x", metrics) is None and view(None, "x", metrics) is None
    # a large results.json (schemas, feature lists) is never passed through
    assert "schema" not in json.dumps(view({**res, "features": ["a"] * 999}, "x", metrics))

    # number check: rounding, percentages, suffixes pass; invented numbers are caught
    sent = {"view": v, "rows": 5573, "price_mean": 4766729.2477}
    assert unverified("LightGBM scored F1 0.9512 on test (CV 0.94 ± 0.012).", sent) == []
    assert unverified("That is 95.1% on the test set, train 99.87%.", sent) == []
    assert unverified("Average price about 4.77M, or 4,766,729 exactly; 5,573 rows.", sent) == []
    assert unverified("It stopped at epoch 77 of 87, 3 models, step 1.", sent) == []
    assert unverified("XGBoost reached 0.87 (its CV 0.8745, rounded).", sent) == []
    assert unverified("XGBoost reached 0.83 and 162 houses have 3 bedrooms.", sent) == ["0.83", "162"]
    assert unverified("F1 of 0.9999 is suspicious", sent) == ["0.9999"]
    print("results view self-check passed")
