"""The automated report (plan §15) of a dataset project: PDF, Excel and JSON.

Contents, in the plan's order: dataset description, data-quality summary,
cleaning + preprocessing operations, exploratory analysis, model methodology and
parameter settings, evaluation metrics and graphs, statistical tests (with the
hypothesis decision table), explainability results, interpretations, limitations
and reproducibility details. Only what was actually computed is reported --
nothing is re-trained to make the report; a section says so when its part of the
workflow was not run.

Interpretations are automated and labelled as such (plan §24); the researcher
reviews them before using the report.
"""

import datetime
import io
import json

import pandas as pd

from core import store
from eda.eda import render_charts
from explain import importance
from preprocessing.detector import run_quality_report
from preprocessing.profiler import profile_dataset
from report import charts as C
from report.layout import Document
from stats import tests
from training import curves, diagnostics
from training.metrics import METRICS

REVIEW = ("Automated interpretation - written by software from the numbers above. Confirm the assumptions and "
          "read it in the context of the study before using it.")
LIMITATIONS = [  # plan §31 + §24
    "Results describe this dataset; they generalise only as far as the data represent the population.",
    "Automated recommendations (cleaning, tests, models) cannot replace domain knowledge or research judgement.",
    "Feature importance and SHAP values show what the model relies on; they do not prove causation.",
    "Model output is not medical, legal or policy advice.",
    "Cross-validation scores of tuned models are slightly optimistic; the locked test score is the honest estimate.",
]


def _f(v, d=3):
    if v is None:
        return "n/a"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def _p(p):
    """p-values the usual way: below 0.0001 is written '< 0.0001'."""
    return "n/a" if p is None else "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def _pv(p):
    """In a sentence: 'p < 0.0001' or 'p = 0.0123'."""
    v = _p(p)
    return f"p {v}" if v.startswith("<") else f"p = {v}"


def _op_text(o):
    """impute(median), outliers(clip_iqr, k=1.5), retype(datetime, dayfirst=True) ..."""
    main = [str(o[k]) for k in ("method", "strategy", "to") if k in o]
    extra = [f"{k}={json.dumps(v, default=str) if isinstance(v, (list, dict)) else v}" for k, v in o.items()
             if k not in ("op", "column", "note", "method", "strategy", "to") and v is not None]
    col = o.get("column")
    return f"{o['op']}({', '.join(main + extra)})" + (f" on {col}" if col else "")


def latest_run(ds_id, kinds=("train", "tune", "cluster")):
    from core import jobs
    for r in jobs.list_runs(ds_id):
        settings = store.read_json(store.run_dir(ds_id, r["run_id"]) / "settings.json") or {}
        if r.get("state") == "done" and settings.get("kind") in kinds:
            return r["run_id"]
    return None


def gather(ds_id, run_id=None):
    """Everything the report shows, read from disk."""
    m = store.meta(ds_id) or {}
    df = store.load_frame(ds_id)
    prep = store.load_prep(ds_id)
    run_id = run_id or latest_run(ds_id)
    res = settings = None
    if run_id:
        res = store.read_json(store.run_dir(ds_id, run_id) / "results.json")
        settings = store.read_json(store.run_dir(ds_id, run_id) / "settings.json")
    return {"meta": m, "df": df, "prep": prep, "run_id": run_id, "results": res, "settings": settings,
            "stats": store.analyses(ds_id, "stats")}


def _best_charts(doc, ds_id, run_id, res):
    best = res.get("best")
    if not best:
        return
    preds = store.read_json(store.run_dir(ds_id, run_id) / "predictions" / f"{best}.json") or {}
    c = curves.curves(res, preds)
    label = res["models"][best]["label"]
    doc.heading(f"Graphs of the best model: {label}", 2)
    if c.get("task") == "classification":
        doc.charts([C.confusion(c), C.curve(c["roc"], "fpr", "tpr", "auc", "ROC curve (test set)",
                                            "false positive rate", "true positive rate", True)] if c["roc"]
                   else [C.confusion(c)])
        if c["pr"]:
            doc.charts([C.curve(c["pr"], "recall", "precision", "ap", "Precision-recall (test set)", "recall", "precision"),
                        C.calibration(c["calibration"])])
    elif c.get("task") == "regression" and c.get("n", 0) >= 2:
        doc.charts([C.actual_vs_predicted(c), C.residuals(c)])
        doc.charts([C.histogram(c), C.qq(c["qq"])])
    lc = diagnostics.cached(ds_id, run_id, f"{best}_learning")
    if lc:
        doc.chart(C.lines(lc["train_rows"], [("training rows", lc["train_mean"], lc["train_sd"]),
                                             ("held-out fold", lc["val_mean"], lc["val_sd"])],
                          f"Learning curve - {lc['metric_label']}", "training rows", lc["metric_label"]), 2.6)


def _supervised(doc, ds_id, run_id, res, settings):
    primary = res["primary_metric"]
    plabel, higher = METRICS[primary]
    task = res["task"]
    doc.heading("5. Model methodology and parameter settings")
    doc.para(f"Task: {task}. Target: {res['target']}. Each model is cross-validated with {res['folds']} "
             f"{'stratified ' if task == 'classification' else ''}folds on the training split (preprocessing refitted "
             f"inside every fold, so no fold sees statistics of the rows it is scored on), refitted on all training "
             f"rows and scored once on the locked test split. Primary metric: {plabel} "
             f"({'higher' if higher else 'lower'} is better). Seed: {res['seed']}."
             + (f" Positive class: {res['positive_class']}." if res.get("positive_class") else ""))
    rows = [[r["label"], "; ".join(f"{k}={v}" for k, v in (r.get("params") or {}).items()) or "defaults",
             r.get("device", "cpu")] for k, r in res["models"].items()]
    doc.table(["Model", "Parameters used", "Device"], rows, [0.3, 0.6, 0.1])
    tuning = res.get("tuning")
    if tuning:
        doc.para(f"Hyperparameter search: {tuning['method']} search on the same folds, optimising {plabel}. "
                 "The winning settings were then trained like the other models.")
        for k, t in tuning["models"].items():
            best = t.get("best") or {}
            doc.para(f"{t['label']}: {len(t['trials'])} trial(s); best CV {plabel} {_f(best.get('cv_mean'))} with "
                     f"{json.dumps(best.get('params', {}), default=str)}", 8.5, indent=0.15)

    doc.heading("6. Evaluation")
    if primary == "accuracy" and (res.get("facts") or {}).get("imbalanced"):
        doc.para("Caution: the classes are imbalanced and models were ranked by accuracy, which can stay high while "
                 "the smaller class is never predicted. Read F1, recall, MCC, balanced accuracy and PR-AUC below "
                 "before choosing a model.", 8.5, color="#b45309")
    metric_keys = [m for m in METRICS if any(m in (r.get("test") or {}) for r in res["models"].values())]
    shown = [primary] + [m for m in metric_keys if m != primary][:6]
    rows = []
    for k, r in res["models"].items():
        if "error" in r:
            rows.append([r["label"], "failed", "", *[""] * len(shown)])
            continue
        cv = (r.get("cv") or {}).get(primary) or {}
        rows.append([r["label"] + (" (best)" if res.get("best") == k else "") + (" [reference]" if r.get("baseline") else ""),
                     f"{_f(cv.get('mean'))} ± {_f(cv.get('sd'))}", _f((r.get("train") or {}).get(primary)),
                     *[_f((r.get("test") or {}).get(m)) for m in shown]])
    doc.table(["Model", f"{plabel} CV mean ± sd", f"{plabel} train", *[f"{METRICS[m][0]} test" for m in shown]],
              rows, [0.26, 0.14, 0.09] + [0.51 / len(shown)] * len(shown), size=6.8)
    scored = [(r["label"], r["cv"][primary]["mean"], r["cv"][primary]["sd"] or 0) for r in res["models"].values()
              if "error" not in r and (r.get("cv") or {}).get(primary, {}).get("mean") is not None]
    scored.sort(key=lambda x: -x[1] if higher else x[1])
    if scored:
        doc.chart(C.bars([s[0] for s in scored], [s[1] for s in scored], f"{plabel} - cross-validation mean ± sd",
                         plabel, [s[2] for s in scored]), 2.4, left=1.9)
    _best_charts(doc, ds_id, run_id, res)


def _clustering(doc, res):
    primary = res["primary_metric"]
    doc.heading("5. Clustering methodology")
    doc.para("Clustering groups the preprocessed rows without a target; every row gets a cluster (DBSCAN may mark "
             "isolated rows as noise). Scores measure compactness and separation only - they cannot show that the "
             "groups are meaningful for the research question.")
    doc.table(["Method", "Parameters", "Clusters", "Noise rows", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
              [[r["label"], "; ".join(f"{k}={v}" for k, v in (r.get("params") or {}).items()), r.get("clusters", "-"),
                r.get("noise", "-"), _f((r.get("scores") or {}).get("silhouette")),
                _f((r.get("scores") or {}).get("davies_bouldin")), _f((r.get("scores") or {}).get("calinski_harabasz"), 1)]
               for r in res["models"].values() if "error" not in r],
              [0.22, 0.24, 0.09, 0.09, 0.12, 0.12, 0.12], size=7)
    if res.get("elbow"):
        e = res["elbow"]
        doc.charts([C.lines(e["k"], [("inertia", e["inertia"], None)], "Elbow (K-Means)", "k", "inertia"),
                    C.lines(e["k"], [("silhouette", e["silhouette"], None)], "Silhouette by k", "k", "silhouette")])
    best = res.get("best")
    if best and res["models"][best].get("scatter"):
        doc.chart(C.clusters(res["models"][best]["scatter"], f"Clusters of {res['models'][best]['label']} in 2-D"), 3.2)
    doc.para(f"Best by {primary.replace('_', ' ')}: {res['models'][best]['label'] if best else 'n/a'}.")


def _stats(doc, saved):
    doc.heading("7. Statistical analysis")
    if not saved:
        doc.para("No statistical tests were run (Statistics tab).")
        return
    for r in reversed(saved):
        a = r.get("analysis")
        if a == "describe":
            doc.heading("Descriptive statistics", 3)
            num = [x for x in r["rows"] if x["kind"] == "numeric"]
            doc.table(["Column", "n", "mean", "sd", "median", "IQR", "skew", "kurtosis", "95% CI of mean"],
                      [[x["column"], x["n"], _f(x.get("mean")), _f(x.get("sd")), _f(x.get("median")), _f(x.get("iqr")),
                        _f(x.get("skewness")), _f(x.get("kurtosis")), f"{_f(x.get('ci_low'))} to {_f(x.get('ci_high'))}"]
                       for x in num], [0.2, 0.06, 0.09, 0.09, 0.09, 0.09, 0.08, 0.1, 0.2], size=7)
        elif a == "frequencies":
            for t in r["tables"]:
                doc.heading(f"Frequencies: {t['column']}", 3)
                doc.table(["Value", "Count", "%", "Valid %", "Cumulative %"],
                          [[x["value"], x["count"], x["percent"], x["valid_percent"], x["cumulative"]] for x in t["rows"]],
                          [0.4, 0.15, 0.15, 0.15, 0.15], max_rows=25)
        elif a == "normality":
            doc.heading("Normality (Shapiro-Wilk)", 3)
            doc.table(["Column", "Group", "n", "W", "p", "skew", "kurtosis"],
                      [[x["column"], x["group"] or "all", x["n"], _f((x["shapiro"] or {}).get("W")),
                        _p((x["shapiro"] or {}).get("p")), _f(x["skewness"]), _f(x["kurtosis"])] for x in r["rows"]],
                      [0.25, 0.15, 0.08, 0.12, 0.12, 0.14, 0.14])
        elif a == "reliability":
            doc.heading(r["title"], 3)
            doc.para(f"Cronbach's alpha = {_f(r['alpha'])} ({r['label']}), standardised {_f(r['standardized_alpha'])}, "
                     f"k = {r['k']} items, n = {r['n']} respondents.")
            doc.table(["Item", "Reversed", "Mean", "SD", "Item-total r", "Alpha if deleted"],
                      [[x["item"], "yes" if x["reversed"] else "", _f(x["mean"]), _f(x["sd"]), _f(x["item_total_r"]),
                        _f(x["alpha_if_deleted"])] for x in r["items"]], [0.3, 0.1, 0.12, 0.12, 0.18, 0.18])
        elif a == "regression":
            doc.heading(r["title"], 3)
            stat = "t" if r["kind"] == "linear" else "z"
            doc.table(["Term", "Estimate", "SE", stat, "p", "95% CI"] + (["Odds ratio"] if r["kind"] == "logistic" else []),
                      [[c["term"], _f(c["estimate"], 4), _f(c["se"], 4), _f(c[stat]), _p(c["p"]),
                        f"{_f(c['low'], 3)} to {_f(c['high'], 3)}"] + ([_f(c.get("odds_ratio"))] if r["kind"] == "logistic" else [])
                       for c in r["coefficients"]], None, size=7)
            doc.para((f"R² = {_f(r['r2'])}, adjusted R² = {_f(r['adj_r2'])}, F({r['f_df'][0]}, {r['f_df'][1]}) = {_f(r['f'], 2)}, "
                      f"{_pv(r['p'])}.") if r["kind"] == "linear" else
                     f"McFadden pseudo R² = {_f(r['pseudo_r2'])}, likelihood-ratio chi² = {_f(r['lr_chi2'], 2)} "
                     f"(df {r['lr_df']}), {_pv(r['p'])}.", 8.5)
        elif a in ("compare", "paired", "association"):
            eff = r.get("effect") or {}
            doc.heading(f"{r['test']}: {r['title']}", 3)
            doc.para(f"{r['statistic_name']} = {_f(r['statistic'])}" + (f", df = {r['df']}" if r.get("df") is not None else "")
                     + f", {_pv(r['p'])}; {eff.get('name', 'effect')} = {_f(eff.get('value'))} ({eff.get('magnitude') or 'n/a'})."
                     + (f" Why this test: {r['reason']}." if r.get("reason") else ""), 8.5)
        elif a == "correlation":
            doc.heading(r["title"], 3)
            doc.table(["Pair", "Method", "r", "p", "n", "Strength"],
                      [[f"{p['a']} - {p['b']}", p.get("method", ""), _f(p.get("r")), _p(p.get("p")), p["n"],
                        p.get("magnitude") or ""] for p in r["pairs"]], [0.35, 0.13, 0.12, 0.14, 0.1, 0.16])
    rows = tests.decisions(saved)
    if rows:
        doc.heading("Hypothesis decision table", 3)
        doc.table(["Hypothesis", "Test", "p", "Effect size", "Decision"],
                  [[x["hypothesis"], x["test"], _p(x["p"]), x["effect"], x["decision"]] for x in rows],
                  [0.36, 0.18, 0.08, 0.2, 0.18], size=6.8)
        doc.para(REVIEW, 7.5, color="#555555")


def _explain(doc, ds_id, run_id, res):
    doc.heading("8. Explainable AI")
    best = res.get("best") if res else None
    ex = importance.cached(ds_id, run_id, best) if best else None
    if not ex:
        doc.para("No explanation was computed for the best model (Train tab -> Explain).")
        return
    p = ex["permutation"]
    doc.para(f"Permutation importance of {ex['label']}: the drop in {p['metric_label']} when one input column is "
             f"shuffled ({p['repeats']} repeats, {ex['rows']} locked-test rows).")
    doc.chart(C.bars([c["column"] for c in p["columns"]], [c["mean"] for c in p["columns"]],
                     f"Permutation importance ({p['metric_label']} drop)", "importance", [c["sd"] for c in p["columns"]]), 2.6,
              left=1.9)
    if ex.get("shap"):
        s = ex["shap"]
        doc.para(f"SHAP ({s['method']}): mean absolute contribution of each column to {s['explained']}, in {s['units']}.")
        doc.chart(C.bars([c["column"] for c in s["columns"]], [c["mean_abs"] for c in s["columns"]],
                         "Mean |SHAP value|", s["units"], color=3), 2.6, left=1.9)
    elif ex.get("shap_note"):
        doc.para(ex["shap_note"], 8.5)
    pdp = [c for c in (ex.get("pdp") or {}).get("columns", []) if not c.get("error")]
    if pdp:
        doc.para(f"Partial dependence: the average prediction ({ex['pdp']['units']}) over {ex['pdp']['rows']} test "
                 "rows when one column is set to each value of a grid, the other columns unchanged.")
        for c in pdp:
            doc.chart(C.lines(c["x"], [(cv["label"], cv["y"], None) for cv in c["curves"]],
                              f"Partial dependence - {c['column']}", c["column"], ex["pdp"]["units"]), 2.2)
    doc.para("These explanations support interpretation; they do not prove causation.", 8, color="#555555")


def pdf(ds_id, run_id=None):
    g = gather(ds_id, run_id)
    m, df, prep, res, settings = g["meta"], g["df"], g["prep"], g["results"], g["settings"]
    if df is None:
        raise LookupError("Dataset not found.")
    doc = Document(m.get("title") or m.get("filename") or "Dataset", "Analysis report - ResearchAI Studio")
    doc.cover([f"Data file: {m.get('filename')}", f"Rows x columns: {df.shape[0]:,} x {df.shape[1]}",
               f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
               f"Training run: {g['run_id'] or 'none yet'}", "",
               "Every number is computed from the data by the software; automated interpretations are labelled "
               "and must be reviewed by the researcher."])
    profile, quality = profile_dataset(df), run_quality_report(df)
    types = {c["column"]: c["detected_type"] for c in quality["checks"]["column_types"]["columns"]}
    doc.heading("1. Dataset description")
    doc.para(f"{df.shape[0]:,} rows and {df.shape[1]} columns; {profile['duplicate_rows']:,} duplicate row(s).")
    doc.para(f"Ethics approval / consent (entered by the researcher):{m.get('ethics') or 'not recorded'}.", 8)
    personal = quality["checks"]["personal_data"]["columns"]
    if personal:
        doc.para("Possible personal data: " + ", ".join(f"{p['column']} ({p['kind']})" for p in personal)
                 + ". Check that its use is covered by consent or remove it.", 8, color="#9a3412")
    info = profile["columns_info"]
    doc.table(["Column", "Type", "Missing %", "Distinct", "Min", "Max", "Mean"],
              [[c, types.get(c), info[c]["missing_percent"], info[c]["unique_values"],
                _f((info[c].get("statistics") or {}).get("min")), _f((info[c].get("statistics") or {}).get("max")),
                _f((info[c].get("statistics") or {}).get("mean"))] for c in profile["column_names"]],
              [0.28, 0.12, 0.1, 0.1, 0.13, 0.13, 0.14], size=7, max_rows=120)
    q = quality["quality_score"]
    doc.heading("2. Data-quality summary")
    doc.para(f"Quality score {q['score']}/100 ({q['grade']})"
             + (": " + ", ".join(f"-{d['points']} {d['reason']}" for d in q["deductions"]) if q["deductions"] else
                ": nothing wrong found") + ".")
    found = [k.replace("_", " ") for k, v in quality["issues_found"].items() if v]
    doc.para("Issues detected: " + (", ".join(found) if found else "none") + ".")

    doc.heading("3. Cleaning and preprocessing")
    ops = m.get("clean_ops") or []
    doc.para(f"Cleaning operations ({len(ops)}), applied in order to the whole table before the split:"
             if ops else "No cleaning operations were applied.")
    doc.bullets([_op_text(o) for o in ops][:60])
    if prep:
        s = prep["fitted"]["settings"]
        doc.para(f"Task: {s['task']}. " + ("Split: " + f"{round((1 - s['test_size']) * 100)}:{round(s['test_size'] * 100)} "
                 f"train/test, seed {s['random_state']}, {'stratified' if s.get('stratify') and s['task'] == 'classification' else 'random'}"
                 + (f", grouped by {s['group']}" if s.get("group") else "") + "." if s["task"] != "clustering" else
                 "No split (clustering uses every row)."))
        doc.para("Per-column operations (fitted on the training split only):")
        doc.bullets([f"{c}: " + ", ".join(_op_text(o).replace(f" on {c}", "") for o in col_ops)
                     for c, col_ops in s["columns"].items() if col_ops][:80])
        if s.get("pipeline"):
            doc.para("Dataset-level steps: " + "; ".join(f"{k} = {json.dumps(v)}" for k, v in s["pipeline"].items()))
    else:
        doc.para("Preprocessing has not been run yet.")

    doc.heading("4. Exploratory analysis")
    num = [c for c in profile["column_names"] if info[c].get("statistics")]
    if num:
        doc.table(["Column", "Mean", "SD", "Median", "Q1", "Q3", "Skew", "Kurtosis"],
                  [[c, *[_f(info[c]["statistics"].get(k)) for k in ("mean", "std", "median", "q1", "q3", "skew", "kurtosis")]]
                   for c in num], [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], size=7, max_rows=60)
    for chart in render_charts(df)[:4]:
        doc.image(chart["image"], 2.6, chart["title"])

    if res and res.get("task") == "clustering":
        _clustering(doc, res)
    elif res and res.get("models"):
        _supervised(doc, ds_id, g["run_id"], res, settings)
    else:
        doc.heading("5. Models")
        doc.para("No training run has finished yet.")
    _stats(doc, g["stats"])
    _explain(doc, ds_id, g["run_id"], res if res and res.get("task") != "clustering" else None)

    doc.heading("9. Interpretation")
    lines = []
    if res and res.get("task") in ("classification", "regression") and res.get("best"):
        b = res["models"][res["best"]]
        pm, (plabel, _) = res["primary_metric"], METRICS[res["primary_metric"]]
        base = res["models"].get("baseline", {})
        lines.append(f"The model with the best cross-validated {plabel} is {b['label']} "
                     f"(CV {_f(b['cv'][pm]['mean'])} ± {_f(b['cv'][pm]['sd'])}; locked test {_f(b['test'].get(pm))}; "
                     f"on its own training rows {_f((b.get('train') or {}).get(pm))}).")
        bt, mt = (base.get("test") or {}).get(pm), b["test"].get(pm)
        if bt is not None and mt is not None:
            better = mt > bt if METRICS[pm][1] else mt < bt
            lines.append(f"The reference baseline (always the most common class / the training mean) scores "
                         f"{_f(bt)} on the test set, so {b['label']} {'beats' if better else 'does NOT beat'} it.")
    for r in g["stats"]:
        if r.get("decision") and r.get("p") is not None and r.get("test"):
            lines.append(f"{r['test']} ({r.get('title')}): {r['decision']}.")
    doc.bullets(lines or ["Nothing to interpret yet."])
    doc.para(REVIEW, 7.5, color="#555555")

    doc.heading("10. Limitations")
    doc.bullets(LIMITATIONS)
    doc.heading("11. Reproducibility")
    if res:
        doc.para(f"Library versions: {', '.join(f'{k} {v}' for k, v in (res.get('versions') or {}).items())}.", 8)
        doc.para(f"Seed {res.get('seed')}; folds {res.get('folds', '-')}; data fingerprint {res.get('fingerprint')} "
                 "(same cleaning, preprocessing, target, split and folds give the same fingerprint and the same results).", 8)
    doc.para("The experiment configuration and results are available as JSON from the Report page.", 8)
    return doc.bytes()


def json_export(ds_id, run_id=None):
    """§15 JSON: experiment configuration and results."""
    g = gather(ds_id, run_id)
    prep = g["prep"]
    return {"dataset": {k: g["meta"].get(k) for k in ("title", "filename", "created", "shape", "ethics")},
            "clean_ops": g["meta"].get("clean_ops"),
            "preprocessing": prep["fitted"]["settings"] if prep else None,
            "run": {"id": g["run_id"], "settings": g["settings"], "results": g["results"]},
            "statistics": g["stats"], "generated": datetime.datetime.now().isoformat(timespec="seconds")}


def excel(ds_id, run_id=None):
    """§15 Excel: tables, metrics and the data dictionary in one workbook."""
    g = gather(ds_id, run_id)
    df, res = g["df"], g["results"]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        prof = profile_dataset(df)
        pd.DataFrame([{"column": c, "pandas type": v["data_type"], "missing": v["missing_values"],
                       "missing %": v["missing_percent"], "distinct": v["unique_values"],
                       **(v.get("statistics") or {})} for c, v in prof["columns_info"].items()]
                     ).to_excel(xw, sheet_name="Data dictionary", index=False)
        if res and res.get("models") and res.get("task") != "clustering":
            rows = []
            for k, r in res["models"].items():
                row = {"model": r["label"], "best": res.get("best") == k, "error": r.get("error"),
                       "fit seconds": r.get("fit_seconds")}
                for mk, v in (r.get("cv") or {}).items():
                    row[f"{mk} CV mean"], row[f"{mk} CV sd"] = v.get("mean"), v.get("sd")
                for mk, v in (r.get("test") or {}).items():
                    row[f"{mk} test"] = v
                rows.append(row)
            pd.DataFrame(rows).to_excel(xw, sheet_name="Model metrics", index=False)
            pd.DataFrame([{"model": r["label"], **{k: json.dumps(v) if isinstance(v, list) else v
                                                    for k, v in (r.get("params") or {}).items()}}
                          for r in res["models"].values()]).to_excel(xw, sheet_name="Parameters", index=False)
        elif res and res.get("task") == "clustering":
            pd.DataFrame([{"method": r["label"], "clusters": r.get("clusters"), "noise": r.get("noise"),
                           **(r.get("scores") or {})} for r in res["models"].values()]
                         ).to_excel(xw, sheet_name="Clustering", index=False)
        if g["stats"]:
            pd.DataFrame(tests.decisions(g["stats"])).to_excel(xw, sheet_name="Hypothesis decisions", index=False)
            for r in g["stats"]:
                if r.get("analysis") == "describe":
                    pd.DataFrame(r["rows"]).to_excel(xw, sheet_name="Descriptives", index=False)
                    break
    return buf.getvalue()
