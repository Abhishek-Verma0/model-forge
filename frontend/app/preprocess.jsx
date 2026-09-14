"use client";

import { useEffect, useState } from "react";

import { API } from "./apiclient";

// Each control maps to ONE separable operation. Scaling defaults to "none" --
// median-impute must not silently scale a column (that was the 0.5/0.6 bug).
//
// Every choice list and default (`opts`) comes from the backend:
// data.preprocess_options, built by preprocessing/execute.py options(). This file
// only decides how they are shown -- no second copy of any number to drift.

// How each outlier method's parameters are shown (label, input step, bounds). These
// change the DATA, so they ride on the op and land in the run summary; their
// defaults come from the backend (opts.outlier_defaults).
const OUTLIER_PARAMS = {
  clip_iqr:    [{ key: "k", label: "IQR k", step: 0.1, min: 0.1 }],
  remove_rows: [{ key: "k", label: "IQR k", step: 0.1, min: 0.1 }],
  zscore:      [{ key: "threshold", label: "sd", step: 0.5, min: 0.1 }],
  winsorize:   [{ key: "lower", label: "low q", step: 0.01, min: 0, max: 0.49 },
                { key: "upper", label: "high q", step: 0.01, min: 0.51, max: 1 }],
};
// `method` can be "none" (opCode reads the IQR k before checking the method), so a
// method without defaults yields undefined -- callers only use the value for its own method.
const pval = (opts, step, method, p) => step.params?.[method]?.[p.key] ?? opts.outlier_defaults[method]?.[p.key];
const NOT_IMPUTE = ["none", "drop_rows", "drop_column"]; // "missing" choices that are not an impute strategy
const SPLIT_OPTS = [
  { v: 0.3, label: "70 : 30" },
  { v: 0.25, label: "75 : 25" },
  { v: 0.2, label: "80 : 20" },
  { v: 0.1, label: "90 : 10" },
];

function meta(data, target) {
  const { profile, report } = data;
  const info = profile.columns_info;
  const semType = {};
  for (const c of report.checks.column_types.columns) semType[c.column] = c.detected_type;
  const missPct = {};
  for (const m of report.checks.missing_values.columns) missPct[m.column] = m.missing_percent;
  const feats = profile.column_names.filter((c) => c !== target);
  const idCols = new Set(report.checks.id_columns.columns.map((c) => c.column));
  return { info, semType, missPct, feats, idCols };
}

// Rule-based defaults (instant, offline). Impute missing + encode categoricals.
// NO scaling by default -- the user adds it only when a model needs it.
export function defaultSteps(data, target) {
  const { info, semType, missPct, feats, idCols } = meta(data, target);
  const steps = {};
  for (const col of feats) {
    const numeric = semType[col] === "numeric";
    const text = semType[col] === "text";
    const miss = missPct[col] || 0;
    const step = { missing: "none", scale: "none", outliers: "none", encode: "none" };
    if (miss > 60) step.missing = "drop_column"; // mostly empty -> no signal
    else {
      // text blanks stay blank -- the text encoder handles them; a filled-in review is fake
      if (miss > 0 && !text) step.missing = numeric ? "median" : "most_frequent";
      if (!numeric) step.encode = idCols.has(col) ? "drop" : text ? "text"
        : info[col].unique_values > data.preprocess_options.max_onehot ? "drop" : "onehot";
    }
    steps[col] = step;
  }
  return steps;
}

// One column's controls -> the ordered op list the backend executes.
function toOps(opts, step, numeric) {
  if (step.missing === "drop_column" || step.encode === "drop") return [{ op: "drop_column" }];
  const ops = [];
  if (!NOT_IMPUTE.includes(step.missing)) ops.push({ op: "impute", strategy: step.missing });
  else if (step.missing === "drop_rows") ops.push({ op: "drop_rows_missing" });
  if (numeric) {
    if (step.outliers !== "none") {
      const op = { op: "outliers", method: step.outliers };
      for (const p of OUTLIER_PARAMS[step.outliers] || []) op[p.key] = Number(pval(opts, step, step.outliers, p));
      ops.push(op);
    }
    if (step.scale !== "none") ops.push({ op: "scale", method: step.scale });
  } else if (step.encode === "onehot" || step.encode === "ordinal") {
    ops.push({ op: "encode", method: step.encode });
  } else if (step.encode === "text") {
    ops.push({ op: "encode", method: "text" });
  }
  return ops;
}

// AI's op list -> our control state, so an AI plan can populate the cards.
function stepFromOps(ops, numeric) {
  const step = { missing: "none", scale: "none", outliers: "none", encode: "none" };
  for (const o of ops) {
    if (o.op === "drop_column") { numeric ? (step.missing = "drop_column") : (step.encode = "drop"); }
    else if (o.op === "impute") step.missing = o.strategy || "median";
    else if (o.op === "drop_rows_missing") step.missing = "drop_rows";
    else if (o.op === "scale") step.scale = o.method || "standard";
    else if (o.op === "outliers") {
      step.outliers = o.method || "clip_iqr";
      for (const p of OUTLIER_PARAMS[step.outliers] || []) {
        if (o[p.key] != null) step.params = { ...step.params, [step.outliers]: { ...step.params?.[step.outliers], [p.key]: o[p.key] } };
      }
    }
    else if (o.op === "encode") step.encode = o.method || "onehot";
  }
  return step;
}

// The actual pandas each op runs as -- shown so the user sees exactly what
// happens before hitting Run (requirement: "proposed operations/code").
function opCode(opts, step, numeric, col) {
  if (step.missing === "drop_column" || step.encode === "drop") return [`df = df.drop(columns=["${col}"])`];
  const c = `df["${col}"]`;
  const lines = [];
  const fills = {
    median: `${c}.median()`, mean: `${c}.mean()`,
    most_frequent: `${c}.mode()[0]`, constant: `"Missing"`,
  };
  if (fills[step.missing]) lines.push(`${c} = ${c}.fillna(${fills[step.missing]})  # train-fitted`);
  else if (step.missing === "drop_rows") lines.push(`df = df.dropna(subset=["${col}"])`);
  if (numeric) {
    const k = pval(opts, step, step.outliers, OUTLIER_PARAMS.clip_iqr[0]);
    if (step.outliers === "clip_iqr") lines.push(`${c} = ${c}.clip(lower, upper)  # Q1/Q3 ± ${k}·IQR from train`);
    if (step.outliers === "remove_rows") lines.push(`train = train[within_iqr("${col}", k=${k})]  # applied during training, inside each fold`);
    if (step.outliers === "zscore") lines.push(`${c} = ${c}.clip(mean ± ${pval(opts, step, "zscore", OUTLIER_PARAMS.zscore[0])}·std)  # train stats`);
    if (step.outliers === "winsorize") lines.push(`${c} = ${c}.clip(*train.quantile([${pval(opts, step, "winsorize", OUTLIER_PARAMS.winsorize[0])}, ${pval(opts, step, "winsorize", OUTLIER_PARAMS.winsorize[1])}]))`);
    const sc = { standard: "StandardScaler", robust: "RobustScaler", minmax: "MinMaxScaler" }[step.scale];
    if (sc) lines.push(`${c} = ${sc}().fit(train[["${col}"]]).transform(...)`);
  } else {
    if (step.encode === "onehot") lines.push(`pd.get_dummies(${c}, prefix="${col}")  # fit categories on train`);
    if (step.encode === "ordinal") lines.push(`${c} = ${c}.map(train_category_codes)`);
    if (step.encode === "text") lines.push(
      `# ${col} stays as text in the CSV`,
      `TfidfVectorizer(words 1-2).fit(train["${col}"]) + TfidfVectorizer(letters 3-5).fit(...)  # train only, saved in pipeline`);
  }
  return lines.length ? lines : [`# ${col}: kept as-is`];
}

function ruleNote(opts, step, numeric) {
  if (step.missing === "drop_column") return "Mostly empty or free-text/ID — dropped.";
  if (step.encode === "drop") return "Too many unique values to encode — dropped.";
  const bits = [];
  const f = { median: "gaps filled with the median", mean: "gaps filled with the average",
    most_frequent: "gaps filled with the most common value", constant: "gaps marked “Missing”",
    drop_rows: "rows with gaps removed" }[step.missing];
  if (f) bits.push(f);
  if (numeric) {
    const out = { clip_iqr: `extreme values capped to ${pval(opts, step, "clip_iqr", OUTLIER_PARAMS.clip_iqr[0])}×IQR bounds`,
      remove_rows: "extreme training rows removed during training (inside each fold, not in the export)",
      zscore: `values beyond ${pval(opts, step, "zscore", OUTLIER_PARAMS.zscore[0])} standard deviations capped`,
      winsorize: `capped to the ${pval(opts, step, "winsorize", OUTLIER_PARAMS.winsorize[0])}–${pval(opts, step, "winsorize", OUTLIER_PARAMS.winsorize[1])} quantile range` }[step.outliers];
    if (out) bits.push(out);
    const s = { robust: "scaled by median/IQR (outlier-safe)", standard: "standardized", minmax: "squeezed to 0–1" }[step.scale];
    if (s) bits.push(s);
  } else {
    const e = { onehot: "one column per category", ordinal: "mapped to ordered numbers",
      text: "kept as text; its word and letter patterns are learned on train and saved in the pipeline for training" }[step.encode];
    if (e) bits.push(e);
  }
  if (!bits.length) return "Kept as-is — already model-ready.";
  const s = bits.join(", ");
  return s.charAt(0).toUpperCase() + s.slice(1) + ".";
}

function actionText(opts, step, numeric) {
  const ops = toOps(opts, step, numeric);
  if (!ops.length) return "keep as-is";
  return ops.map((o) => o.op === "impute" ? `impute(${o.strategy})`
    : o.op === "scale" ? `scale(${o.method})`
    : o.op === "encode" ? `encode(${o.method})`
    : o.op === "outliers" ? `outliers(${o.method})`
    : o.op === "drop_rows_missing" ? "drop rows w/ missing" : "drop").join(", ");
}

// Plan summary for the AI chat context (aggregated metadata only, never raw rows).
function planSummary(opts, steps, semType, missPct, target, task) {
  const columns = Object.entries(steps).map(([name, step]) => ({
    name, type: semType[name], missing_percent: missPct[name] || 0,
    action: actionText(opts, step, semType[name] === "numeric"),
  }));
  return { target, task, columns };
}

export function recommendedPlan(data, target, task) {
  const { semType, missPct } = meta(data, target);
  return planSummary(data.preprocess_options, defaultSteps(data, target), semType, missPct, target, task);
}

const sameStep = (a, b) =>
  a.missing === b.missing && a.scale === b.scale && a.outliers === b.outliers && a.encode === b.encode &&
  JSON.stringify(a.params || {}) === JSON.stringify(b.params || {});

// AI's dataset-level pipeline -> our flat control state (missing sections = "none").
function pipelineFromAI(opts, p) {
  const d = opts.pipeline_defaults;
  return {
    imputation: p.imputation?.method || "none",
    knn_n: p.imputation?.n_neighbors || d.knn_n,
    outlier_removal: p.outlier_removal?.method || "none",
    contamination: p.outlier_removal?.contamination || d.contamination,
    feature_selection: p.feature_selection?.method || "none",
    fs_k: p.feature_selection?.k || d.fs_k,
    imbalance: p.imbalance?.method || "none",
    reduction: p.reduction?.method || "none",
    red_n: p.reduction?.n || d.red_n,
  };
}

export default function Preprocess({ data, target, task, plan, aiPipeline, planLoading, planErr, onApplyAI, onPlan, onDone }) {
  const { info, semType, missPct, feats } = meta(data, target);
  const opts = data.preprocess_options;
  const [steps, setSteps] = useState(() => defaultSteps(data, target));
  // Train/test split -- plan §10 wants these user-chosen, and a recorded seed is
  // what makes a run reproducible. Defaults come from the backend.
  const [split, setSplit] = useState({ test_size: opts.split.test_size, random_state: opts.split.random_state,
                                       stratify: opts.split.stratify });
  const setSp = (k, v) => setSplit((s) => ({ ...s, [k]: v }));

  const [pipeline, setPipeline] = useState({
    imputation: "none", outlier_removal: "none", feature_selection: "none", imbalance: "none", reduction: "none",
    ...opts.pipeline_defaults,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Lift a summary of the live plan up so the AI chat remembers the user's
  // actual decisions (not just the rule defaults).
  useEffect(() => {
    onPlan?.(planSummary(opts, steps, semType, missPct, target, task));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [steps, target]);

  if (!data.id) return null;
  if (!target)
    return (
      <section>
        <p className="note">Choose a target column above to configure preprocessing.</p>
      </section>
    );

  const aiSteps = {}; // AI's recommendation mapped to control state
  const aiNotes = {};
  if (plan) {
    for (const [col, ops] of Object.entries(plan)) {
      if (!feats.includes(col)) continue;
      aiSteps[col] = stepFromOps(ops, semType[col] === "numeric");
      aiNotes[col] = ops.map((o) => o.note).filter(Boolean).join(" ");
    }
  }
  const baseline = (col) => aiSteps[col] || defaultSteps(data, target)[col];
  const changedCount = feats.filter((c) => !sameStep(steps[c], baseline(c))).length;

  function setParam(col, method, key, value) {
    setSteps((s) => ({ ...s, [col]: { ...s[col],
      params: { ...s[col].params, [method]: { ...s[col].params?.[method], [key]: value } } } }));
  }

  function setStep(col, field, value) {
    setSteps((s) => ({ ...s, [col]: { ...s[col], [field]: value } }));
  }
  const hasAiPipeline = aiPipeline && Object.keys(aiPipeline).length > 0;
  function applyAI() {
    if (!plan) return;
    setSteps((s) => {
      const next = { ...s };
      for (const col of feats) if (aiSteps[col]) next[col] = aiSteps[col];
      return next;
    });
    if (hasAiPipeline) setPipeline(pipelineFromAI(opts, aiPipeline)); // also adopt the AI's dataset-level steps
    onApplyAI?.();
  }

  function setPipe(field, value) {
    setPipeline((p) => ({ ...p, [field]: value }));
  }
  // dataset-level control state -> the backend `pipeline` object (omit "none").
  function buildPipeline() {
    const p = {};
    if (pipeline.imputation !== "none") {
      p.imputation = { method: pipeline.imputation };
      if (pipeline.imputation === "knn") p.imputation.n_neighbors = Number(pipeline.knn_n) || 5;
    }
    if (pipeline.outlier_removal !== "none")
      p.outlier_removal = { method: pipeline.outlier_removal,
                            contamination: Number(pipeline.contamination) || 0.05 };
    if (pipeline.feature_selection !== "none")
      p.feature_selection = { method: pipeline.feature_selection, k: Number(pipeline.fs_k) || 10 };
    if (pipeline.imbalance !== "none") p.imbalance = { method: pipeline.imbalance };
    if (pipeline.reduction !== "none") p.reduction = { method: pipeline.reduction, n: Number(pipeline.red_n) || 2 };
    return p;
  }

  async function apply() {
    setLoading(true); setError(""); setResult(null);
    const columns = {};
    for (const col of feats) columns[col] = toOps(opts, steps[col], semType[col] === "numeric");
    try {
      const res = await fetch(`${API}/api/preprocess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: data.id, target, task, columns, pipeline: buildPipeline(),
          test_size: Number(split.test_size), random_state: Number(split.random_state),
          stratify: split.stratify,
        }),
      });
      if (!res.ok) {
        let d = `Error ${res.status}`;
        try { const j = await res.json(); if (j.detail) d = j.detail; } catch {}
        throw new Error(d);
      }
      setResult(await res.json());
      onDone?.();
    } catch (e) {
      setError(e instanceof TypeError ? `Cannot reach the backend at ${API}.` : e.message);
    } finally { setLoading(false); }
  }

  const detected = data.report.total_issue_types_found;

  return (
    <section>
      <p className="sectitle">Preprocess — target: {target} · {task}</p>

      {/* Steps 1–2: what was detected + what the AI recommends */}
      <div className="pp-steps">
        <div className="pp-step">
          <span className="pp-step-n">1</span>
          <div><b>Detected</b><p>{detected} issue type(s) found in the audit — see the Insights tab.</p></div>
        </div>
        <div className="pp-step">
          <span className="pp-step-n">2</span>
          <div>
            <b>AI recommends</b>
            {planLoading && <p>Asking the assistant…</p>}
            {planErr && <p className="pp-muted">AI offline — using safe rule-based defaults. {planErr}</p>}
            {plan && !planLoading && (
              <p>
                Suggested a plan for {Object.keys(aiSteps).length} column(s)
                {hasAiPipeline ? " + dataset-level steps" : ""}.{" "}
                <button className="pp-link" onClick={applyAI}>Apply AI plan</button>
              </p>
            )}
            {!plan && !planLoading && !planErr && <p className="pp-muted">No AI plan loaded.</p>}
          </div>
        </div>
      </div>

      {/* Train/test split -- ratio, seed, stratify */}
      <div className="panel" style={{ padding: 16, marginBottom: 12 }}>
        <p style={{ margin: "0 0 4px", fontWeight: 600, fontSize: 14 }}>Train / test split</p>
        <p className="note" style={{ margin: "0 0 10px" }}>
          Same seed + same settings reproduce the same split.
        </p>
        <div className="builder-row">
          <label className="pp-field"><span>Split</span>
            <select value={split.test_size} onChange={(e) => setSp("test_size", e.target.value)}>
              {SPLIT_OPTS.filter((x) => x.v >= opts.split.test_size_min && x.v <= opts.split.test_size_max).map((o) => <option key={o.v} value={o.v}>{o.label}</option>)}
            </select>
          </label>
          <label className="pp-field"><span>Random seed</span>
            <input type="number" min="0" value={split.random_state}
                   onChange={(e) => setSp("random_state", e.target.value)} />
          </label>
          <label className="pp-field"><span>Stratified</span>
            <input type="checkbox" checked={split.stratify}
                   disabled={task !== "classification"}
                   onChange={(e) => setSp("stratify", e.target.checked)} />
          </label>
        </div>
      </div>

      {/* Dataset-level steps (optional) -- act on the whole train matrix */}
      <div className="panel" style={{ padding: 16, marginBottom: 12 }}>
        <p style={{ margin: "0 0 4px", fontWeight: 600, fontSize: 14 }}>Dataset-level steps (optional)</p>
        <p className="note" style={{ margin: "0 0 10px" }}>
          Run after the per-column ops, fit on the training split only. Leave “none” to skip.
        </p>
        <div className="builder-row">
          <label className="pp-field"><span>Advanced imputation</span>
            <select value={pipeline.imputation} onChange={(e) => setPipe("imputation", e.target.value)}>
              {opts.pipeline.imputation.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </label>
          {pipeline.imputation === "knn" && (
            <label className="pp-field" style={{ maxWidth: 92 }}><span>neighbours</span>
              <input type="number" min="1" step="1" value={pipeline.knn_n}
                     onChange={(e) => setPipe("knn_n", e.target.value)} />
            </label>
          )}
          <label className="pp-field"><span>Outlier removal (applied during training)</span>
            <select value={pipeline.outlier_removal} onChange={(e) => setPipe("outlier_removal", e.target.value)}>
              {opts.pipeline.outlier_removal.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </label>
          {pipeline.outlier_removal !== "none" && (
            <label className="pp-field" style={{ maxWidth: 110 }}><span>contamination</span>
              <input type="number" min="0.01" max="0.49" step="0.01" value={pipeline.contamination}
                     onChange={(e) => setPipe("contamination", e.target.value)} />
            </label>
          )}
          <label className="pp-field"><span>Feature selection</span>
            <select value={pipeline.feature_selection} onChange={(e) => setPipe("feature_selection", e.target.value)}>
              {opts.pipeline.feature_selection.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </label>
          {pipeline.feature_selection !== "none" && (
            <label className="pp-field"><span>Keep top-k</span>
              <input type="number" min="1" value={pipeline.fs_k}
                     onChange={(e) => setPipe("fs_k", e.target.value)} style={{ width: 70 }} />
            </label>
          )}
          <label className="pp-field"><span>Imbalance{task !== "classification" ? " (clf only)" : " (applied during training)"}</span>
            <select value={pipeline.imbalance} disabled={task !== "classification"}
                    onChange={(e) => setPipe("imbalance", e.target.value)}>
              {opts.pipeline.imbalance.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </label>
          <label className="pp-field"><span>Reduction</span>
            <select value={pipeline.reduction} onChange={(e) => setPipe("reduction", e.target.value)}>
              {opts.pipeline.reduction.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </label>
          {pipeline.reduction !== "none" && (
            <label className="pp-field"><span>Components</span>
              <input type="number" min="1" value={pipeline.red_n}
                     onChange={(e) => setPipe("red_n", e.target.value)} style={{ width: 70 }} />
            </label>
          )}
        </div>
      </div>

      {/* Step 5: Run */}
      <div className="panel" style={{ padding: 16 }}>
        <p style={{ margin: "0 0 8px", fontSize: 14 }}>
          Ready when you are. {changedCount > 0 && <b>{changedCount} column(s) changed from the recommendation.</b>}
        </p>
        <button className="primary" onClick={apply} disabled={loading}>
          {loading ? "Running…" : "Run preprocessing"}
        </button>
      </div>

      {/* Step 3–4: proposed ops per column (the code) + your changes */}
      <p className="pp-cards-title">Per-column operations · change any you like</p>
      <div className="pp-cards">
        {feats.map((col) => {
          const numeric = semType[col] === "numeric";
          const step = steps[col];
          const hasMissing = (missPct[col] || 0) > 0;
          const dropped = step.missing === "drop_column" || step.encode === "drop";
          const changed = !sameStep(step, baseline(col));
          return (
            <div className={"pp-card" + (dropped ? " dropped" : "")} key={col}>
              <div className="pp-card-head">
                <span className="pp-card-name" title={col}>{col}</span>
                <span className="tag">{semType[col]}</span>
                {changed && <span className="pp-changed">changed</span>}
              </div>

              {hasMissing && (
                <label className="pp-field">
                  <span>Missing ({missPct[col]}%)</span>
                  <select value={step.missing} onChange={(e) => setStep(col, "missing", e.target.value)}>
                    {opts.missing.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                </label>
              )}

              {numeric ? (
                <>
                  <label className="pp-field">
                    <span>Outliers</span>
                    <select value={step.outliers} onChange={(e) => setStep(col, "outliers", e.target.value)}>
                      {opts.outliers.map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </label>
                  {/* Parameters appear only for the method actually chosen, pre-filled
                      with the default -- the guided path stays one click (plan §8.4). */}
                  {(OUTLIER_PARAMS[step.outliers] || []).map((p) => (
                    <label className="pp-field" key={p.key} style={{ maxWidth: 92 }}>
                      <span>{p.label}</span>
                      <input type="number" step={p.step} min={p.min} max={p.max}
                             value={pval(opts, step, step.outliers, p)}
                             onChange={(e) => setParam(col, step.outliers, p.key, e.target.value)} />
                    </label>
                  ))}
                  <label className="pp-field">
                    <span>Scale</span>
                    <select value={step.scale} onChange={(e) => setStep(col, "scale", e.target.value)}>
                      {opts.scale.map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </label>
                </>
              ) : (
                <>
                  <label className="pp-field">
                    <span>Encode</span>
                    <select value={step.encode} onChange={(e) => setStep(col, "encode", e.target.value)}>
                      {opts.encode.map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </label>
                </>
              )}

              <pre className="pp-code">{opCode(opts, step, numeric, col).join("\n")}</pre>
              <p className="pp-reason">{aiNotes[col] || ruleNote(opts, step, numeric)}</p>
            </div>
          );
        })}
      </div>

      {error && <div className="error">{error}</div>}

      {/* Step 6: results + what changed */}
      {result && (
        <div className="panel" style={{ marginTop: 12, padding: 16 }}>
          <div className="target-note" style={{ marginBottom: 10 }}>
            {result.rows_before.toLocaleString()} rows · {result.features_in} → {result.features_out} features ·
            train/test {result.train_rows.toLocaleString()}/{result.test_rows.toLocaleString()} ·
            {result.stratified ? " stratified" : " random"} split
            {` ${Math.round((1 - result.test_size) * 100)}:${Math.round(result.test_size * 100)}`} ·
            seed {result.random_state}
            {result.dropped_columns.length > 0 && ` · dropped: ${result.dropped_columns.join(", ")}`}
          </div>

          {result.notes?.length > 0 && (
            <div className="pp-ai" style={{ marginBottom: 10 }}>
              {result.notes.map((n, i) => <p key={i} style={{ margin: 0 }}>{n}</p>)}
            </div>
          )}

          <details className="pp-summary" open>
            <summary>What changed, per column</summary>
            <table>
              <thead><tr><th>Column</th><th>Type before → after</th><th>Operations</th></tr></thead>
              <tbody>
                {result.summary.map((s) => (
                  <tr key={s.column}>
                    <td>{s.column}</td>
                    <td>{s.dtype_before} → {s.dtype_after}</td>
                    <td>{s.ops.join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>

          {result.pipeline && result.pipeline.length > 0 && (
            <div className="pp-ai" style={{ marginTop: 10 }}>
              <b>Dataset-level steps applied:</b>
              <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
                {result.pipeline.map((n, i) => <li key={i}>{n}</li>)}
              </ul>
              {result.class_weights && (
                <p style={{ margin: "6px 0 0" }}>
                  Class weights — {Object.entries(result.class_weights).map(([k, v]) => `${k}: ${v}`).join(" · ")}
                  {" "}(use these in your model, e.g. <code>class_weight=</code>).
                </p>
              )}
            </div>
          )}

          <PreviewTable rows={result.preview} />
          <a
            className="ghost"
            style={{ display: "inline-block", marginTop: 12, padding: "8px 16px", textDecoration: "none" }}
            href={`${API}/api/download?id=${data.id}`}
          >
            Download cleaned CSV
          </a>
          <a
            className="ghost"
            style={{ display: "inline-block", marginTop: 12, marginLeft: 8, padding: "8px 16px", textDecoration: "none" }}
            href={`${API}/api/pipeline?id=${data.id}`}
            title="Fitted steps + text TF-IDF, to apply the same preprocessing at training and on new rows"
          >
            Download pipeline
          </a>
        </div>
      )}
    </section>
  );
}

function PreviewTable({ rows }) {
  if (!rows || !rows.length) return null;
  const cols = Object.keys(rows[0]);
  const fmt = (v) =>
    typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 3 }) : String(v);
  return (
    <div className="tablescroll" style={{ maxHeight: 320, overflow: "auto" }}>
      <table>
        <thead>
          <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.slice(0, 10).map((r, i) => (
            <tr key={i}>{cols.map((c) => <td key={c}>{fmt(r[c])}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
