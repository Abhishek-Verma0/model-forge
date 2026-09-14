"use client";

import { useEffect, useRef, useState } from "react";
import { API, call, dur, fmt } from "./apiclient";
import DataPreview from "./datapreview";
import EpochChart from "./epochchart";
import ModelCards from "./modelcards";
import RunHistory from "./runs";
import ModelTester from "./tester";

const ACTIVE = ["queued", "running"];
const POLL_MS = 1500; // our starting point: status refresh while a run is active

// Train tab: pick models (suggested pre-ticked, incompatible disabled with the
// reason), set their parameters, run CV in the background, watch live epoch
// curves, compare runs, then test and save models.
export default function Train({ data, target, active, onResults, onAsk }) {
  const [opt, setOpt] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [picked, setPicked] = useState([]);
  const [params, setParams] = useState({});      // model key -> edited values
  const [suggestion, setSuggestion] = useState(null);  // {source, reason, models:[{key, why}]}
  const [touched, setTouched] = useState(false);       // user changed the selection by hand
  const [primary, setPrimary] = useState("");
  const [folds, setFolds] = useState(5);
  const [limitMin, setLimitMin] = useState(60);
  const [share, setShare] = useState(0.1);
  const [positive, setPositive] = useState("");
  const [run, setRun] = useState(null); // {id, kind}
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [estimates, setEstimates] = useState({});
  const [curves, setCurves] = useState(null);     // {key, data}
  const [refreshKey, setRefreshKey] = useState(0);
  const timer = useRef(null);

  async function load() {
    setLoading(true); setErr("");
    try {
      const o = await call(`/api/train/options?id=${data.id}`);
      setOpt(o);
      setPicked([]);
      setTouched(false);
      setSuggestion(null);
      loadSuggestion();
      setPrimary(o.defaults.primary_metric);
      setFolds(Math.min(o.defaults.folds, o.limits.folds_max));
      setLimitMin(o.defaults.time_limit_min);
      setShare(o.defaults.validation_share);
      setPositive(o.defaults.positive_class || "");
    } catch (e) { setErr(e.message); } finally { setLoading(false); }
  }

  useEffect(() => { if (active && !opt && !loading) load(); }, [active]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => clearTimeout(timer.current), []);
  useEffect(() => { onResults?.(results); }, [results]); // eslint-disable-line react-hooks/exhaustive-deps -- the chat reads what is on the page

  async function loadSuggestion() {
    let s;
    try {
      s = await call("/api/train/suggest", { id: data.id });
    } catch (e) {
      s = { source: "rules", reason: `suggestions unavailable: ${e.message}`, models: [] };
    }
    setSuggestion(s);
    setTouched((wasTouched) => {
      if (!wasTouched) setPicked(s.models.map((m) => m.key));  // pre-tick only if the user hasn't chosen yet
      return wasTouched;
    });
  }

  async function showCurves(r, key) {
    try { setCurves({ key, data: await call(`/api/run/epochs?id=${data.id}&run=${r.id}&model=${key}`) }); } catch {}
  }

  function poll(r) {
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        const [s, res] = await Promise.all([
          call(`/api/run?id=${data.id}&run=${r.id}`),
          call(`/api/run/results?id=${data.id}&run=${r.id}`),
        ]);
        setStatus(s);
        if (r.kind === "estimate") setEstimates(res.estimates || {});
        else {
          setResults(res);
          const key = s.progress?.key;
          if (key && opt?.models.find((m) => m.key === key)?.epochs) await showCurves(r, key);
        }
        if (ACTIVE.includes(s.state)) poll(r);
        else if (r.kind === "train") setRefreshKey((k) => k + 1);
      } catch (e) { setErr(e.message); }
    }, POLL_MS);
  }

  async function start(kind) {
    setErr("");
    try {
      const body = { id: data.id, models: picked, folds: Number(folds), primary_metric: primary,
                     time_limit_min: Number(limitMin), positive_class: positive || null,
                     validation_share: Number(share),
                     params: Object.fromEntries(picked.map((k) => [k, params[k] || {}])) };
      const { run: id } = await call(kind === "estimate" ? "/api/train/estimate" : "/api/train", body);
      const r = { id, kind };
      setRun(r); setStatus({ state: "queued" });
      if (kind === "estimate") setEstimates({}); else { setResults(null); setCurves(null); }
      poll(r);
    } catch (e) { setErr(e.message); }
  }

  async function cancel() {
    try { await call("/api/run/cancel", { id: data.id, run: run.id }); } catch (e) { setErr(e.message); }
  }

  if (!opt) {
    return (
      <section>
        <p className="sectitle">Train models</p>
        {loading && <p className="note">Measuring the preprocessed training data…</p>}
        {err && <div className="error">{err}</div>}
        {!loading && <button className="ghost" onClick={load}>Retry</button>}
      </section>
    );
  }

  const running = status && ACTIVE.includes(status.state);
  const toggle = (key) => {
    setTouched(true);
    setPicked((p) => (p.includes(key) ? p.filter((k) => k !== key) : [...p, key]));
  };
  const f = opt.facts;
  const prog = status?.progress || {};
  const anyEpochs = picked.some((k) => opt.models.find((m) => m.key === k)?.epochs);

  return (
    <section>
      <p className="sectitle">Train models — target: {opt.target} · {opt.task}</p>
      {target && target !== opt.target && (
        <div className="error" style={{ marginBottom: 10 }}>
          You selected “{target}” as the target, but the training data was preprocessed for “{opt.target}”.
          Run Preprocessing again to train on “{target}”.
        </div>
      )}
      <div className="target-note" style={{ marginBottom: 10 }}>
        {f.rows.toLocaleString()} training rows · {f.features.toLocaleString()} features
        {f.sparse ? " · text columns (sparse)" : ""}{f.nan ? " · missing values left" : ""}
        {f.class_counts ? ` · classes: ${Object.entries(f.class_counts).map(([c, n]) => `${c} ${n}`).join(", ")}` : ""}
        {" · "}baseline always included
      </div>
      {opt.notes.map((n, i) => <p key={i} className="note">{n}</p>)}
      <DataPreview dsId={data.id} />

      <div className="panel" style={{ padding: 16, marginBottom: 12 }}>
        <div className="builder-row">
          <label className="pp-field"><span>Primary metric</span>
            <select value={primary} onChange={(e) => setPrimary(e.target.value)}>
              {opt.metrics.map((m) => <option key={m.key} value={m.key}>{m.label}{m.higher_is_better ? "" : " (lower is better)"}</option>)}
            </select>
          </label>
          <label className="pp-field" style={{ maxWidth: 110 }}><span>Folds (2–{opt.limits.folds_max})</span>
            <input type="number" min="2" max={opt.limits.folds_max} value={folds} onChange={(e) => setFolds(e.target.value)} />
          </label>
          <label className="pp-field" style={{ maxWidth: 130 }}><span>Time limit (min)</span>
            <input type="number" min="1" value={limitMin} onChange={(e) => setLimitMin(e.target.value)} />
          </label>
          {anyEpochs && (
            <label className="pp-field" style={{ maxWidth: 190 }}
                   title="Epoch-based models hold back this share of each fold's training rows for the validation curve and early stopping. 0 = train on all rows, training loss only.">
              <span>Validation share (0–{opt.limits.validation_share_max})</span>
              <input type="number" min="0" max={opt.limits.validation_share_max} step="0.05" value={share} onChange={(e) => setShare(e.target.value)} />
            </label>
          )}
          {opt.classes?.length === 2 && (
            <label className="pp-field"><span>Positive class</span>
              <select value={positive} onChange={(e) => setPositive(e.target.value)}>
                {opt.classes.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
          )}
          <label className="pp-field"><span>Seed (from preprocessing)</span><input value={opt.defaults.seed} disabled /></label>
        </div>
      </div>

      <ModelCards models={opt.models} suggestion={suggestion} picked={picked} onToggle={toggle}
                  params={params} onParams={(key, v) => setParams((p) => ({ ...p, [key]: v }))}
                  running={running} estimates={estimates} />

      <div className="panel" style={{ padding: 16, marginBottom: 12 }}>
        <button className="primary" disabled={running || !picked.length} onClick={() => start("train")}>Run training</button>{" "}
        <button className="ghost" disabled={running || !picked.length} onClick={() => start("estimate")}
                title="Fits each ticked model (with its parameters) on 250 and 500 rows and projects the full run">Estimate time</button>{" "}
        {running && <button className="ghost" onClick={cancel}>Cancel</button>}{" "}
        <button className="ghost" disabled={running} onClick={load}>Refresh (after re-preprocessing)</button>
        {status && (
          <p className="note" style={{ marginTop: 8 }}>
            {run?.kind === "estimate" ? "Estimate" : "Training"}: {status.state}
            {running && prog.model && ` · ${prog.model} (${prog.model_index}/${prog.models})`}
            {running && prog.fold && ` · fold ${prog.fold}/${prog.folds}`}
            {running && prog.epoch && ` · epoch ${prog.epoch}/${prog.epochs}`}
            {running && prog.eta_seconds != null && ` · ~${dur(prog.eta_seconds)} left`}
            {status.error && ` · ${status.error}`}
          </p>
        )}
      </div>

      {err && <div className="error">{err}</div>}
      {curves && <EpochChart data={curves.data} title={opt.models.find((m) => m.key === curves.key)?.label || curves.key} />}
      {results?.models && (
        <Results results={results} metrics={opt.metrics} models={opt.models} dsId={data.id} runId={run?.id}
                 done={!running} onCurves={(key) => showCurves(run, key)} onAsk={onAsk} />
      )}

      <RunHistory dsId={data.id} metrics={opt.metrics} refreshKey={refreshKey} />
      <ModelTester dsId={data.id} metrics={opt.metrics} refreshKey={refreshKey} />
    </section>
  );
}

function Results({ results, metrics, models, dsId, runId, done, onCurves, onAsk }) {
  const primary = results.primary_metric;
  const meta = Object.fromEntries(metrics.map((m) => [m.key, m]));
  const higher = meta[primary]?.higher_is_better ?? true;
  const rows = Object.entries(results.models).map(([key, r]) => ({ key, ...r }));
  const mean = (r) => r.cv?.[primary]?.mean;
  const scored = rows.filter((r) => !r.baseline && mean(r) != null)
    .sort((a, b) => (higher ? mean(b) - mean(a) : mean(a) - mean(b)));
  const order = [...rows.filter((r) => r.baseline), ...scored, ...rows.filter((r) => !r.baseline && mean(r) == null)];
  const shown = metrics.filter((m) => m.key !== primary && rows.some((r) => r.cv?.[m.key]));
  const hasEpochs = (key) => models.find((m) => m.key === key)?.epochs;

  return (
    <div className="panel tablescroll" style={{ padding: 12, marginTop: 8 }}>
      <p style={{ margin: "0 0 8px" }}>
        Sorted by <b>{meta[primary]?.label}</b> (CV mean{higher ? ", higher is better" : ", lower is better"}).
        The test score is the honest number for your report; CV picks the model.
        Train is the score on the model's own training rows: far better than CV means it memorised them (overfitting).
        {results.positive_class && <> Positive class: <b>{results.positive_class}</b>.</>}
      </p>
      <table>
        <thead>
          <tr>
            <th>Model</th><th>{meta[primary]?.label} train</th>
            <th>{meta[primary]?.label} CV mean ± sd</th><th>{meta[primary]?.label} test</th>
            {shown.map((m) => <th key={m.key}>{m.label} (CV)</th>)}
            <th>Epochs (trained / best)</th><th>Fit s/fold</th>
          </tr>
        </thead>
        <tbody>
          {order.map((r) => (
            <tr key={r.key}>
              <td>
                {r.label}
                {r.baseline && <span className="tag" style={{ marginLeft: 6 }}>reference</span>}
                {r.device === "gpu" && <span className="tag" style={{ marginLeft: 6 }}>GPU</span>}
                {results.best === r.key && <span className="tag" style={{ marginLeft: 6 }}>Best by {meta[primary]?.label} (CV mean)</span>}
                {r.note && <div className="note">{r.note}</div>}
                {hasEpochs(r.key) && !r.error && <div><button className="ghost" onClick={() => onCurves(r.key)}>Epoch curves</button></div>}
                {onAsk && done && (
                  <div><button className="ghost" onClick={() => onAsk(`How well does ${r.label} work? Compare its train, CV and test scores - did it overfit?`)}>
                    Ask AI
                  </button></div>
                )}
              </td>
              {r.error ? (
                <td colSpan={5 + shown.length} className="note">failed: {r.error}</td>
              ) : (
                <>
                  <td>{fmt(r.train?.[primary])}</td>
                  <td>{fmt(r.cv[primary]?.mean)} ± {fmt(r.cv[primary]?.sd)}</td>
                  <td>{fmt(r.test[primary])}</td>
                  {shown.map((m) => <td key={m.key}>{fmt(r.cv[m.key]?.mean)}</td>)}
                  <td>{r.epochs ? `${r.epochs.epochs_trained ?? "—"} / ${r.epochs.best_epoch ?? "—"}` : "—"}</td>
                  <td>{r.fit_seconds}</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      {done && results.best && (
        <a className="ghost" style={{ display: "inline-block", marginTop: 12, padding: "8px 16px", textDecoration: "none" }}
           href={`${API}/api/run/model?id=${dsId}&run=${runId}`}>Download best model</a>
      )}
    </div>
  );
}
