"use client";

import { useEffect, useRef, useState } from "react";
import { call, download, dur, fmt } from "./apiclient";
import Clusters from "./clusters";
import DataPreview from "./datapreview";
import Diagnostics from "./diagnostics";
import EnsemblePicker from "./ensembles";
import EpochChart from "./epochchart";
import Explain from "./explain";
import ModelCards from "./modelcards";
import ResultCharts from "./resultcharts";
import RunHistory from "./runs";
import ModelTester from "./tester";
import TuneSettings, { TuneTrials, gridCount } from "./tuning";

const ACTIVE = ["queued", "running"];
const POLL_MS = 1500; // our starting point: status refresh while a run is active
const LEADERS_K = 3;  // the backend's leader rule default (training/tuning.py)

// Train tab: pick models (suggested pre-ticked, incompatible disabled with the
// reason), set their parameters, optionally add ensembles or search parameters, run
// CV in the background, watch live epoch curves, inspect each model (charts,
// explanation, learning / validation curves), compare runs, then test and save.
// Data preprocessed for clustering gets the clustering view instead.
export default function Train({ data, target, active, onResults, onAsk, onTrained }) {
  const [opt, setOpt] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [picked, setPicked] = useState([]);
  const [params, setParams] = useState({});      // model key -> edited values
  const [suggestion, setSuggestion] = useState(null);
  const [touched, setTouched] = useState(false);
  const [primary, setPrimary] = useState("");
  const [folds, setFolds] = useState(5);
  const [limitMin, setLimitMin] = useState(60);
  const [share, setShare] = useState(0.1);
  const [positive, setPositive] = useState("");
  const [ensembles, setEnsembles] = useState([]);
  const [tuneOn, setTuneOn] = useState(false);
  const [tune, setTune] = useState({ method: "random", nTrials: 20, space: {}, grid: {} });
  const [spaces, setSpaces] = useState({});
  const [run, setRun] = useState(null); // {id, kind}
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [tuning, setTuning] = useState(null);
  const [estimates, setEstimates] = useState({});
  const [curves, setCurves] = useState(null);     // {key, data}
  const [panel, setPanel] = useState(null);      // {kind: charts|explain|curves, key, data?}
  const [refreshKey, setRefreshKey] = useState(0);
  const timer = useRef(null);

  async function load() {
    setLoading(true); setErr("");
    try {
      const o = await call(`/api/train/options?id=${data.id}`);
      setOpt(o);
      if (o.task === "clustering") return;
      setPicked([]); setTouched(false); setSuggestion(null); setEnsembles([]);
      loadSuggestion();
      call(`/api/train/space?id=${data.id}`).then(setSpaces).catch(() => {});
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
    try { s = await call("/api/train/suggest", { id: data.id }); }
    catch (e) { s = { source: "rules", reason: `suggestions unavailable: ${e.message}`, models: [] }; }
    setSuggestion(s);
    setTouched((wasTouched) => {
      if (!wasTouched) setPicked(s.models.map((m) => m.key));  // pre-tick only if the user hasn't chosen yet
      return wasTouched;
    });
  }

  async function showCurves(r, key) {
    try { setCurves({ key, data: await call(`/api/run/epochs?id=${data.id}&run=${r.id}&model=${key}`) }); } catch {}
  }

  async function showCharts(key) {
    try { setPanel({ kind: "charts", key, data: await call(`/api/run/curves?id=${data.id}&run=${run.id}&model=${key}`) }); }
    catch (e) { setErr(e.message); }
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
          setResults(res.models ? res : null);
          if (r.kind === "tune") setTuning(res.tuning || await call(`/api/run/tuning?id=${data.id}&run=${r.id}`));
          const key = s.progress?.key;
          if (key && opt?.models.find((m) => m.key === key)?.epochs) await showCurves(r, key);
        }
        if (ACTIVE.includes(s.state)) poll(r);
        else if (r.kind !== "estimate") { setRefreshKey((k) => k + 1); if (s.state === "done") onTrained?.(); }
      } catch (e) { setErr(e.message); }
    }, POLL_MS);
  }

  function body() {
    const keys = [...new Set([...picked, ...ensembles.flatMap((e) => e.members)])];
    return { id: data.id, models: picked, folds: Number(folds), primary_metric: primary,
             time_limit_min: Number(limitMin), positive_class: positive || null, validation_share: Number(share),
             params: Object.fromEntries(keys.map((k) => [k, params[k] || {}])),
             ensembles: ensembles.filter((e) => e.members.length).map(({ kind, members, weights }) => ({ kind, members, weights })) };
  }

  async function start(kind) {
    setErr("");
    try {
      let path = "/api/train", payload = body();
      if (kind === "estimate") path = "/api/train/estimate";
      if (kind === "tune") {
        path = "/api/train/tune";
        payload = { ...payload, method: tune.method, n_trials: Number(tune.nTrials), space: tune.space, grid: tune.grid };
      }
      const { run: id } = await call(path, payload);
      const r = { id, kind };
      setRun(r); setStatus({ state: "queued" }); setPanel(null);
      if (kind === "estimate") setEstimates({}); else { setResults(null); setCurves(null); setTuning(null); }
      poll(r);
    } catch (e) { setErr(e.message); }
  }

  // Tick exactly these models, switch the search on and bring its settings into view.
  function tuneOnly(keys) {
    setPicked(keys); setTouched(true); setTuneOn(true);
    setTimeout(() => document.getElementById("tune-settings")?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
  }

  async function tuneLeaders() {
    try {
      const keys = await call(`/api/run/leaders?id=${data.id}&run=${run.id}&k=${LEADERS_K}`);
      if (!keys.length) { setErr("No model beat the baseline, so there is nothing to tune."); return; }
      tuneOnly(keys);
    } catch (e) { setErr(e.message); }
  }

  async function cancel() {
    try { await call("/api/run/cancel", { id: data.id, run: run.id }); } catch (e) { setErr(e.message); }
  }

  if (opt?.task === "clustering") return <Clusters data={data} opt={opt} onReload={load} onTrained={onTrained} />;

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
  const toggle = (key) => { setTouched(true); setPicked((p) => (p.includes(key) ? p.filter((k) => k !== key) : [...p, key])); };
  const f = opt.facts;
  const prog = status?.progress || {};
  const anyEpochs = picked.some((k) => opt.models.find((m) => m.key === k)?.epochs);
  const searchFits = picked.reduce((n, k) => {
    const sp = { ...(spaces[k]?.space || {}) };
    return n + (tune.method === "grid" ? gridCount(sp, tune.grid[k]) : Number(tune.nTrials) || 0) * Number(folds);
  }, 0);
  const label = (key) => results?.models?.[key]?.label || opt.models.find((m) => m.key === key)?.label || key;
  const sweepable = (key) => {
    const m = opt.models.find((x) => x.key === key);
    return m ? m.params.filter((p) => spaces[key]?.space?.[p.name]) : [];
  };

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
        {opt.group ? ` · folds keep each ${opt.group} together` : ""}
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

      <EnsemblePicker kinds={opt.ensembles || {}} models={opt.models} value={ensembles} onChange={setEnsembles} disabled={running} />

      <label id="tune-settings" style={{ display: "flex", gap: 8, alignItems: "center", margin: "12px 0 8px", scrollMarginTop: 80 }}>
        <input type="checkbox" checked={tuneOn} disabled={running} onChange={(e) => setTuneOn(e.target.checked)} />
        <b>Search hyperparameters for the ticked models</b> <span className="note">(grid or random search, plan §10)</span>
      </label>
      {tuneOn && picked.length > 0 && (
        <TuneSettings models={opt.models} picked={picked} spaces={spaces} value={tune} onChange={setTune} disabled={running} />
      )}

      <div className="panel" style={{ padding: 16, marginBottom: 12 }}>
        {tuneOn
          ? <button className="primary" disabled={running || !picked.length} onClick={() => start("tune")}>
              Search + train ({searchFits.toLocaleString()} fits)</button>
          : <button className="primary" disabled={running || (!picked.length && !ensembles.length)} onClick={() => start("train")}>Run training</button>}{" "}
        <button className="ghost" disabled={running || !picked.length} onClick={() => start("estimate")}
                title="Fits each ticked model (with its parameters) on 250 and 500 rows and projects the full run">Estimate time</button>{" "}
        {running && <button className="ghost" onClick={cancel}>Cancel</button>}{" "}
        <button className="ghost" disabled={running} onClick={load}>Refresh (after re-preprocessing)</button>
        {status && (
          <p className="note" style={{ marginTop: 8 }}>
            {run?.kind === "estimate" ? "Estimate" : run?.kind === "tune" ? "Search + training" : "Training"}: {status.state}
            {running && prog.step === "search" && ` · searching ${prog.model} (${prog.model_index}/${prog.models}) · trial ${prog.trial}/${prog.trials}`}
            {running && prog.step !== "search" && prog.model && ` · ${prog.model} (${prog.model_index}/${prog.models})`}
            {running && prog.fold && ` · fold ${prog.fold}/${prog.folds}`}
            {running && prog.epoch && ` · epoch ${prog.epoch}/${prog.epochs}`}
            {running && prog.eta_seconds != null && ` · ~${dur(prog.eta_seconds)} left`}
            {status.error && ` · ${status.error}`}
          </p>
        )}
      </div>

      {err && <div className="error">{err}</div>}
      {tuning && <TuneTrials tuning={tuning} metrics={opt.metrics} />}
      {curves && <EpochChart data={curves.data} title={label(curves.key)} />}
      {results?.models && (
        <Results results={results} metrics={opt.metrics} models={opt.models} done={!running}
                 onCurves={(key) => showCurves(run, key)} onAsk={onAsk}
                 onPanel={(kind, key) => (kind === "charts" ? showCharts(key) : setPanel({ kind, key }))}
                 onDownload={() => download(`/api/run/model?id=${data.id}&run=${run.id}`, "best_model.joblib").catch((e) => setErr(e.message))}
                 onTuneLeaders={run?.kind === "train" && !running ? tuneLeaders : null}
                 onTune={running ? null : (key) => tuneOnly([key])} />
      )}
      {panel?.kind === "charts" && (
        <div style={{ marginTop: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <b>Test-set charts — {label(panel.key)}</b><button className="ghost" onClick={() => setPanel(null)}>Close</button>
          </div>
          <ResultCharts data={panel.data} title={label(panel.key)} />
        </div>
      )}
      {panel?.kind === "explain" && run && (
        <Explain dsId={data.id} runId={run.id} model={panel.key} label={label(panel.key)} onClose={() => setPanel(null)} />
      )}
      {panel?.kind === "curves" && run && (
        <Diagnostics dsId={data.id} runId={run.id} model={panel.key} label={label(panel.key)} params={sweepable(panel.key)}
                     onClose={() => setPanel(null)} />
      )}

      <RunHistory dsId={data.id} metrics={opt.metrics} refreshKey={refreshKey}
                  onOpen={(r) => { setRun({ id: r.run_id, kind: r.kind }); setStatus({ state: r.state }); setPanel(null); setCurves(null);
                                   call(`/api/run/results?id=${data.id}&run=${r.run_id}`).then((res) => { setResults(res); setTuning(res.tuning || null); }).catch((e) => setErr(e.message)); }} />
      <ModelTester dsId={data.id} metrics={opt.metrics} refreshKey={refreshKey} />
    </section>
  );
}

function Results({ results, metrics, models, done, onCurves, onPanel, onAsk, onDownload, onTuneLeaders, onTune }) {
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
        Train is the score on the model&apos;s own training rows: far better than CV means it memorised them (overfitting).
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
                {r.family === "Ensemble" && <span className="tag" style={{ marginLeft: 6 }}>ensemble</span>}
                {r.device === "gpu" && <span className="tag" style={{ marginLeft: 6 }}>GPU</span>}
                {results.best === r.key && <span className="tag" style={{ marginLeft: 6 }}>Best by {meta[primary]?.label} (CV mean)</span>}
                {r.note && <div className="note">{r.note}</div>}
                {!r.error && r.test && done && (
                  <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 4 }}>
                    {hasEpochs(r.key) && <button className="ghost" onClick={() => onCurves(r.key)}>Epochs</button>}
                    {!r.baseline && <button className="ghost" onClick={() => onPanel("charts", r.key)}>Charts</button>}
                    {!r.baseline && <button className="ghost" onClick={() => onPanel("explain", r.key)}>Explain</button>}
                    {!r.baseline && <button className="ghost" onClick={() => onPanel("curves", r.key)}>Learning curves</button>}
                    {onTune && !r.baseline && models.some((m) => m.key === r.key) && (
                      <button className="ghost" onClick={() => onTune(r.key)}
                              title="Tick only this model and open the hyperparameter search">Tune</button>
                    )}
                    {onAsk && <button className="ghost" onClick={() => onAsk(`How well does ${r.label} work? Compare its train, CV and test scores - did it overfit?`)}>Ask AI</button>}
                  </div>
                )}
              </td>
              {r.error ? (
                <td colSpan={5 + shown.length} className="note">failed: {r.error}</td>
              ) : (
                <>
                  <td>{fmt(r.train?.[primary])}</td>
                  <td>{fmt(r.cv?.[primary]?.mean)} ± {fmt(r.cv?.[primary]?.sd)}</td>
                  <td>{fmt(r.test?.[primary])}</td>
                  {shown.map((m) => <td key={m.key}>{fmt(r.cv?.[m.key]?.mean)}</td>)}
                  <td>{r.epochs ? `${r.epochs.epochs_trained ?? "—"} / ${r.epochs.best_epoch ?? "—"}` : "—"}</td>
                  <td>{r.fit_seconds}</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      {done && results.best && (
        <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
          <button className="ghost" onClick={onDownload}>Download best model</button>
          {onTuneLeaders && <button className="ghost" onClick={onTuneLeaders}
                                    title="Pre-tick the best models that beat the baseline and open the search settings">
            Tune the top {LEADERS_K}</button>}
        </div>
      )}
    </div>
  );
}
