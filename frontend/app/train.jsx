"use client";

import { useEffect, useRef, useState } from "react";
import { call, download, dur, fmt } from "./apiclient";
import Clusters from "./clusters";
import DataPreview from "./datapreview";
import EnsemblePicker from "./ensembles";
import EpochChart from "./epochchart";
import Icon from "./icons";
import ModelCards, { Preparing } from "./modelcards";
import { ModelWindow, RunNotice } from "./modelwindow";
import RunHistory from "./runs";
import ModelTester from "./tester";
import TuneSettings, { TuneTrials, gridCount } from "./tuning";

const ACTIVE = ["queued", "running"];
const POLL_MS = 1500; // our starting point: status refresh while a run is active
const POLL_RETRIES = 3; // our starting point: failed status reads in a row before giving up
const LEADERS_K = 3;  // the backend's leader rule default (training/tuning.py)
const NOTIFY_KEY = "modelforge_notify_done"; // this browser's "Notify me when it finishes" choice

// Train tab: pick models (suggested pre-ticked, incompatible disabled with the
// reason), set their parameters, optionally add ensembles or search parameters, run
// CV in the background, watch live epoch curves, inspect each model in a window
// (charts, explanation, learning / validation curves, epochs), compare runs, then
// test and save. Data preprocessed for clustering gets the clustering view instead.
export default function Train({ data, target, active, onResults, onAsk, onTrained, onShowTraining }) {
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
  const [detail, setDetail] = useState(null);    // {key, tab: charts|explain|curves|epochs} -> model window
  const [notice, setNotice] = useState(null);    // {ok, title, body} -> "training finished" card
  const [notify, setNotify] = useState(false);   // desktop pop-up when a run ends in a background tab
  const [refreshKey, setRefreshKey] = useState(0);
  const timer = useRef(null);
  const notifyRef = useRef(false);               // read by the poll closure, which outlives renders

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
  useEffect(() => {
    let on = false;
    try { on = localStorage.getItem(NOTIFY_KEY) === "1"; } catch {}
    on = on && typeof Notification !== "undefined" && Notification.permission === "granted";
    setNotify(on); notifyRef.current = on;
  }, []);

  async function toggleNotify(e) {
    const want = e.target.checked;
    if (want && Notification.permission === "default") await Notification.requestPermission();
    const on = want && Notification.permission === "granted";
    setNotify(on); notifyRef.current = on;
    if (want && !on) setErr("This browser blocks notifications for this site. Allow them in the site settings to use “Notify me”.");
    try { localStorage.setItem(NOTIFY_KEY, on ? "1" : ""); } catch {}
  }

  // A train / search run just ended: show the card, and a desktop pop-up when asked for
  // and the page is in a background tab (the card is enough when it is in view).
  function announce(r, s, res) {
    const what = r.kind === "tune" ? "Search and training" : "Training";
    let n;
    if (s.state === "done") {
      const primary = res.primary_metric;
      const best = res.best && res.models?.[res.best];
      const metric = opt?.metrics.find((m) => m.key === primary)?.label || primary;
      const count = Object.values(res.models || {}).filter((m) => !m.baseline && !m.error).length;
      const secs = s.finished && (s.started || s.created) ? s.finished - (s.started || s.created) : null;
      n = { ok: true, title: `${what} finished`,
            body: [best && `Best: ${best.label}, ${metric} ${fmt(best.cv?.[primary]?.mean)} (CV mean).`,
                   `${count} model${count === 1 ? "" : "s"} trained${secs != null ? ` in ${dur(secs)}` : ""}.`].filter(Boolean).join(" ") };
    } else if (s.state === "cancelled") {
      n = { ok: false, title: `${what} cancelled` };
    } else {
      n = { ok: false, title: s.state === "timed_out" ? `${what} hit the time limit` : `${what} failed`, body: s.error || "" };
    }
    setNotice(n);
    if (notifyRef.current && document.hidden && typeof Notification !== "undefined" && Notification.permission === "granted") {
      const pop = new Notification(n.title, { body: n.body || "" });
      pop.onclick = () => { window.focus(); pop.close(); };
    }
  }

  function viewResults() {
    setNotice(null);
    onShowTraining?.();
    setTimeout(() => document.getElementById("train-results")?.scrollIntoView({ behavior: "smooth", block: "start" }), 60);
  }

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

  function poll(r, misses = 0) {
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
        else if (r.kind !== "estimate") {
          setRefreshKey((k) => k + 1);
          if (s.state === "done") onTrained?.();
          announce(r, s, res);
        }
      } catch (e) {
        // one failed status read must not end the watch: retry a few times, then report
        if (misses < POLL_RETRIES) poll(r, misses + 1); else setErr(e.message);
      }
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
      setRun(r); setStatus({ state: "queued" }); setDetail(null);
      if (kind !== "estimate") setNotice(null);
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
        {loading && <Preparing withAi={!!target} />}
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
  const modelWindow = (key) => {
    const r = results.models[key];
    const primary = results.primary_metric;
    const metric = opt.metrics.find((m) => m.key === primary)?.label || primary;
    const tabs = ["charts", "explain", "curves", ...(opt.models.find((m) => m.key === key)?.epochs ? ["epochs"] : [])];
    const tags = [results.best === key && `Best by ${metric}`, r.family === "Ensemble" && "ensemble", r.device === "gpu" && "GPU"]
      .filter(Boolean);
    return (
      <ModelWindow key={`${run.id}:${key}`} dsId={data.id} runId={run.id} model={key} label={r.label} tags={tags}
                   summary={`${metric} ${fmt(r.test?.[primary])} on the test set, ${fmt(r.cv?.[primary]?.mean)} ± ${fmt(r.cv?.[primary]?.sd)} in cross-validation`}
                   tabs={tabs} tab={detail.tab} onTab={(tab) => setDetail((d) => ({ ...d, tab }))} sweep={sweepable(key)}
                   onTune={!running && opt.models.some((m) => m.key === key) ? () => tuneOnly([key]) : null}
                   onAsk={onAsk ? () => onAsk(`How well does ${r.label} work? Compare its train, CV and test scores - did it overfit?`) : null}
                   onClose={() => setDetail(null)} />
    );
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
        {/* plan §11.4: on imbalanced data accuracy alone is not a sufficient yardstick */}
        {opt.task === "classification" && opt.facts?.imbalanced && primary === "accuracy" && (
          <p className="error" role="alert" style={{ margin: "10px 0 0" }}>
            The classes are imbalanced (smallest: {opt.facts.minority}, {Object.values(opt.facts.class_counts || {}).length
              ? `${Math.round(100 * Math.min(...Object.values(opt.facts.class_counts)) / Object.values(opt.facts.class_counts).reduce((a, b) => a + b, 0))}% of rows`
              : "a small share"}). Accuracy can look high while the small class is never predicted — a model that always
            answers the majority class scores the majority share. Rank by F1, recall, MCC, balanced accuracy or PR-AUC instead,
            and read accuracy only beside them.
          </p>
        )}
      </div>

      <ModelCards models={opt.models} suggestion={suggestion} picked={picked} onToggle={toggle}
                  params={params} onParams={(key, v) => setParams((p) => ({ ...p, [key]: v }))}
                  running={running} estimates={estimates} />

      <EnsemblePicker kinds={opt.ensembles || {}} models={opt.models} value={ensembles} onChange={setEnsembles} disabled={running} />

      <label id="tune-settings" style={{ display: "flex", gap: 8, alignItems: "center", margin: "12px 0 8px", scrollMarginTop: 80 }}>
        <input type="checkbox" checked={tuneOn} disabled={running} onChange={(e) => setTuneOn(e.target.checked)} />
        <b>Search hyperparameters for the ticked models</b> <span className="note">(grid or random search)</span>
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
        {typeof Notification !== "undefined" && (
          <label className="notify-toggle"
                 title={window.isSecureContext ? "A desktop pop-up when training ends while this page is in a background tab"
                                               : "Browsers only allow notifications on https:// sites (and localhost)"}>
            <input type="checkbox" checked={notify} disabled={!window.isSecureContext} onChange={toggleNotify} />
            Notify me when it finishes{!window.isSecureContext && " (needs https)"}
          </label>
        )}
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
        <Results results={results} metrics={opt.metrics} done={!running} onAsk={onAsk}
                 onDetails={(key) => setDetail({ key, tab: "charts" })}
                 onDownload={() => download(`/api/run/model?id=${data.id}&run=${run.id}`, "best_model.joblib").catch((e) => setErr(e.message))}
                 onTuneLeaders={run?.kind === "train" && !running ? tuneLeaders : null} />
      )}
      {detail && run && results?.models?.[detail.key] && modelWindow(detail.key)}
      <RunNotice notice={notice} onView={viewResults} onClose={() => setNotice(null)} />

      <RunHistory dsId={data.id} metrics={opt.metrics} refreshKey={refreshKey}
                  onOpen={(r) => { setRun({ id: r.run_id, kind: r.kind }); setStatus({ state: r.state }); setDetail(null); setCurves(null);
                                   call(`/api/run/results?id=${data.id}&run=${r.run_id}`).then((res) => { setResults(res); setTuning(res.tuning || null); }).catch((e) => setErr(e.message)); }} />
      <ModelTester dsId={data.id} metrics={opt.metrics} refreshKey={refreshKey} />
    </section>
  );
}

// Run results as a leaderboard: the best model with its actions, one short row per
// model (rank, bar, train / CV / test of the primary metric, one Details button that
// opens the model window), and every other metric in a collapsed full table.
function Results({ results, metrics, done, onDetails, onAsk, onDownload, onTuneLeaders }) {
  const primary = results.primary_metric;
  const meta = Object.fromEntries(metrics.map((m) => [m.key, m]));
  const higher = meta[primary]?.higher_is_better ?? true;
  const metric = meta[primary]?.label || primary;
  const rows = Object.entries(results.models).map(([key, r]) => ({ key, ...r }));
  const mean = (r) => r.cv?.[primary]?.mean;
  const sd = (r) => r.cv?.[primary]?.sd;
  const scored = rows.filter((r) => !r.baseline && mean(r) != null)
    .sort((a, b) => (higher ? mean(b) - mean(a) : mean(a) - mean(b)));
  const order = [...scored, ...rows.filter((r) => !r.baseline && mean(r) == null), ...rows.filter((r) => r.baseline)];
  const shown = metrics.filter((m) => m.key !== primary && rows.some((r) => r.cv?.[m.key]));
  const best = results.best && results.models[results.best];

  // Bar = how far a model's CV mean is from the baseline's, as a share of the best
  // model's distance (works for "lower is better" too: both distances are negative).
  const base = rows.find((r) => r.baseline);
  const span = base && scored.length ? mean(scored[0]) - mean(base) : 0;
  const bars = !!base && mean(base) != null && (higher ? span > 0 : span < 0);
  const pos = (v) => Math.max(0, Math.min(100, ((v - mean(base)) / span) * 100));

  const tags = (r) => (
    <>
      {results.best === r.key && <span className="tag">best</span>}
      {r.baseline && <span className="tag">reference</span>}
      {r.family === "Ensemble" && <span className="tag">ensemble</span>}
      {r.device === "gpu" && <span className="tag">GPU</span>}
    </>
  );

  return (
    <div id="train-results" className="lb-wrap">
      {done && best && (
        <div className="panel lb-best">
          <div>
            <p className="lb-best-title">Best model: {best.label}</p>
            <p className="lb-best-sub">
              {metric} {fmt(mean(best))} ± {fmt(sd(best))} in cross-validation, {fmt(best.test?.[primary])} on the test set
            </p>
          </div>
          <div className="lb-best-actions">
            <button type="button" className="ghost" onClick={onDownload}><Icon name="download" size={16} /> Download model</button>
            {onTuneLeaders && <button type="button" className="ghost" onClick={onTuneLeaders}
                                      title="Pre-tick the best models that beat the baseline and open the search settings">
              Tune the top {LEADERS_K}</button>}
          </div>
        </div>
      )}
      <p className="lb-intro">
        Ranked by <b>{metric}</b> (CV mean, {higher ? "higher" : "lower"} is better).
        {bars && (higher ? " The bar shows how far above the baseline a model scores." : " The bar shows how far below the baseline's error a model gets.")}
        {" "}Use the test score in your report; train far better than CV means the model memorised its training rows (overfitting).
        {results.positive_class != null && <> Positive class: <b>{results.positive_class}</b>.</>}
      </p>
      <div className="panel lb-panel">
        <table className="lb">
          <thead>
            <tr>
              <th className="lb-rank">#</th><th>Model</th>{bars && <th className="lb-bar">Gain over baseline</th>}
              <th className="lb-num">Train</th><th className="lb-num">CV mean ± sd</th><th className="lb-num">Test</th>
              <th><span className="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            {order.map((r) => {
              const rank = scored.indexOf(r);
              const m = mean(r);
              return (
                <tr key={r.key} className={results.best === r.key ? "is-best" : undefined}>
                  <td className="lb-rank">{rank >= 0 ? rank + 1 : "–"}</td>
                  <td className="lb-model">
                    <span className="lb-name">{r.label}{tags(r)}</span>
                    {r.note && <span className="lb-note">{r.note}</span>}
                  </td>
                  {r.error ? (
                    <td colSpan={bars ? 4 : 3} className="lb-failed">failed: {r.error}</td>
                  ) : (
                    <>
                      {bars && (
                        <td className="lb-bar" data-label="Gain over baseline">
                          {r.baseline ? <span className="lb-note">the bars start here</span>
                            : m == null ? null
                            : pos(m) === 0 ? <span className="lb-note">no better than the baseline</span>
                            : (
                              <span className="lb-track" role="img"
                                    aria-label={`${Math.round(pos(m))}% of the best model's gain over the baseline`}>
                                <span className="lb-fill" style={{ width: `${pos(m)}%` }} />
                                {sd(r) != null && (
                                  <span className="lb-sd-mark" style={{ left: `${Math.min(pos(m - sd(r)), pos(m + sd(r)))}%`,
                                                                        width: `${Math.abs(pos(m + sd(r)) - pos(m - sd(r)))}%` }} />
                                )}
                              </span>
                            )}
                        </td>
                      )}
                      <td className="lb-num lb-train" data-label="Train">{fmt(r.train?.[primary])}</td>
                      <td className="lb-num" data-label="CV mean ± sd">{fmt(m)} <span className="lb-sd">± {fmt(sd(r))}</span></td>
                      <td className="lb-num lb-test" data-label="Test">{fmt(r.test?.[primary])}</td>
                    </>
                  )}
                  <td className="lb-act">
                    {!r.error && r.test && done && !r.baseline && (
                      <button type="button" className="ghost lb-btn" onClick={() => onDetails(r.key)}>Details</button>
                    )}
                    {!r.error && r.test && done && r.baseline && onAsk && (
                      <button type="button" className="ghost lb-btn"
                              onClick={() => onAsk(`How well does ${r.label} work? Compare its train, CV and test scores - did it overfit?`)}>
                        Ask AI</button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <details className="lb-all">
        <summary>All metrics{shown.length ? ` (${shown.map((m) => m.label).join(", ")})` : ""}</summary>
        <div className="panel tablescroll lb-all-table">
          <table>
            <thead>
              <tr>
                <th>Model</th><th>{metric} train</th><th>{metric} CV mean ± sd</th><th>{metric} test</th>
                {shown.map((m) => <th key={m.key}>{m.label} (CV)</th>)}
                <th>Epochs (trained / best)</th><th>Fit s/fold</th>
              </tr>
            </thead>
            <tbody>
              {order.map((r) => (
                <tr key={r.key}>
                  <td><span className="lb-name">{r.label}{tags(r)}</span></td>
                  {r.error ? (
                    <td colSpan={5 + shown.length} className="note">failed: {r.error}</td>
                  ) : (
                    <>
                      <td>{fmt(r.train?.[primary])}</td>
                      <td>{fmt(mean(r))} ± {fmt(sd(r))}</td>
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
        </div>
      </details>
    </div>
  );
}
