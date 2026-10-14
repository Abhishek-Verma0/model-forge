"use client";

// User and project dashboard (plan §8.1): the account's projects and how far each got,
// the latest experiment results, saved reports, quick starts and the analysis tools.
import { useEffect, useState } from "react";
import { call, download, fmt } from "./apiclient";
import Icon from "./icons";

const RESULT_PROJECTS = 4; // our starting point: projects whose latest run is summarised
const RECENT = 6;          // our starting point: projects listed before "View all"
const TRAINED = ["train", "tune", "cluster"];

const rel = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
export function ago(t) {
  if (!t) return "";
  const s = t - Date.now() / 1000;
  for (const [unit, n] of [["year", 31536000], ["month", 2592000], ["week", 604800], ["day", 86400], ["hour", 3600], ["minute", 60]])
    if (Math.abs(s) >= n) return rel.format(Math.round(s / n), unit);
  return "just now";
}

/** Where a saved project stands, from the projects list. */
export function stage(p) {
  if (p.kind === "bibliometric") return { label: "Publications", tone: "neutral" };
  if (p.runs > 0) return { label: "Trained", tone: "good" };
  if (p.preprocessed) return { label: "Ready to train", tone: "accent" };
  return { label: "Uploaded", tone: "neutral" };
}

export default function Dashboard({ user, samples, current, onOpen, onContinue, onNew, onSample, onGo }) {
  const [list, setList] = useState(null);
  const [results, setResults] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState("");

  useEffect(() => {
    let live = true;
    call("/api/projects").then(async (ps) => {
      if (!live) return;
      setList(ps);
      const withRuns = ps.filter((p) => p.kind !== "bibliometric" && p.runs > 0).slice(0, RESULT_PROJECTS);
      const rows = await Promise.all(withRuns.map(async (p) => {
        try {
          const runs = await call(`/api/runs/summary?id=${p.id}`);
          const run = runs.find((r) => TRAINED.includes(r.kind) && r.state === "done");
          return run ? { p, run } : null;
        } catch { return null; }
      }));
      if (live) setResults(rows.filter(Boolean));
    }).catch((e) => live && setErr(e.message));
    return () => { live = false; };
  }, []);

  async function report(p) {
    setBusy(p.id);
    try { await download(`/api/report/pdf?id=${p.id}`, "report.pdf"); } catch (e) { setErr(e.message); } finally { setBusy(""); }
  }

  const datasets = list?.filter((p) => p.kind !== "bibliometric") || [];
  const stats = list && [
    ["Projects", list.length, "folder"],
    ["Ready to train", datasets.filter((p) => p.preprocessed).length, "sliders"],
    ["Training runs", datasets.reduce((n, p) => n + p.runs, 0), "cpu"],
    ["Saved models", datasets.reduce((n, p) => n + p.saved_models, 0), "database"],
  ];
  const first = (user?.full_name || user?.email || "").split(/[\s@]/)[0];
  const resume = current || datasets[0];

  return (
    <div className="dash">
      <header className="dash-head">
        <div>
          <h1 className="dash-title">{first ? `Welcome back, ${first}` : "Welcome back"}</h1>
          <p className="dash-sub">Your projects, experiments and reports in one place.</p>
        </div>
        <div className="dash-head-actions">
          <button type="button" className="btn btn-secondary" onClick={() => onGo("biblio")}>
            <Icon name="book" size={16} /> Analyse publications
          </button>
          <button type="button" className="btn btn-primary" onClick={onNew}>
            <Icon name="upload" size={16} /> Upload a dataset
          </button>
        </div>
      </header>

      {err && <div className="error">{err}</div>}

      <section className="dash-stats" aria-label="Workspace totals">
        {(stats || [0, 1, 2, 3].map((i) => [i])).map(([k, v, icon]) => (
          <div className="dash-stat" key={k}>
            {stats ? (
              <>
                <span className="dash-stat-icon"><Icon name={icon} size={18} /></span>
                <span className="dash-stat-v">{v.toLocaleString()}</span>
                <span className="dash-stat-k">{k}</span>
              </>
            ) : <span className="skeleton" style={{ height: 44 }} />}
          </div>
        ))}
      </section>

      {resume && (
        <section className="dash-resume">
          <div className="dash-resume-icon"><Icon name="flask" size={22} /></div>
          <div className="dash-resume-text">
            <p className="dash-kicker">{current ? "Open now" : "Most recent project"}</p>
            <h2>{resume.title}</h2>
            <p className="dash-meta">
              {current
                ? `${current.done} of ${current.total} steps done · next: ${current.next}`
                : `${resume.shape ? `${resume.shape[0].toLocaleString()} rows × ${resume.shape[1]} columns · ` : ""}${stage(resume).label.toLowerCase()} · ${ago(resume.updated)}`}
            </p>
            {current && (
              <div className="dash-resume-bar" aria-hidden="true"><span style={{ width: `${(current.done / current.total) * 100}%` }} /></div>
            )}
          </div>
          <button type="button" className="btn btn-primary" onClick={() => (current ? onContinue() : onOpen(resume))}>
            Continue <Icon name="arrowRight" size={16} />
          </button>
        </section>
      )}

      <div className="dash-grid">
        <section className="dash-card dash-projects">
          <div className="dash-card-head">
            <h2>Recent projects</h2>
            {list?.length > 0 && <button type="button" className="link-btn" onClick={() => onGo("projects")}>View all ({list.length})</button>}
          </div>
          {!list && !err && <div className="dash-list">{[0, 1, 2].map((i) => <span key={i} className="skeleton" style={{ height: 56 }} />)}</div>}
          {list && list.length === 0 && (
            <div className="empty">
              <Icon name="folder" size={28} />
              <p><b>No projects yet.</b> Upload a dataset or start from a sample below.</p>
            </div>
          )}
          {list?.length > 0 && (
            <ul className="dash-list">
              {list.slice(0, RECENT).map((p) => {
                const st = stage(p);
                return (
                  <li key={p.id} className="dash-row">
                    <span className="dash-row-icon"><Icon name={p.kind === "bibliometric" ? "book" : "table"} size={18} /></span>
                    <span className="dash-row-text">
                      <b title={p.title}>{p.title}</b>
                      <span>
                        {p.shape ? `${p.shape[0].toLocaleString()} × ${p.shape[1]}` : p.filename}
                        {p.runs > 0 && ` · ${p.runs} run${p.runs > 1 ? "s" : ""}`} · {ago(p.updated)}
                      </span>
                    </span>
                    <span className={`pill pill-${st.tone}`}>{st.label}</span>
                    <button type="button" className="btn btn-secondary btn-sm" onClick={() => onOpen(p)}>Open</button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="dash-card dash-results">
          <div className="dash-card-head"><h2>Latest results</h2></div>
          {results === null && list?.some((p) => p.runs > 0) && <span className="skeleton" style={{ height: 96 }} />}
          {(results?.length === 0 || (list && !list.some((p) => p.runs > 0))) && (
            <div className="empty">
              <Icon name="trendingUp" size={28} />
              <p>Finished training runs show up here with their best model and test score.</p>
            </div>
          )}
          {results?.length > 0 && (
            <ul className="dash-list">
              {results.map(({ p, run }) => {
                const best = run.models.find((m) => m.key === run.best);
                return (
                  <li key={p.id} className="result-row">
                    <div className="result-main">
                      <b title={p.title}>{p.title}</b>
                      <span>{best ? best.label : run.kind === "cluster" ? "Clustering run" : "No model beat the baseline"} · {ago(run.created)}</span>
                    </div>
                    {best?.test != null && (
                      <div className="result-score">
                        <span className="result-v">{fmt(best.test)}</span>
                        <span className="result-k">{run.primary_label || run.primary_metric} (test)</span>
                      </div>
                    )}
                    <div className="result-actions">
                      <button type="button" className="btn btn-quiet btn-sm" disabled={busy === p.id} onClick={() => report(p)}
                              title="Download the PDF report of this project">
                        <Icon name="download" size={15} /> {busy === p.id ? "Building…" : "PDF report"}
                      </button>
                      <button type="button" className="btn btn-secondary btn-sm" onClick={() => onOpen(p)}>Open</button>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="dash-card dash-tools">
          <div className="dash-card-head"><h2>Analysis tools</h2></div>
          <div className="tool-grid">
            <button type="button" className="tool" onClick={() => onGo("statistics")} disabled={!current}
                    title={current ? undefined : "Open a dataset first"}>
              <Icon name="sigma" size={20} />
              <b>Statistics</b>
              <span>{current ? "Tests, regression and reliability on the open dataset" : "Open a dataset to run tests"}</span>
            </button>
            <button type="button" className="tool" onClick={() => onGo("biblio")}>
              <Icon name="network" size={20} />
              <b>Bibliometrics</b>
              <span>Scopus, Web of Science or PubMed exports</span>
            </button>
            <button type="button" className="tool" onClick={() => onGo("projects")}>
              <Icon name="folder" size={20} />
              <b>Manage projects</b>
              <span>Rename, reopen or delete with everything made from them</span>
            </button>
            {user?.role === "admin" && (
              <button type="button" className="tool" onClick={() => onGo("admin")}>
                <Icon name="users" size={20} />
                <b>Accounts</b>
                <span>Activate or deactivate users</span>
              </button>
            )}
          </div>
        </section>

        {samples?.length > 0 && (
          <section className="dash-card dash-samples">
            <div className="dash-card-head"><h2>Start from a sample</h2></div>
            <p className="dash-card-note">Datasets bundled with scikit-learn: a quick way to try the whole pipeline.</p>
            <div className="sample-list">
              {samples.map((s) => (
                <button type="button" key={s.key} className="sample-btn" onClick={() => onSample(s.key, s.task)}
                        title={`Target: ${s.target}`}>
                  <Icon name={s.task === "regression" ? "trendingUp" : s.task === "clustering" ? "scatter" : "target"} size={17} />
                  <span><b>{s.title}</b><span>{s.task}</span></span>
                </button>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
