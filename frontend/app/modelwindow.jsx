"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { call } from "./apiclient";
import Diagnostics from "./diagnostics";
import EpochChart from "./epochchart";
import Explain from "./explain";
import Icon from "./icons";
import ResultCharts from "./resultcharts";

const TAB_LABEL = { charts: "Charts", explain: "Explain", curves: "Learning curves", epochs: "Epochs" };

// One trained model's results in a modal window. A native <dialog> opened with
// showModal(): Esc closes it, focus stays inside and the page behind is inert. A tab
// mounts on its first visit and then stays mounted (hidden), so an explanation or
// curve job that is still computing keeps going while another tab is shown.
export function ModelWindow({ dsId, runId, model, label, tags, summary, tabs, tab, onTab, sweep, onTune, onAsk, onClose }) {
  const ref = useRef(null);
  const [seen, setSeen] = useState(() => new Set([tab]));
  const [charts, setCharts] = useState(null);
  const [epochs, setEpochs] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    const d = ref.current;
    if (d && !d.open) d.showModal();
    d?.querySelector('[role="tab"][aria-selected="true"]')?.focus(); // not the first button: that is "Tune this model"
  }, []);

  useEffect(() => {
    setSeen((s) => (s.has(tab) ? s : new Set(s).add(tab)));
    const q = `id=${dsId}&run=${runId}&model=${model}`;
    if (tab === "charts" && !charts) call(`/api/run/curves?${q}`).then(setCharts).catch((e) => setErr(e.message));
    if (tab === "epochs" && !epochs) call(`/api/run/epochs?${q}`).then(setEpochs).catch((e) => setErr(e.message));
  }, [tab]); // eslint-disable-line react-hooks/exhaustive-deps -- each tab's data loads once per window

  const close = () => ref.current?.close();
  const then = (fn) => () => { close(); fn(); };
  const panel = (key, body) => seen.has(key) && (
    <div key={key} role="tabpanel" id={`mw-panel-${key}`} aria-labelledby={`mw-tab-${key}`} hidden={tab !== key}>{body}</div>
  );
  const loading = (what) => !err && <p className="note">Loading the {what}…</p>;

  return (
    <dialog ref={ref} className="mw" aria-labelledby="mw-title" onClose={onClose}
            onClick={(e) => { if (e.target === ref.current) close(); }}>
      <div className="mw-inner">
        <header className="mw-head">
          <div className="mw-titles">
            <h2 id="mw-title" className="mw-title">
              {label}
              {tags.map((t) => <span key={t} className="tag">{t}</span>)}
            </h2>
            {summary && <p className="mw-sub">{summary}</p>}
          </div>
          <div className="mw-actions">
            {onTune && <button type="button" className="ghost mw-btn" onClick={then(onTune)}
                               title="Tick only this model and open the hyperparameter search">Tune this model</button>}
            {onAsk && <button type="button" className="ghost mw-btn" onClick={then(onAsk)}>Ask AI</button>}
            <button type="button" className="ghost mw-x" onClick={close} aria-label="Close"><Icon name="x" /></button>
          </div>
        </header>
        <div className="mw-tabs" role="tablist" aria-label="Results for this model">
          {tabs.map((k) => (
            <button key={k} type="button" role="tab" id={`mw-tab-${k}`} aria-controls={`mw-panel-${k}`}
                    aria-selected={tab === k} className="mw-tab" onClick={() => onTab(k)}>{TAB_LABEL[k]}</button>
          ))}
        </div>
        <div className="mw-body">
          {err && <div className="error">{err}</div>}
          {panel("charts", charts ? <ResultCharts data={charts} title={label} /> : loading("test-set charts"))}
          {panel("explain", <Explain dsId={dsId} runId={runId} model={model} label={label} />)}
          {panel("curves", <Diagnostics dsId={dsId} runId={runId} model={model} label={label} params={sweep} />)}
          {panel("epochs", epochs ? <EpochChart data={epochs} title={label} /> : loading("epoch curve"))}
        </div>
      </div>
    </dialog>
  );
}

// "Training finished" card, drawn on <body> so it shows on whichever step is open (the
// Train step stays mounted but hidden). While the browser tab is in the background the
// page title carries the news too, until the tab is shown again.
export function RunNotice({ notice, onView, onClose }) {
  useEffect(() => {
    if (!notice || !document.hidden) return undefined;
    const base = document.title;
    document.title = `${notice.ok ? "✓" : "!"} ${notice.title} · ${base}`;
    const back = () => { if (!document.hidden) document.title = base; };
    document.addEventListener("visibilitychange", back);
    return () => { document.removeEventListener("visibilitychange", back); document.title = base; };
  }, [notice]);

  if (!notice) return null;
  return createPortal(
    <div className={`notice ${notice.ok ? "is-ok" : "is-bad"}`} role={notice.ok ? "status" : "alert"}>
      <span className="notice-icon" aria-hidden="true"><Icon name={notice.ok ? "check" : "alert"} size={16} strokeWidth={2.5} /></span>
      <div className="notice-text">
        <b>{notice.title}</b>
        {notice.body && <p>{notice.body}</p>}
        {notice.ok && onView && <button type="button" className="ghost notice-btn" onClick={onView}>View results</button>}
      </div>
      <button type="button" className="notice-x" onClick={onClose} aria-label="Dismiss"><Icon name="x" size={16} /></button>
    </div>,
    document.body,
  );
}
