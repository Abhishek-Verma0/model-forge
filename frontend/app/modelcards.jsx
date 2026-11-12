"use client";

import { useEffect, useState } from "react";
import { dur } from "./apiclient";
import Icon from "./icons";
import ParamsForm from "./params";

const SHOW_AFTER_S = 0.3; // our starting point: loads faster than this show no panel (no flash)
// Families listed in their own group under the other models: [heading, note]
const GROUPS = {
  "Deep learning": ["Deep learning", "PyTorch networks (plan §9.4), trained epoch by epoch with live loss and accuracy " +
    "curves. On ordinary tables gradient-boosted trees usually win - the comparison table shows it on your data."],
  "Tested pipeline": ["Tested pipelines", "A fixed preprocessing step + a model, refitted inside every fold. Only these " +
    "pairs are offered, so no untested combination runs."],   // plan §9.6 (training/presets.py)
};

// Seconds since `on` became true, refreshed 4x a second; 0 while off.
function useSeconds(on) {
  const [s, setS] = useState(0);
  useEffect(() => {
    if (!on) return undefined;
    const t0 = Date.now();
    setS(0);
    const id = setInterval(() => setS((Date.now() - t0) / 1000), 250);
    return () => clearInterval(id);
  }, [on]);
  return s;
}

// Shown while the Train tab waits for /api/train/options. The data read and the model
// list come back as one answer, so they are one step; the AI step starts after it.
export function Preparing({ withAi }) {
  const s = useSeconds(true);
  if (s < SHOW_AFTER_S) return null;
  return (
    <div className="panel prep" role="status">
      <div className="prep-head">
        <b>Getting training ready</b>
        <span className="prep-time" aria-hidden="true"><Icon name="clock" size={14} /> {Math.floor(s)} s</span>
      </div>
      <ol className="prep-steps">
        <li className="is-active"><span className="prep-dot" aria-hidden="true" />Reading your training data and preparing the model list</li>
        {withAi && <li><span className="prep-dot" aria-hidden="true" />Ask the AI which models suit this data</li>}
      </ol>
      <div className="prep-bar" aria-hidden="true"><i /></div>
      <p className="note">The first load can take longer while the model libraries start up.</p>
    </div>
  );
}

// Model picker: every model as a small tile (suggested ones first), and a full card with
// parameters only for the models ticked for training. Everything shown comes from the
// backend model registry (description, parameters, compatibility); suggestions only
// reorder and mark.
export default function ModelCards({ models, suggestion, picked, onToggle, params, onParams, running, estimates }) {
  const waiting = useSeconds(!suggestion);
  const why = Object.fromEntries((suggestion?.models || []).map((m) => [m.key, m.why]));
  const suggestedKeys = (suggestion?.models || []).map((m) => m.key);
  const cards = models.filter((m) => !m.baseline);
  const suggested = suggestedKeys.map((k) => cards.find((m) => m.key === k)).filter(Boolean);
  const others = cards.filter((m) => !suggestedKeys.includes(m.key));
  const selected = [...suggested, ...others].filter((m) => picked.includes(m.key));
  const rules = suggestion?.source === "rules";
  const mark = rules ? "rule suggested" : "AI suggested";

  const estimate = (key) => {
    const e = estimates[key];
    if (!e) return null;
    return e.error ? `estimate failed: ${e.error}` : `~${dur(e.projected_seconds)}`;
  };

  const tile = (m) => {
    const on = picked.includes(m.key);
    const est = estimate(m.key);
    return (
      <button key={m.key} type="button" className={`mp-tile${on ? " is-on" : ""}`} aria-pressed={on}
              disabled={!m.compatible || running} onClick={() => onToggle(m.key)}
              title={m.compatible ? `Works well when: ${m.about.good}` : undefined}>
        <span className="mp-name">
          <span className="mp-check" aria-hidden="true">{on && <Icon name="check" size={12} strokeWidth={3} />}</span>
          {m.label}
        </span>
        <span className="mp-meta">
          {m.family}{m.device === "gpu" ? " · GPU" : ""}{m.epochs ? " · epochs" : ""}{est ? ` · ${est}` : ""}
        </span>
        {!m.compatible && <span className="mp-why">Can&apos;t train on this data: {m.reason}</span>}
      </button>
    );
  };

  return (
    <section className="mp">
      <p className="sectitle">Models</p>
      <div className="panel mp-pick">
        <div className="mp-head">
          <h3 className="mp-title">Suggested for this data</h3>
          {suggestion?.source === "ai" && <span className="tag">AI suggested</span>}
          {suggestion?.source === "ai_cached" && <span className="tag">AI suggested · saved for this data</span>}
          {rules && <span className="tag">Rule-based</span>}
        </div>
        {!suggestion && (
          <>
            <p className="mp-wait" role="status">
              <span className="mp-spin" aria-hidden="true" />
              Asking the AI which models suit this data…
              <span className="prep-time" aria-hidden="true">{Math.floor(waiting)} s</span>
            </p>
            <div className="mp-grid" aria-hidden="true">
              {[0, 1, 2, 3].map((i) => <span key={i} className="skeleton mp-ghost" />)}
            </div>
          </>
        )}
        {rules && <p className="note mp-sub">The AI couldn&apos;t answer, so these come from fixed rules ({suggestion.reason}).</p>}
        {suggestion && (suggested.length
          ? <div className="mp-grid">{suggested.map(tile)}</div>
          : <p className="note mp-sub">No suggestions for this data. Pick models from the list below.</p>)}

        <div className="mp-head mp-head-gap">
          <h3 className="mp-title">{suggested.length ? "Other models" : "All models"}</h3>
        </div>
        <div className="mp-grid">{others.filter((m) => !GROUPS[m.family]).map(tile)}</div>
        {Object.entries(GROUPS).map(([family, [title, note]]) => others.some((m) => m.family === family) && (
          <div key={family}>
            <div className="mp-head mp-head-gap">
              <h3 className="mp-title">{title}</h3>
              <span className="note mp-sub">{note}</span>
            </div>
            <div className="mp-grid">{others.filter((m) => m.family === family).map(tile)}</div>
          </div>
        ))}
      </div>

      <div className="mp-head mp-head-gap">
        <h3 className="mp-title">Selected for training ({selected.length})</h3>
        <span className="note mp-sub">Baseline is always trained as a reference.</span>
      </div>
      {!selected.length && (
        <p className="mp-empty">Pick at least one model above. Its settings will open here.</p>
      )}
      {selected.map((m) => {
        const est = estimate(m.key);
        return (
          <article key={m.key} className="panel mp-card">
            <header className="mp-card-head">
              <div className="mp-card-title">
                <b>{m.label}</b>
                <span className="tag">{m.family}</span>
                {suggestedKeys.includes(m.key) && <span className="tag">{mark}</span>}
                {m.epochs && <span className="tag">epochs</span>}
                {m.device === "gpu" && <span className="tag">GPU</span>}
                {est && <span className="note mp-sub">{est}</span>}
              </div>
              <button type="button" className="ghost mp-remove" disabled={running} onClick={() => onToggle(m.key)}
                      aria-label={`Remove ${m.label} from training`} title="Remove from training">
                <Icon name="x" size={16} />
              </button>
            </header>
            {!m.compatible && <p className="error mp-card-error">Can&apos;t train on this data: {m.reason}</p>}
            {why[m.key] && (
              <p className="note mp-sub">
                {rules ? "Why" : "AI's reason (its opinion, not a measured fact - check “About this model”)"}: {why[m.key]}
              </p>
            )}
            <div className="mp-params">
              <p className="mp-label">Parameters</p>
              <ParamsForm model={m} values={params[m.key] || {}} disabled={running} onChange={(v) => onParams(m.key, v)} />
            </div>
            <details className="mp-about">
              <summary>About this model</summary>
              <dl>
                <dt>What it is</dt><dd>{m.about.what}</dd>
                <dt>How it learns</dt><dd>{m.about.how}</dd>
                <dt>Works well when</dt><dd>{m.about.good}</dd>
                <dt>Watch out</dt><dd>{m.about.watch}</dd>
              </dl>
            </details>
          </article>
        );
      })}
    </section>
  );
}
