"use client";

import { dur } from "./apiclient";
import ParamsForm from "./params";

// One card per model, one per row. Everything shown comes from the backend model
// registry (description, parameters, compatibility); suggestions only reorder and mark.
// Order: suggested models first (in suggestion order), then the registry order.
export default function ModelCards({ models, suggestion, picked, onToggle, params, onParams, running, estimates }) {
  const why = Object.fromEntries((suggestion?.models || []).map((m) => [m.key, m.why]));
  const suggestedKeys = (suggestion?.models || []).map((m) => m.key);
  const cards = models.filter((m) => !m.baseline);
  const ordered = [
    ...suggestedKeys.map((k) => cards.find((m) => m.key === k)).filter(Boolean),
    ...cards.filter((m) => !suggestedKeys.includes(m.key)),
  ];
  const mark = suggestion?.source === "rules" ? "rule suggested" : "AI suggested";

  return (
    <section>
      <p className="sectitle">Models</p>
      {!suggestion && <p className="note">Asking the AI which models suit this data… (all models are listed below)</p>}
      {suggestion?.source === "ai_cached" && <p className="note">AI suggestions (saved earlier for this exact data and model list).</p>}
      {suggestion?.source === "rules" && <p className="note">Rule-based suggestions — {suggestion.reason}.</p>}
      <p className="note">Baseline is always trained as a reference and is not listed.</p>

      {ordered.map((m) => {
        const selected = picked.includes(m.key);
        const isSuggested = suggestedKeys.includes(m.key);
        return (
          <div key={m.key} className="panel" aria-disabled={!m.compatible}
               style={{ padding: 14, marginBottom: 10, opacity: m.compatible ? 1 : 0.55,
                        borderColor: selected ? "var(--accent, #c2410c)" : undefined, borderWidth: selected ? 2 : undefined }}>
            <label style={{ display: "flex", gap: 10, alignItems: "baseline", cursor: m.compatible && !running ? "pointer" : "default" }}>
              <input type="checkbox" checked={selected} disabled={!m.compatible || running} onChange={() => onToggle(m.key)} />
              <span style={{ fontWeight: 600, fontSize: 15 }}>{m.label}</span>
              <span className="tag">{m.family}</span>
              {isSuggested && <span className="tag">{mark}</span>}
              {m.epochs && <span className="tag">epochs</span>}
              {m.device === "gpu" && <span className="tag">GPU</span>}
              {estimates[m.key] && (
                <span className="note">
                  {estimates[m.key].error ? `estimate failed: ${estimates[m.key].error}` : `~${dur(estimates[m.key].projected_seconds)}`}
                </span>
              )}
            </label>
            {isSuggested && why[m.key] && (
              <p className="note" style={{ margin: "6px 0 0" }}>
                {suggestion.source === "rules" ? "Why" : "AI's reason (its opinion, not a measured fact - check the card below)"}: {why[m.key]}
              </p>
            )}
            {!m.compatible && <p className="error" style={{ margin: "6px 0 0" }}>Can't train on this data: {m.reason}</p>}

            <dl style={{ display: "grid", gridTemplateColumns: "max-content 1fr", gap: "4px 12px", margin: "10px 0 0" }}>
              <dt className="note">What it is</dt><dd style={{ margin: 0 }}>{m.about.what}</dd>
              <dt className="note">How it learns</dt><dd style={{ margin: 0 }}>{m.about.how}</dd>
              <dt className="note">Works well when</dt><dd style={{ margin: 0 }}>{m.about.good}</dd>
              <dt className="note">Watch out</dt><dd style={{ margin: 0 }}>{m.about.watch}</dd>
            </dl>

            {selected && (
              <div style={{ marginTop: 10 }}>
                <p style={{ margin: "0 0 4px", fontWeight: 600 }}>Parameters</p>
                <ParamsForm model={m} values={params[m.key] || {}} disabled={running} onChange={(v) => onParams(m.key, v)} />
              </div>
            )}
          </div>
        );
      })}
    </section>
  );
}
