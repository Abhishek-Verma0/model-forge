"use client";

import { useState } from "react";

import { API } from "./apiclient";
const RETYPE_OPTS = ["keep", "number", "datetime", "category"];

// One AI clean op -> plain words, so the user can SEE the suggestion.
function readableClean(o) {
  if (o.op === "trim_whitespace") return "trim whitespace";
  if (o.op === "merge_categories") return "merge inconsistent categories";
  if (o.op === "retype") return `convert to ${o.to || "number"}`;
  if (o.op === "rename_column") return `rename to “${o.to}”`;
  if (o.op === "nullify") return "treat placeholders as missing";
  if (o.op === "map_values") return "relabel values";
  if (o.op === "drop_column") return "drop this column";
  return o.op;
}

// The card's CURRENT plan from its controls -- shown live so the user always
// sees what Run will do. `dropped` comes from the shared dropCols set.
function plannedClean(s, dropped) {
  if (dropped) return "drop this column";
  const parts = [];
  if (s.trim) parts.push("trim whitespace");
  if (s.merge) parts.push("merge inconsistent categories");
  if (s.retype && s.retype !== "keep") parts.push(`convert to ${s.retype}`);
  return parts.length ? parts.join(", ") : "no change";
}

// Columns the audit flagged as needing a cleaning fix, with a suggested default.
function gatherFlags(report) {
  const checks = report.checks;
  const flags = {};
  const add = (col, label, suggest) => {
    flags[col] = flags[col] || { labels: [], suggest: {} };
    if (!flags[col].labels.includes(label)) flags[col].labels.push(label);
    Object.assign(flags[col].suggest, suggest);
  };
  for (const c of checks.consistency_issues.columns) add(c.column, "inconsistent categories", { merge: true });
  for (const c of checks.value_issues.columns) {
    for (const k of Object.keys(c.issues)) {
      if (k === "whitespace_values") add(c.column, "whitespace", { trim: true });
      if (k === "numeric_stored_as_text") add(c.column, "numbers stored as text", { retype: "number" });
    }
  }
  for (const c of checks.constant_columns.columns) add(c.column, "constant", { drop: true });
  for (const c of checks.id_columns.columns) add(c.column, "ID-like", { drop: true });
  for (const c of checks.high_cardinality.columns) add(c.column, "many values (text-encoded in Preprocessing)", {});
  return flags;
}

export default function Clean({ data, target, plan, planLoading, planErr, onCleaned, onApplyAI }) {
  const { profile, report, sample } = data;
  const info = profile.columns_info;
  const flags = gatherFlags(report);
  const hasDupes = report.checks.duplicate_rows.has_duplicates;
  const allCols = profile.column_names;

  const semType = {};
  for (const c of report.checks.column_types.columns) semType[c.column] = c.detected_type;

  // low-cardinality categoricals get a card too (mostly so you can relabel their
  // values), even when the audit didn't flag them.
  const lowcard = allCols.filter(
    (c) => semType[c] !== "numeric" && info[c]?.top_values && info[c].unique_values <= 20
  );

  // AI clean-plan overlay: suggestions per column (visible before you apply) +
  // notes + a global note. drop_column suggestions feed the drop panel.
  const aiByCol = {};
  const aiGlobal = [];
  const aiDrops = [];
  const aiNames = {}; // col -> AI-suggested new name (shown as a hint until applied)
  for (const o of plan || []) {
    if (o.op === "drop_duplicates") { if (o.note) aiGlobal.push(o.note); continue; }
    const col = o.column;
    if (!col) continue;
    aiByCol[col] = aiByCol[col] || { actions: [], notes: [] };
    aiByCol[col].actions.push(readableClean(o));
    if (o.note) aiByCol[col].notes.push(o.note);
    if (o.op === "drop_column") aiDrops.push(col);
    if (o.op === "rename_column" && o.to) aiNames[col] = o.to;
  }
  const shown = Array.from(new Set([...Object.keys(flags), ...Object.keys(aiByCol), ...lowcard]));

  // audit-suggested drops are pre-checked; AI drops are pre-checked too once the
  // plan has loaded. (Both just seed the boxes -- you can uncheck any of them.)
  const seedDrops = new Set([
    ...Object.keys(flags).filter((c) => flags[c].suggest?.drop),
    ...aiDrops,
  ]);

  // per-column fix ops (drop lives separately in dropCols, one source of truth)
  const init = {};
  for (const col of shown) {
    const s = flags[col]?.suggest || {};
    init[col] = { trim: !!s.trim, merge: !!s.merge, retype: s.retype || "keep" };
  }

  const [dedupe, setDedupe] = useState(hasDupes);
  const [nullify, setNullify] = useState(false);
  const [cols, setCols] = useState(init);
  const [dropCols, setDropCols] = useState(seedDrops);
  const [names, setNames] = useState({}); // col -> new name (only when changed from original)
  const [valueMaps, setValueMaps] = useState({}); // col -> { originalValue: newValue }
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [summary, setSummary] = useState(null);

  function set(col, field, value) {
    setCols((c) => ({ ...c, [col]: { ...c[col], [field]: value } }));
  }
  function setName(col, value) {
    setNames((n) => ({ ...n, [col]: value }));
  }
  function setValue(col, orig, value) {
    setValueMaps((m) => ({ ...m, [col]: { ...m[col], [orig]: value } }));
  }
  function toggleDrop(col) {
    setDropCols((s) => { const n = new Set(s); n.has(col) ? n.delete(col) : n.add(col); return n; });
  }
  function applyAI() {
    if (!plan) return;
    const nextCols = { ...cols };
    const nextDrops = new Set(dropCols);
    const nextNames = { ...names };
    for (const o of plan) {
      if (o.op === "drop_duplicates") { setDedupe(true); continue; }
      if (o.op === "nullify") { setNullify(true); continue; }
      const col = o.column;
      if (!col) continue;
      if (o.op === "drop_column") { nextDrops.add(col); continue; }
      if (o.op === "rename_column") { if (o.to) nextNames[col] = o.to; continue; }
      nextCols[col] = { trim: false, merge: false, retype: "keep", ...nextCols[col] };
      if (o.op === "trim_whitespace") nextCols[col].trim = true;
      else if (o.op === "merge_categories") nextCols[col].merge = true;
      else if (o.op === "retype") nextCols[col].retype = o.to || "number";
    }
    setCols(nextCols);
    setDropCols(nextDrops);
    setNames(nextNames);
    onApplyAI?.();
  }

  function buildOps() {
    const ops = [];
    if (dedupe) ops.push({ op: "drop_duplicates" });
    // nullify runs first so the placeholders it clears count as missing downstream.
    if (nullify) for (const col of allCols) if (!dropCols.has(col)) ops.push({ op: "nullify", column: col });
    for (const col of dropCols) ops.push({ op: "drop_column", column: col });
    for (const col of shown) {
      if (dropCols.has(col)) continue; // dropped -> its fix ops are moot
      const s = cols[col] || {};
      if (s.trim) ops.push({ op: "trim_whitespace", column: col });
      if (s.merge) ops.push({ op: "merge_categories", column: col });
      const vm = valueMaps[col];
      if (vm) {
        const mapping = {};
        for (const [k, v] of Object.entries(vm)) if (v !== k && v !== "") mapping[k] = v;
        if (Object.keys(mapping).length) ops.push({ op: "map_values", column: col, mapping });
      }
      if (s.retype && s.retype !== "keep") ops.push({ op: "retype", column: col, to: s.retype });
    }
    // renames LAST -- they change column names, and every op above uses originals.
    for (const col of allCols) {
      if (dropCols.has(col)) continue;
      const nm = (names[col] ?? col).trim();
      if (nm && nm !== col) ops.push({ op: "rename_column", column: col, to: nm });
    }
    return ops;
  }

  async function run() {
    // client-side guard: renames must be non-blank and not collide (backend re-checks).
    const finalNames = allCols.filter((c) => !dropCols.has(c)).map((c) => (names[c] ?? c).trim());
    if (finalNames.some((n) => !n)) { setError("A column name can’t be blank."); return; }
    const dup = finalNames.find((n, i) => finalNames.indexOf(n) !== i);
    if (dup) { setError(`Duplicate column name “${dup}” — names must be unique.`); return; }
    setRunning(true); setError(""); setSummary(null);
    try {
      const res = await fetch(`${API}/api/clean`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: data.id, ops: buildOps(), filename: data.filename }),
      });
      if (!res.ok) {
        let d = `Error ${res.status}`;
        try { const j = await res.json(); if (j.detail) d = j.detail; } catch {}
        throw new Error(d);
      }
      const fresh = await res.json();
      setSummary(fresh.clean_summary);
      onCleaned?.(fresh); // downstream tabs (EDA, Preprocess) now use the cleaned data
    } catch (e) {
      setError(e instanceof TypeError ? `Cannot reach the backend at ${API}.` : e.message);
    } finally { setRunning(false); }
  }

  const previewCols = allCols;

  return (
    <section>
      <p className="sectitle">Clean — fix the data before preprocessing</p>

      {/* AI status */}
      <div className="pp-steps">
        <div className="pp-step">
          <span className="pp-step-n">AI</span>
          <div>
            {planLoading && <p>Getting AI suggestions…</p>}
            {planErr && <p className="pp-muted">AI offline — the cards below use rule-based suggestions.</p>}
            {plan && !planLoading && Object.keys(aiByCol).length > 0 && (
              <p>
                AI suggested changes for {Object.keys(aiByCol).length} column(s).{" "}
                <button className="pp-link" onClick={applyAI}>Apply AI suggestions</button>
              </p>
            )}
            {plan && !planLoading && Object.keys(aiByCol).length === 0 && (
              <p className="pp-muted">{aiGlobal.join(" ") || "The cards below use rule-based suggestions — review and Run."}</p>
            )}
          </div>
        </div>
      </div>

      {/* Data preview grid */}
      {sample && sample.length > 0 && (
        <div className="panel" style={{ padding: 12, marginBottom: 12 }}>
          <p style={{ margin: "0 0 8px", fontSize: 13 }}>Data preview — first {sample.length} rows</p>
          <div className="tablescroll" style={{ maxHeight: 260, overflow: "auto" }}>
            <table>
              <thead><tr>{previewCols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {sample.slice(0, 20).map((r, i) => (
                  <tr key={i}>{previewCols.map((c) => <td key={c}>{r[c] === null || r[c] === undefined ? "—" : String(r[c])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* All columns: rename or drop -- ANY column, not just flagged ones */}
      <div className="panel" style={{ padding: 16, marginBottom: 12 }}>
        <p style={{ margin: "0 0 4px", fontWeight: 600, fontSize: 14 }}>Columns — rename or drop</p>
        <p className="note" style={{ margin: "0 0 10px" }}>
          Edit a name to rename it; check <b>drop</b> to remove it (IDs, notes, constants that won’t help
          predict{target ? ` “${target}”` : " the target"}). {dropCols.size} to drop.
          {" "}<b>suggested</b> = the audit or AI flagged it.
        </p>
        <div style={{ display: "grid", gap: 6, maxHeight: 340, overflow: "auto" }}>
          {allCols.map((c) => {
            const isTarget = c === target;
            const dropped = dropCols.has(c);
            const nm = names[c] ?? c;
            const hint = isTarget
              ? "target — kept"
              : nm !== c
                ? `was “${c}”`
                : aiNames[c] && aiNames[c] !== c
                  ? `AI: “${aiNames[c]}”`
                  : "";
            return (
              <div key={c} style={{ display: "flex", alignItems: "center", gap: 8, opacity: dropped ? 0.5 : 1 }}>
                <input
                  value={nm}
                  disabled={isTarget || dropped}
                  onChange={(e) => setName(c, e.target.value)}
                  style={{ flex: "0 0 220px", padding: "4px 8px", fontSize: 13 }}
                  title={c}
                />
                <span className="note" style={{ flex: 1, minWidth: 0, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {hint}
                </span>
                <label className="pp-check" style={{ margin: 0, flex: "0 0 auto" }}
                       title={isTarget ? "This is your target — it can’t be dropped." : ""}>
                  <input type="checkbox" disabled={isTarget} checked={dropped} onChange={() => toggleDrop(c)} />
                  drop{seedDrops.has(c) && <span className="tag" style={{ marginLeft: 4 }}>suggested</span>}
                </label>
              </div>
            );
          })}
        </div>
      </div>

      {/* Global dedupe + placeholders + Run */}
      <div className="panel" style={{ padding: 16 }}>
        {hasDupes && (
          <label className="pp-check">
            <input type="checkbox" checked={dedupe} onChange={(e) => setDedupe(e.target.checked)} />
            Remove {report.checks.duplicate_rows.summary.duplicate_rows.toLocaleString()} duplicate rows
          </label>
        )}
        <label className="pp-check">
          <input type="checkbox" checked={nullify} onChange={(e) => setNullify(e.target.checked)} />
          Treat common placeholders (N/A, -, ?, none, null, unknown) as missing
        </label>
        <div style={{ marginTop: 10 }}>
          <button className="primary" onClick={run} disabled={running}>
            {running ? "Cleaning…" : "Run clean"}
          </button>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {summary && (
        <div className="panel" style={{ marginTop: 12, padding: 16 }}>
          <b style={{ fontSize: 14 }}>Cleaned. What changed:</b>
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 13.5, lineHeight: 1.7 }}>
            {summary.map((s, i) => (
              <li key={i}>{s.column ? <b>{s.column}: </b> : null}{s.detail}</li>
            ))}
          </ul>
          <p className="note" style={{ marginTop: 8 }}>EDA and Preprocessing now use the cleaned data.</p>
        </div>
      )}

      {/* Per-column cleaning cards -- trim / merge / retype. Drop is in the panel
          above, but the card shows a drop toggle too for convenience. */}
      {shown.length > 0 ? (
        <>
          <p className="pp-cards-title">Columns that need attention · review &amp; adjust</p>
          <div className="pp-cards">
            {shown.map((col) => {
              const numeric = semType[col] === "numeric";
              const s = cols[col] || {};
              const dropped = dropCols.has(col);
              return (
                <div className={"pp-card" + (dropped ? " dropped" : "")} key={col}>
                  <div className="pp-card-head">
                    <span className="pp-card-name" title={col}>{col}</span>
                    <span className="tag">{semType[col]}</span>
                  </div>
                  {flags[col]?.labels?.length > 0 && (
                    <p className="pp-flags">{flags[col].labels.join(" · ")}</p>
                  )}

                  {aiByCol[col] && (
                    <div className="pp-ai">
                      <b>AI suggests:</b> {aiByCol[col].actions.join(", ")}
                      {aiByCol[col].notes.length > 0 && (
                        <span className="pp-ai-note"> — {aiByCol[col].notes.join(" ")}</span>
                      )}
                    </div>
                  )}

                  {!numeric && !dropped && (
                    <>
                      <label className="pp-check">
                        <input type="checkbox" checked={!!s.trim} onChange={(e) => set(col, "trim", e.target.checked)} />
                        Trim whitespace
                      </label>
                      <label className="pp-check">
                        <input type="checkbox" checked={!!s.merge} onChange={(e) => set(col, "merge", e.target.checked)} />
                        Merge inconsistent categories
                      </label>
                      {info[col]?.top_values && (
                        <details className="pp-values">
                          <summary>Relabel values ({info[col].top_values.length})</summary>
                          {info[col].top_values.map((tv) => {
                            const k = String(tv.value);
                            return (
                              <label className="pp-field" key={k}>
                                <span title={k}>{k} · {tv.count}</span>
                                <input value={valueMaps[col]?.[k] ?? k} onChange={(e) => setValue(col, k, e.target.value)} />
                              </label>
                            );
                          })}
                        </details>
                      )}
                    </>
                  )}
                  {!dropped && (
                    <label className="pp-field">
                      <span>Change type</span>
                      <select value={s.retype || "keep"} onChange={(e) => set(col, "retype", e.target.value)}>
                        {RETYPE_OPTS.map((o) => <option key={o} value={o}>{o}</option>)}
                      </select>
                    </label>
                  )}
                  <label className="pp-check">
                    <input type="checkbox" checked={dropped} onChange={() => toggleDrop(col)} />
                    Drop this column
                  </label>

                  <p className="pp-reason"><b>Will apply:</b> {plannedClean(s, dropped)}</p>
                </div>
              );
            })}
          </div>
        </>
      ) : (
        <p className="note" style={{ marginTop: 12 }}>No column-level issues detected. Use the drop panel above to remove any columns you don’t need{hasDupes ? ", or remove duplicates" : ""}.</p>
      )}
    </section>
  );
}
