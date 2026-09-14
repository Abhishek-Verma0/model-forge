"use client";

import { useEffect, useState } from "react";
import { API, call, fmt } from "./apiclient";

// Test a trained model (from a run, or saved) on an uploaded file or on the run's
// locked test set; save run models under a name; manage saved models.
export default function ModelTester({ dsId, metrics, refreshKey }) {
  const [runs, setRuns] = useState([]);
  const [saved, setSaved] = useState([]);
  const [source, setSource] = useState("");   // "run:<run>:<model>" | "saved:<id>"
  const [file, setFile] = useState(null);
  const [pred, setPred] = useState(null);
  const [page, setPage] = useState(null);
  const [mistakes, setMistakes] = useState(false);
  const [offset, setOffset] = useState(0);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const [mode, setMode] = useState("form");      // "form" | "file"
  const [schema, setSchema] = useState(null);    // {schema, target, classes, ...} of the chosen model
  const [values, setValues] = useState({});
  const [rowResult, setRowResult] = useState(null);

  async function load() {
    try {
      const [r, s] = await Promise.all([call(`/api/runs/summary?id=${dsId}`), call(`/api/models?id=${dsId}`)]);
      setRuns(r.filter((x) => x.kind === "train" && x.state === "done"));
      setSaved(s);
    } catch (e) { setErr(e.message); }
  }
  useEffect(() => { load(); }, [dsId, refreshKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const [kind, a, b] = source.split(":");

  useEffect(() => {
    setSchema(null); setValues({}); setRowResult(null);
    if (!source) return;
    const q = kind === "saved" ? `saved=${a}` : `run=${a}&model=${b}`;
    call(`/api/predict/schema?id=${dsId}&${q}`).then(setSchema).catch((e) => setErr(e.message));
  }, [source]); // eslint-disable-line react-hooks/exhaustive-deps

  async function predictRow() {
    setBusy(true); setErr(""); setRowResult(null);
    try { setRowResult(await call("/api/predict/row", { id: dsId, ...fields, values })); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  }
  const fields = kind === "saved" ? { saved: a } : kind === "run" ? { run: a, model: b } : null;
  const label = (m) => metrics.find((x) => x.key === m)?.label || m;

  function form(fmtName) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("id", dsId);
    fd.append("format", fmtName);
    Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
    return fd;
  }

  async function predict() {
    setBusy(true); setErr(""); setPred(null);
    try { setPred(await call("/api/predict", form("json"))); } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  async function downloadCsv() {
    setErr("");
    try {
      const res = await fetch(`${API}/api/predict`, { method: "POST", body: form("csv") });
      if (!res.ok) throw new Error((await res.json()).detail || `Error ${res.status}`);
      const url = URL.createObjectURL(await res.blob());
      Object.assign(document.createElement("a"), { href: url, download: "predictions.csv" }).click();
      URL.revokeObjectURL(url);
    } catch (e) { setErr(e.message); }
  }

  async function loadTestSet(off = 0, onlyMistakes = mistakes) {
    setErr("");
    try {
      setPage(await call(`/api/run/testset?id=${dsId}&run=${a}&model=${b}&mistakes=${onlyMistakes}&offset=${off}&limit=50`));
      setOffset(off);
    } catch (e) { setErr(e.message); }
  }

  async function save() {
    setErr(""); setMsg("");
    try {
      const m = await call("/api/models/save", { id: dsId, run: a, model: b, name });
      setMsg(`Saved as "${m.name}".`); setName(""); load();
    } catch (e) { setErr(e.message); }
  }

  async function remove(id) {
    if (!window.confirm("Delete this saved model?")) return;
    try { await call(`/api/models?id=${dsId}&model=${id}`, undefined, "DELETE"); if (source === `saved:${id}`) setSource(""); load(); }
    catch (e) { setErr(e.message); }
  }

  return (
    <section style={{ marginTop: 16 }}>
      <p className="sectitle">Test &amp; save a model</p>
      {err && <div className="error">{err}</div>}
      {msg && <p className="note">{msg}</p>}
      <div className="panel" style={{ padding: 12 }}>
        <label className="pp-field" style={{ maxWidth: 520 }}><span>Model</span>
          <select value={source} onChange={(e) => { setSource(e.target.value); setPred(null); setPage(null); }}>
            <option value="">Choose a model…</option>
            {runs.map((r) => (
              <optgroup key={r.run_id} label={`Run ${new Date(r.created * 1000).toLocaleString()}`}>
                {r.models.filter((m) => !m.error).map((m) => (
                  <option key={m.key} value={`run:${r.run_id}:${m.key}`}>
                    {m.label}{r.best === m.key ? " (best)" : ""} — {label(r.primary_metric)} test {fmt(m.test)}
                  </option>
                ))}
              </optgroup>
            ))}
            {saved.length > 0 && (
              <optgroup label="Saved models">
                {saved.map((m) => <option key={m.model_id} value={`saved:${m.model_id}`}>{m.name} ({m.label})</option>)}
              </optgroup>
            )}
          </select>
        </label>

        {kind === "run" && (
          <div className="builder-row" style={{ marginTop: 8 }}>
            <label className="pp-field"><span>Save this model as</span>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="name" />
            </label>
            <button className="ghost" disabled={!name.trim()} onClick={save}>Save model</button>
          </div>
        )}

        {fields && (
          <div className="tabs" style={{ marginTop: 12 }}>
            <button className={mode === "form" ? "tab on" : "tab"} onClick={() => setMode("form")}>Enter values</button>
            <button className={mode === "file" ? "tab on" : "tab"} onClick={() => setMode("file")}>Upload file</button>
          </div>
        )}

        {fields && mode === "form" && schema && (
          <div style={{ marginTop: 8 }}>
            {!schema.schema && <p className="note">This model was trained before input forms existed — use Upload file.</p>}
            {schema.schema && (
              <>
                <p className="note">Fields come from the model's own saved input schema (the values it was trained on).</p>
                {schema.schema.map((f) => (
                  <label key={f.name} className="pp-field" style={{ display: "block", margin: "6px 0" }}>
                    <span>{f.name}{f.missing_ok ? " (optional)" : ""}
                      {f.kind === "number" && f.min != null && <span className="note"> — trained on {f.min} to {f.max}</span>}
                    </span>
                    {f.kind === "text" && (
                      <textarea rows={2} style={{ width: "100%" }} value={values[f.name] ?? ""}
                                onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.value }))} />
                    )}
                    {f.kind === "number" && (
                      <input type="number" step={f.integer ? 1 : "any"} value={values[f.name] ?? ""}
                             onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.value === "" ? "" : Number(e.target.value) }))} />
                    )}
                    {f.kind === "category" && (
                      <select value={values[f.name] ?? ""} onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.value }))}>
                        <option value="">{f.missing_ok ? "(leave empty)" : "choose…"}</option>
                        {f.choices.map((c) => <option key={c} value={c}>{c}</option>)}
                      </select>
                    )}
                    {f.kind === "boolean" && (
                      <input type="checkbox" checked={!!values[f.name]}
                             onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.checked }))} />
                    )}
                  </label>
                ))}
                <button className="primary" disabled={busy} onClick={predictRow}>{busy ? "Predicting…" : "Predict"}</button>
                {rowResult && (
                  <div className="panel" style={{ padding: 10, marginTop: 8 }}>
                    <p style={{ margin: 0, fontSize: 16 }}>Prediction: <b>{String(rowResult.preview[0].prediction)}</b></p>
                    {rowResult.columns.filter((c) => c.startsWith("probability ")).map((c) => (
                      <p key={c} className="note" style={{ margin: 0 }}>{c}: {fmt(rowResult.preview[0][c], 4)}</p>
                    ))}
                    {rowResult.notes.map((n, i) => <p key={i} className="note" style={{ margin: 0 }}>{n}</p>)}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {fields && mode === "file" && (
          <div style={{ marginTop: 12 }}>
            <p style={{ margin: "0 0 4px", fontWeight: 600 }}>Predict on a file</p>
            <p className="note" style={{ margin: "0 0 6px" }}>CSV or Excel with the same columns as the original upload. Include the target column to also get metrics.</p>
            <input type="file" accept=".csv,.xlsx,.xls" onChange={(e) => { setFile(e.target.files?.[0] || null); setPred(null); }} />{" "}
            <button className="primary" disabled={!file || busy} onClick={predict}>{busy ? "Predicting…" : "Predict"}</button>{" "}
            {pred && <button className="ghost" onClick={downloadCsv}>Download all predictions (CSV)</button>}
          </div>
        )}

        {pred && mode === "file" && (
          <div className="tablescroll" style={{ marginTop: 10 }}>
            <p className="note">
              {pred.rows} row(s) predicted{pred.metrics ? ` · ${pred.scored_rows} scored against ${pred.model.target}` : " · no target column, so no metrics"}
              {pred.rows > pred.preview.length && ` · showing first ${pred.preview.length}`}
            </p>
            {pred.notes.map((n, i) => <p key={i} className="note">{n}</p>)}
            {pred.metrics && (
              <p>{Object.entries(pred.metrics).map(([k, v]) => `${label(k)} ${fmt(v)}`).join(" · ")}</p>
            )}
            <Grid columns={pred.columns} rows={pred.preview.slice(0, 50)} />
          </div>
        )}

        {kind === "run" && (
          <div style={{ marginTop: 12 }}>
            <p style={{ margin: "0 0 4px", fontWeight: 600 }}>Locked test set</p>
            <label><input type="checkbox" checked={mistakes} onChange={(e) => { setMistakes(e.target.checked); loadTestSet(0, e.target.checked); }} /> mistakes only</label>{" "}
            <button className="ghost" onClick={() => loadTestSet(0)}>Show predictions</button>
            {page && (
              <div className="tablescroll" style={{ marginTop: 8 }}>
                <p className="note">
                  Rows {page.total ? offset + 1 : 0}–{Math.min(offset + page.limit, page.total)} of {page.total}{" "}
                  <button className="ghost" disabled={offset === 0} onClick={() => loadTestSet(Math.max(0, offset - page.limit))}>Previous</button>{" "}
                  <button className="ghost" disabled={offset + page.limit >= page.total} onClick={() => loadTestSet(offset + page.limit)}>Next</button>
                </p>
                <Grid columns={page.columns} rows={page.rows} />
              </div>
            )}
          </div>
        )}
      </div>

      {saved.length > 0 && (
        <div className="panel tablescroll" style={{ padding: 12, marginTop: 8 }}>
          <p style={{ margin: "0 0 6px", fontWeight: 600 }}>Saved models</p>
          <table>
            <thead><tr><th>Name</th><th>Model</th><th>Saved</th><th>Test score</th><th>Size</th><th></th></tr></thead>
            <tbody>
              {saved.map((m) => (
                <tr key={m.model_id}>
                  <td>{m.name}</td><td>{m.label}</td><td>{new Date(m.created * 1000).toLocaleString()}</td>
                  <td>{label(m.primary_metric)} {fmt(m.test?.[m.primary_metric])}</td><td>{m.size_mb} MB</td>
                  <td>
                    <a className="ghost" href={`${API}/api/models/download?id=${dsId}&model=${m.model_id}`}>Download</a>{" "}
                    <button className="ghost" onClick={() => remove(m.model_id)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Grid({ columns, rows }) {
  const show = (v) => (v === null || v === undefined ? "" : typeof v === "number" ? Number(v.toFixed(4)) : String(v));
  return (
    <table>
      <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} style={r.correct === false ? { background: "rgba(220,38,38,0.08)" } : undefined}>
            {columns.map((c) => <td key={c} style={{ maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={show(r[c])}>{show(r[c])}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
