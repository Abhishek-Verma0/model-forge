"use client";

// Explainable AI for one run model (plan §12): permutation importance (every model),
// the model's own importance, and SHAP (XGBoost / LightGBM / CatBoost / linear
// models). Computed as a background job, then read back.
import { useEffect, useRef, useState } from "react";
import { call, fmt } from "./apiclient";
import { BarPlot, COLORS, Figure } from "./chartkit";

const POLL_MS = 1500; // our starting point

function Dots({ shap }) {
  const rows = shap.dots;
  const vals = rows.flatMap((r) => r.shap);
  const lim = Math.max(...vals.map(Math.abs), 1e-9);
  const width = 640, label = 170, rowH = 26, top = 10, bottom = 34;
  const height = top + bottom + rows.length * rowH;
  const x = (v) => label + ((v + lim) / (2 * lim)) * (width - label - 14);
  const color = (r) => (r == null || r < 0 ? "#9ca3af" : `rgb(${Math.round(37 + 183 * r)},${Math.round(99 - 61 * r)},${Math.round(235 - 197 * r)})`);
  return (
    <Figure title={`SHAP values per row (${shap.units})`} width={width} height={height}
            note={`Each dot is one test row: right of the centre line pushes the prediction for ${shap.explained} up, left pushes it down. Colour = the row's value of that column (blue low → red high; grey = not a single number).`}>
      <line x1={x(0)} x2={x(0)} y1={top} y2={height - bottom} stroke="currentColor" opacity="0.4" />
      {rows.map((r, i) => (
        <g key={r.column}>
          <text x={label - 6} y={top + i * rowH + rowH / 2 + 3} fontSize="10" textAnchor="end" fill="currentColor">
            {r.column.length > 26 ? r.column.slice(0, 25) + "…" : r.column}
          </text>
          {r.shap.map((v, j) => (
            <circle key={j} cx={x(v)} cy={top + i * rowH + rowH / 2 + (((j * 7919) % 17) - 8) * 0.9} r="2.2"
                    fill={color(r.value_rank ? r.value_rank[j] : null)} opacity="0.75" />
          ))}
        </g>
      ))}
      <text x={x(0)} y={height - 10} fontSize="10" textAnchor="middle" fill="currentColor">0</text>
      <text x={x(-lim)} y={height - 10} fontSize="10" fill="currentColor">{(-lim).toPrecision(2)}</text>
      <text x={x(lim)} y={height - 10} fontSize="10" textAnchor="end" fill="currentColor">{lim.toPrecision(2)}</text>
    </Figure>
  );
}

export default function Explain({ dsId, runId, model, label, onClose }) {
  const [data, setData] = useState(null);
  const [state, setState] = useState("");
  const [err, setErr] = useState("");
  const timer = useRef(null);

  async function read() {
    try { setData(await call(`/api/explain?id=${dsId}&run=${runId}&model=${model}`)); setState("done"); return true; }
    catch { return false; }
  }

  async function start(again = false) {
    setErr(""); setData(null);
    try {
      const r = await call("/api/explain/start", { id: dsId, run: runId, model, again });
      if (r.cached) { await read(); return; }
      setState("running");
      const poll = async () => {
        const s = await call(`/api/run?id=${dsId}&run=${r.job}`);
        if (["queued", "running"].includes(s.state)) { setState(`${s.state}${s.progress?.step ? ` · ${s.progress.step}` : ""}`); timer.current = setTimeout(poll, POLL_MS); }
        else if (s.state === "done") await read();
        else { setState(""); setErr(s.error || `explanation ${s.state}`); }
      };
      poll();
    } catch (e) { setErr(e.message); }
  }

  useEffect(() => { read().then((ok) => { if (!ok) start(); }); return () => clearTimeout(timer.current); }, [dsId, runId, model]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="panel" style={{ padding: 14, marginTop: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <b>Why does {label} predict what it does?</b>
        <span>
          {data && <button className="ghost" onClick={() => start(true)}>Recompute</button>}{" "}
          {onClose && <button className="ghost" onClick={onClose}>Close</button>}
        </span>
      </div>
      {state && state !== "done" && <p className="note">Computing on the locked test rows… {state}</p>}
      {err && <div className="error">{err}</div>}
      {data && (
        <>
          <p className="note" style={{ margin: "6px 0" }}>
            Measured on {data.rows.toLocaleString()} locked test row(s){data.rows < data.test_rows ? ` (a sample of ${data.test_rows.toLocaleString()})` : ""}.
            Explanations show what the model relies on; they do not prove causation.
          </p>
          <BarPlot title={`Permutation importance — drop in ${data.permutation.metric_label} when the column is shuffled`}
                   items={data.permutation.columns.slice(0, 20).map((c) => ({ label: c.column, value: c.mean, error: c.sd }))}
                   xLabel={`${data.permutation.metric_label} drop (± sd over ${data.permutation.repeats} repeats)`}
                   note="Works for every model and speaks in your original columns. Near 0 or negative = the model doesn't need that column." />
          {data.shap ? (
            <>
              <BarPlot title={`SHAP — mean |contribution| (${data.shap.method})`} color={COLORS[3]}
                       items={data.shap.columns.slice(0, 20).map((c) => ({ label: c.column, value: c.mean_abs }))}
                       xLabel={`mean |SHAP| (${data.shap.units})`} />
              <Dots shap={data.shap} />
            </>
          ) : <p className="note">{data.shap_note}</p>}
          {data.native && (
            <BarPlot title="The model's own importance" color={COLORS[1]}
                     items={data.native.columns.slice(0, 20).map((c) => ({ label: c.column, value: c.share }))}
                     xLabel="share of total" note={data.native.kind} />
          )}
          <details style={{ marginTop: 6 }}>
            <summary className="note">Permutation numbers</summary>
            <table><thead><tr><th>Column</th><th>Importance</th><th>sd</th></tr></thead>
              <tbody>{data.permutation.columns.map((c) => <tr key={c.column}><td>{c.column}</td><td>{fmt(c.mean, 4)}</td><td>{fmt(c.sd, 4)}</td></tr>)}</tbody>
            </table>
          </details>
        </>
      )}
    </section>
  );
}
