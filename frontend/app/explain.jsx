"use client";

// Explainable AI for one run model (plan §12): permutation importance (every model),
// the model's own importance, SHAP (XGBoost / LightGBM / CatBoost, scikit-learn tree
// models, linear models), partial dependence (every model), and one prediction
// explained (SHAP where available + LIME for every model). The global part is computed
// as a background job, then read back; one row is explained on request.
import { useEffect, useRef, useState } from "react";
import { call, fmt } from "./apiclient";
import { BarPlot, COLORS, Figure, LinePlot } from "./chartkit";

const UP = COLORS[5], DOWN = COLORS[0];   // pushes the explained output up / down

// One locked test row: what the model predicted and why (SHAP + LIME).
function RowExplain({ dsId, runId, model, rows }) {
  const [n, setN] = useState(1);
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  useEffect(() => { setOut(null); setErr(""); }, [dsId, runId, model]);
  async function go() {
    setBusy(true); setErr("");
    try { setOut(await call("/api/explain/row", { id: dsId, run: runId, model, row: Number(n) - 1 })); }
    catch (e) { setErr(e.message); }
    setBusy(false);
  }
  const bars = (cols, key) => cols.map((c) => ({ label: c.column, value: c[key], color: c[key] >= 0 ? UP : DOWN }));
  return (
    <div className="panel" style={{ padding: 12, marginTop: 10 }}>
      <b>One prediction explained</b>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
        <label className="note" htmlFor={`row-${model}`}>Locked test row (1–{rows.toLocaleString()})</label>
        <input id={`row-${model}`} type="number" min="1" max={rows} value={n} style={{ width: 90 }}
               onChange={(e) => setN(e.target.value)} />
        <button className="ghost" onClick={go} disabled={busy || !(n >= 1 && n <= rows)}>{busy ? "Explaining…" : "Explain this row"}</button>
      </div>
      {err && <div className="error">{err}</div>}
      {out && (
        <>
          <p className="note" style={{ margin: "8px 0 0" }}>
            Row {out.row + 1}: actual <b>{String(out.actual)}</b>, predicted <b>{typeof out.predicted === "number" ? fmt(out.predicted) : out.predicted}</b>
            {out.probabilities && <> ({Object.entries(out.probabilities).map(([c, p]) => `${c} ${fmt(p, 2)}`).join(", ")})</>}.
            {" "}Red pushes {out.task === "classification" ? `the ${out.lime.units}` : "the prediction"} up, blue pushes it down.
          </p>
          {out.shap && (
            <BarPlot title={`SHAP for this row (${out.shap.units})`} items={bars(out.shap.columns, "value")}
                     xLabel={`contribution (${out.shap.units})`} note={`${out.shap.method}. Exact for this model.`} />
          )}
          <BarPlot title={`LIME for this row (${out.lime.units})`} items={bars(out.lime.columns, "weight")}
                   xLabel="change when the column keeps this row's value instead of a typical one"
                   note={`Local surrogate (LIME, ${out.lime.samples.toLocaleString()} perturbed copies of the row): fits the model here with weighted R² ${fmt(out.lime.score, 2)}${out.lime.score < 0.5 ? " - a loose local fit, read it with care" : ""}. Model output ${fmt(out.lime.model_output)}, surrogate ${fmt(out.lime.local_prediction)}.${out.shap ? "" : " No exact SHAP for this model type (needs the shap package)."}`} />
          <details style={{ marginTop: 6 }}>
            <summary className="note">The row's values</summary>
            <table><tbody>{Object.entries(out.values).map(([c, v]) => <tr key={c}><td>{c}</td><td>{v === null ? "blank" : String(v)}</td></tr>)}</tbody></table>
          </details>
        </>
      )}
    </div>
  );
}

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
      let misses = 0;
      const poll = async () => {
        let s;
        try { s = await call(`/api/run?id=${dsId}&run=${r.job}`); misses = 0; }
        catch (e) {   // one failed status read must not end the watch
          if (++misses <= 3) { timer.current = setTimeout(poll, POLL_MS); return; }
          setState(""); setErr(e.message); return;
        }
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
          {data.pdp?.columns.filter((c) => !c.error).map((c) => (
            <LinePlot key={c.column} title={`Partial dependence — ${c.column}`} xLabel={c.column} yLabel={data.pdp.units}
                      series={c.curves.map((cv) => ({ label: cv.label, x: c.x, y: cv.y }))}
                      note={`The average prediction over ${data.pdp.rows.toLocaleString()} test rows when ${c.column} is set to each value (its ${data.pdp.percentiles[0] * 100}th–${data.pdp.percentiles[1] * 100}th percentile range), the other columns as they are. It shows how the model responds, not cause and effect.`} />
          ))}
          <RowExplain dsId={dsId} runId={runId} model={model} rows={data.test_rows} />
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
