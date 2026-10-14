"use client";

// Learning curve + validation curve of one run model (plan §11.1), computed on the
// run's own folds as background jobs (they retrain the model many times).
import { useEffect, useRef, useState } from "react";
import { call } from "./apiclient";
import { LinePlot } from "./chartkit";

const POLL_MS = 1500;

function useJob(dsId, runId) {
  const [state, setState] = useState("");
  const [err, setErr] = useState("");
  const timer = useRef(null);
  useEffect(() => () => clearTimeout(timer.current), []);
  async function run(path, body, name, onDone) {
    setErr("");
    try {
      const cached = await call(`/api/run/diagnostics?id=${dsId}&run=${runId}&name=${name}`).catch(() => null);
      if (cached) { onDone(cached); return; }
      const { job } = await call(path, body);
      setState("queued");
      const poll = async () => {
        const s = await call(`/api/run?id=${dsId}&run=${job}`);
        if (["queued", "running"].includes(s.state)) { setState(s.state); timer.current = setTimeout(poll, POLL_MS); return; }
        setState("");
        if (s.state === "done") onDone(await call(`/api/run/diagnostics?id=${dsId}&run=${runId}&name=${name}`));
        else setErr(s.error || `stopped: ${s.state}`);
      };
      poll();
    } catch (e) { setErr(e.message); }
  }
  return { state, err, run };
}

export default function Diagnostics({ dsId, runId, model, label, params, onClose }) {
  const [learning, setLearning] = useState(null);
  const [curve, setCurve] = useState(null);
  const [param, setParam] = useState("");
  const lj = useJob(dsId, runId);
  const vj = useJob(dsId, runId);
  const sweepable = params || [];

  const plot = (d, x, xLabel, title, note) => (
    <LinePlot title={title} xLabel={xLabel} yLabel={d.metric_label}
              series={[{ label: "rows trained on", x, y: d.train_mean, sd: d.train_sd },
                       { label: "held-out fold", x, y: d.val_mean, sd: d.val_sd }]} note={note} />
  );

  return (
    <section className="panel" style={{ padding: 14, marginTop: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <b>Learning &amp; validation curves — {label}</b>
        {onClose && <button className="ghost" onClick={onClose}>Close</button>}
      </div>
      <p className="note">Both retrain the model on the run&apos;s folds with its parameters (shaded = ± sd across folds).</p>
      <div className="builder-row" style={{ alignItems: "flex-end" }}>
        <button className="ghost" disabled={!!lj.state}
                onClick={() => lj.run("/api/run/learning", { id: dsId, run: runId, model }, `${model}_learning`, setLearning)}>
          {lj.state ? `Learning curve: ${lj.state}…` : "Learning curve"}
        </button>
        {sweepable.length > 0 && (
          <>
            <label className="pp-field" style={{ maxWidth: 220 }}><span>Parameter to sweep</span>
              <select value={param} onChange={(e) => setParam(e.target.value)}>
                <option value="">choose…</option>
                {sweepable.map((p) => <option key={p.name} value={p.name}>{p.label}</option>)}
              </select>
            </label>
            <button className="ghost" disabled={!param || !!vj.state}
                    onClick={() => vj.run("/api/run/validation", { id: dsId, run: runId, model, param },
                                          `${model}_validation_${param}`, setCurve)}>
              {vj.state ? `Validation curve: ${vj.state}…` : "Validation curve"}
            </button>
          </>
        )}
      </div>
      {(lj.err || vj.err) && <div className="error">{lj.err || vj.err}</div>}
      {learning && plot(learning, learning.train_rows, "training rows", `Learning curve — ${label}`,
                        "A wide gap that stays = overfitting (more data may help); both low = underfitting.")}
      {curve && (typeof curve.values[0] === "number"
        ? plot(curve, curve.values, curve.param, `Validation curve — ${label} · ${curve.param}`,
               "Where the held-out line peaks is the best setting for this parameter (the others held fixed).")
        : (
          <div className="panel tablescroll" style={{ padding: 10, marginTop: 8 }}>
            <b>Validation curve — {curve.param}</b>
            <table><thead><tr><th>{curve.param}</th><th>train</th><th>held-out</th></tr></thead>
              <tbody>{curve.values.map((v, i) => (
                <tr key={String(v)}><td>{String(v)}</td><td>{curve.train_mean[i]?.toFixed(4)}</td><td>{curve.val_mean[i]?.toFixed(4)}</td></tr>))}
              </tbody></table>
          </div>
        ))}
    </section>
  );
}
