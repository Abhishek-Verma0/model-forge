"use client";

import { useEffect, useState } from "react";
import { call, fmt } from "./apiclient";

// Run history + comparison. Models from different runs are only compared inside the
// same fingerprint (same cleaning, preprocessing, target, split and folds).
export default function RunHistory({ dsId, metrics, refreshKey }) {
  const [runs, setRuns] = useState([]);
  const [picked, setPicked] = useState([]);
  const [results, setResults] = useState({});
  const [metric, setMetric] = useState("");
  const [err, setErr] = useState("");

  async function load() {
    try {
      const list = (await call(`/api/runs/summary?id=${dsId}`)).filter((r) => r.kind === "train");
      setRuns(list);
      setPicked((p) => p.filter((id) => list.some((r) => r.run_id === id)));
    } catch (e) { setErr(e.message); }
  }
  useEffect(() => { load(); }, [dsId, refreshKey]); // eslint-disable-line react-hooks/exhaustive-deps

  async function toggle(id) {
    const next = picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id];
    setPicked(next);
    if (!results[id]) {
      try {
        const res = await call(`/api/run/results?id=${dsId}&run=${id}`);
        setResults((r) => ({ ...r, [id]: res }));
        if (!metric) setMetric(res.primary_metric);
      } catch (e) { setErr(e.message); }
    }
  }

  async function discard(id) {
    if (!window.confirm("Delete this run and all its trained models? Saved models are kept.")) return;
    try { await call(`/api/run?id=${dsId}&run=${id}`, undefined, "DELETE"); load(); } catch (e) { setErr(e.message); }
  }

  const meta = Object.fromEntries(metrics.map((m) => [m.key, m]));
  const higher = meta[metric]?.higher_is_better ?? true;
  const rows = picked.flatMap((id, n) => Object.entries(results[id]?.models || {}).map(([key, r]) => ({
    id, key, run: n + 1, fingerprint: results[id].fingerprint, ...r,
  })));
  const groups = [...new Set(rows.map((r) => r.fingerprint))];

  return (
    <section style={{ marginTop: 16 }}>
      <p className="sectitle">Run history &amp; comparison</p>
      {err && <div className="error">{err}</div>}
      {!runs.length && <p className="note">No training runs yet.</p>}
      {runs.map((r) => (
        <label key={r.run_id} style={{ display: "block", margin: "4px 0" }}>
          <input type="checkbox" checked={picked.includes(r.run_id)} disabled={r.state !== "done"}
                 onChange={() => toggle(r.run_id)} />{" "}
          {new Date(r.created * 1000).toLocaleString()} · {r.state} · {r.models.filter((m) => !m.baseline).map((m) => m.label).join(", ") || "—"}
          {" "}· {r.size_mb} MB · <span className="note">fingerprint {r.fingerprint || "—"}</span>{" "}
          <button className="ghost" type="button" onClick={() => discard(r.run_id)} disabled={["queued", "running"].includes(r.state)}>Discard</button>
        </label>
      ))}

      {rows.length > 0 && (
        <div className="panel tablescroll" style={{ padding: 12, marginTop: 8 }}>
          <label className="pp-field" style={{ maxWidth: 260 }}><span>Compare by</span>
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              {metrics.map((m) => <option key={m.key} value={m.key}>{m.label}{m.higher_is_better ? "" : " (lower is better)"}</option>)}
            </select>
          </label>
          {groups.map((g, gi) => {
            const inGroup = rows.filter((r) => r.fingerprint === g);
            const mean = (r) => r.cv?.[metric]?.mean;
            const ranked = inGroup.filter((r) => !r.baseline && mean(r) != null)
              .sort((a, b) => (higher ? mean(b) - mean(a) : mean(a) - mean(b)));
            const best = ranked[0];
            return (
              <div key={g || gi} style={{ marginTop: 10 }}>
                <p className="note" style={{ margin: "0 0 4px" }}>
                  {groups.length > 1 ? `Group ${gi + 1} (fingerprint ${g}) — only rows inside a group are comparable` : `All selected runs share fingerprint ${g}`}
                </p>
                <table>
                  <thead><tr><th>Run</th><th>Model</th><th>{meta[metric]?.label} CV mean ± sd</th><th>{meta[metric]?.label} test</th><th>Epochs (trained / best)</th><th>Parameters</th></tr></thead>
                  <tbody>
                    {[...inGroup.filter((r) => r.baseline), ...ranked, ...inGroup.filter((r) => !r.baseline && mean(r) == null)].map((r) => (
                      <tr key={r.id + r.key}>
                        <td>#{r.run}</td>
                        <td>{r.label}{best && best.id === r.id && best.key === r.key && <span className="tag" style={{ marginLeft: 6 }}>Best by {meta[metric]?.label} (CV mean)</span>}</td>
                        {r.error ? <td colSpan={4} className="note">{r.error}</td> : (
                          <>
                            <td>{fmt(r.cv?.[metric]?.mean)} ± {fmt(r.cv?.[metric]?.sd)}</td>
                            <td>{fmt(r.test?.[metric])}</td>
                            <td>{r.epochs ? `${r.epochs.epochs_trained ?? "—"} / ${r.epochs.best_epoch ?? "—"}` : "—"}</td>
                            <td className="note" style={{ maxWidth: 320 }}>{Object.entries(r.params || {}).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(",") : v}`).join(" · ") || "—"}</td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
