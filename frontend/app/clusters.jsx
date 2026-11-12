"use client";

// Clustering (plan §9.3, §11.3): K-Means, hierarchical, DBSCAN, Gaussian mixture on
// the preprocessed rows; scores, elbow, silhouette plot, 2-D view, dendrogram, and the
// rows with their cluster as CSV.
import { useEffect, useRef, useState } from "react";
import { call, download, fmt } from "./apiclient";
import { COLORS, Figure, LinePlot, ScatterPlot } from "./chartkit";
import ParamsForm from "./params";

const ACTIVE = ["queued", "running"];
const POLL_MS = 1500;
const POLL_RETRIES = 3; // our starting point: failed status reads in a row before giving up

function SilhouettePlot({ data, title }) {
  const width = 560, left = 46, rowGap = 6, unit = 0.6;
  const total = data.reduce((n, c) => n + c.values.length, 0);
  const height = 40 + total * unit + rowGap * data.length;
  const lo = Math.min(-0.1, ...data.flatMap((c) => c.values)), hi = 1;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (width - left - 14);
  let y = 10;
  return (
    <Figure title={title} width={width} height={height}
            note="One bar per row, grouped by cluster: near 1 = sits well in its cluster, below 0 = closer to another cluster.">
      {data.map((c, k) => {
        const top = y;
        y += c.values.length * unit + rowGap;
        return (
          <g key={c.cluster}>
            {c.values.map((v, i) => (
              <line key={i} x1={x(0)} x2={x(v)} y1={top + i * unit} y2={top + i * unit} stroke={COLORS[k % COLORS.length]} strokeWidth={unit + 0.2} />
            ))}
            <text x={left - 6} y={top + (c.values.length * unit) / 2 + 3} fontSize="10" textAnchor="end" fill="currentColor">{c.cluster}</text>
          </g>
        );
      })}
      <line x1={x(0)} x2={x(0)} y1={6} y2={height - 26} stroke="currentColor" opacity="0.4" />
      {[lo, 0, 0.5, 1].map((t) => <text key={t} x={x(t)} y={height - 10} fontSize="10" textAnchor="middle" fill="currentColor">{Number(t.toFixed(2))}</text>)}
    </Figure>
  );
}

function Dendrogram({ d }) {
  const xs = d.icoord.flat(), ys = d.dcoord.flat();
  const width = 620, height = 300, left = 46, bottom = 46;
  const xmax = Math.max(...xs), ymax = Math.max(...ys, 1e-9);
  const X = (v) => left + (v / xmax) * (width - left - 10);
  const Y = (v) => height - bottom - (v / ymax) * (height - bottom - 10);
  return (
    <Figure title={`Dendrogram (${d.method} linkage, ${d.rows} sampled rows, last ${d.leaves.length} merges)`} width={width} height={height}
            note="Each join is a merge; its height is the distance at which the two groups merged. Cutting high = few big clusters.">
      {d.icoord.map((ic, i) => (
        <polyline key={i} points={ic.map((v, j) => `${X(v)},${Y(d.dcoord[i][j])}`).join(" ")} fill="none" stroke={COLORS[0]} strokeWidth="1.2" />
      ))}
      {d.leaves.map((l, i) => (
        <text key={i} x={X(5 + 10 * i)} y={height - bottom + 12} fontSize="8" textAnchor="end" fill="currentColor"
              transform={`rotate(-60 ${X(5 + 10 * i)} ${height - bottom + 12})`}>{l}</text>
      ))}
    </Figure>
  );
}

export default function Clusters({ data, opt, onReload, onTrained }) {
  const [picked, setPicked] = useState(opt.models.filter((m) => m.compatible && m.key === "kmeans").map((m) => m.key));
  const [params, setParams] = useState({});
  const [primary, setPrimary] = useState(opt.defaults.primary_metric);
  const [run, setRun] = useState(null);
  const [status, setStatus] = useState(null);
  const [res, setRes] = useState(null);
  const [view, setView] = useState("");
  const [err, setErr] = useState("");
  const [history, setHistory] = useState([]);
  const timer = useRef(null);
  useEffect(() => () => clearTimeout(timer.current), []);
  const loadHistory = () => call(`/api/runs/summary?id=${data.id}`).then((r) => setHistory(r.filter((x) => x.task === "clustering"))).catch(() => {});
  useEffect(() => { loadHistory(); }, [data.id]); // eslint-disable-line react-hooks/exhaustive-deps

  function poll(id, misses = 0) {
    timer.current = setTimeout(async () => {
      try {
        const [s, r] = await Promise.all([call(`/api/run?id=${data.id}&run=${id}`), call(`/api/run/results?id=${data.id}&run=${id}`)]);
        setStatus(s);
        if (r.models) { setRes(r); if (!view && r.best) setView(r.best); }
        if (ACTIVE.includes(s.state)) poll(id);
        else { loadHistory(); if (s.state === "done") onTrained?.(); }
      } catch (e) { if (misses < POLL_RETRIES) poll(id, misses + 1); else setErr(e.message); }
    }, POLL_MS);
  }

  async function start() {
    setErr(""); setRes(null); setView("");
    try {
      const { run: id } = await call("/api/train", { id: data.id, models: picked, primary_metric: primary,
                                                      params: Object.fromEntries(picked.map((k) => [k, params[k] || {}])) });
      setRun(id); setStatus({ state: "queued" }); poll(id);
    } catch (e) { setErr(e.message); }
  }

  const running = status && ACTIVE.includes(status.state);
  const f = opt.facts;
  const meta = Object.fromEntries(opt.metrics.map((m) => [m.key, m]));
  const shownRun = run || history[0]?.run_id;
  const m = res?.models?.[view];

  return (
    <section>
      <p className="sectitle">Clustering — no target, every row gets a group</p>
      <div className="target-note" style={{ marginBottom: 10 }}>
        {f.rows.toLocaleString()} rows · {f.features.toLocaleString()} features{f.sparse ? " · text columns (sparse)" : ""}
        {f.nan ? " · missing values left (add an impute step)" : ""} · {opt.k_note}
      </div>
      {opt.notes.map((n, i) => <p key={i} className="note">{n}</p>)}
      <p className="note">Scores measure how compact and separated the groups are — not whether they mean something for your
        research question. That judgement is yours.</p>
      {opt.elbow && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 12 }}>
          <LinePlot title="Elbow (K-Means inertia by k)" xLabel="k (clusters)" yLabel="inertia" width={520} height={260}
                    series={[{ label: "inertia", x: opt.elbow.k, y: opt.elbow.inertia }]}
                    note="Look for the bend: after it, more clusters barely tighten the groups." />
          <LinePlot title="Silhouette by k" xLabel="k (clusters)" yLabel="silhouette" width={520} height={260}
                    series={[{ label: "silhouette", x: opt.elbow.k, y: opt.elbow.silhouette, color: COLORS[1] }]}
                    note={`Highest at k = ${opt.elbow.best_k} - the default k for K-Means, hierarchical and Gaussian mixture.`} />
          {opt.k_distance && (
            <LinePlot title={`k-distance (distance to the ${opt.k_distance.min_samples}th neighbour)`} xLabel="rows, sorted"
                      yLabel="distance" width={520} height={260} points={false}
                      series={[{ label: "distance", x: opt.k_distance.rank, y: opt.k_distance.distance, color: COLORS[2] }]}
                      note="For DBSCAN: set eps near where this curve bends upward." />
          )}
        </div>
      )}

      <p className="sectitle">Methods</p>
      {opt.models.map((mo) => (
        <div key={mo.key} className="panel" style={{ padding: 12, marginBottom: 8, opacity: mo.compatible ? 1 : 0.55 }}>
          <label style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
            <input type="checkbox" checked={picked.includes(mo.key)} disabled={!mo.compatible || running}
                   onChange={() => setPicked((p) => (p.includes(mo.key) ? p.filter((k) => k !== mo.key) : [...p, mo.key]))} />
            <b>{mo.label}</b><span className="tag">{mo.family}</span>
            {!mo.predicts && <span className="note">labels the rows it was given (can&apos;t place new rows)</span>}
          </label>
          <p className="note" style={{ margin: "4px 0" }}>{mo.about}</p>
          {!mo.compatible && <p className="error" style={{ margin: 0 }}>Can&apos;t run on this data: {mo.reason}</p>}
          {picked.includes(mo.key) && (
            <ParamsForm model={mo} values={params[mo.key] || {}} disabled={running} onChange={(v) => setParams((p) => ({ ...p, [mo.key]: v }))} />
          )}
        </div>
      ))}
      <div className="panel" style={{ padding: 14, marginBottom: 12 }}>
        <div className="builder-row" style={{ alignItems: "flex-end" }}>
          <label className="pp-field" style={{ maxWidth: 240 }}><span>Best by</span>
            <select value={primary} onChange={(e) => setPrimary(e.target.value)}>
              {opt.metrics.map((x) => <option key={x.key} value={x.key}>{x.label}{x.higher_is_better ? "" : " (lower is better)"}</option>)}
            </select>
          </label>
          <button className="primary" disabled={running || !picked.length} onClick={start}>Run clustering</button>
          <button className="ghost" disabled={running} onClick={onReload}>Refresh (after re-preprocessing)</button>
        </div>
        {status && <p className="note">Clustering: {status.state}{running && status.progress?.model ? ` · ${status.progress.model}` : ""}{status.error ? ` · ${status.error}` : ""}</p>}
      </div>
      {err && <div className="error">{err}</div>}

      {res && (
        <div className="panel tablescroll" style={{ padding: 12 }}>
          <table>
            <thead><tr><th>Method</th><th>Clusters</th><th>Noise rows</th>
              {opt.metrics.map((x) => <th key={x.key}>{x.label}{x.higher_is_better ? "" : " ↓"}</th>)}<th>Fit s</th><th /></tr></thead>
            <tbody>
              {Object.entries(res.models).map(([k, r]) => (
                <tr key={k} style={view === k ? { background: "rgba(194,110,56,0.08)" } : undefined}>
                  <td>{r.label}{res.best === k && <span className="tag" style={{ marginLeft: 6 }}>Best by {meta[res.primary_metric]?.label}</span>}</td>
                  {r.error ? <td colSpan={opt.metrics.length + 3} className="note">failed: {r.error}</td> : (
                    <>
                      <td>{r.clusters}</td><td>{r.noise}</td>
                      {opt.metrics.map((x) => <td key={x.key}>{fmt(r.scores?.[x.key], x.key === "calinski_harabasz" ? 1 : 3)}</td>)}
                      <td>{r.fit_seconds}</td>
                      <td><button className="ghost" onClick={() => setView(k)}>Show</button>{" "}
                        <button className="ghost" onClick={() => download(`/api/run/clusters?id=${data.id}&run=${shownRun}&model=${k}`, `clusters_${k}.csv`).catch((e) => setErr(e.message))}>Rows + cluster CSV</button>
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          {Object.values(res.models).some((r) => r.scores_note) && (
            <p className="note">{[...new Set(Object.values(res.models).map((r) => r.scores_note).filter(Boolean))].join(" · ")}</p>
          )}
        </div>
      )}
      {m && !m.error && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 12 }}>
          <ScatterPlot title={`${m.label} — clusters in 2-D (${m.scatter.method})`} xLabel="component 1" yLabel="component 2"
                       points={m.scatter.x.map((x, i) => ({ x, y: m.scatter.y[i], color: m.scatter.label[i] === -1 ? "#bbbbbb" : COLORS[m.scatter.label[i] % COLORS.length] }))}
                       legend={Object.keys(m.sizes).slice(0, 10).map((c) => ({ label: c === "-1" ? `noise (${m.sizes[c]})` : `cluster ${c} (${m.sizes[c]})`,
                                                                              color: c === "-1" ? "#bbbbbb" : COLORS[Number(c) % COLORS.length] }))}
                       note="The 2-D view squeezes all features into two; overlap here does not always mean overlap in the full data." />
          {m.silhouette_plot && <SilhouettePlot data={m.silhouette_plot} title={`${m.label} — silhouette plot`} />}
          {m.dendrogram && <Dendrogram d={m.dendrogram} />}
        </div>
      )}
      {history.length > 0 && (
        <section style={{ marginTop: 14 }}>
          <p className="sectitle">Clustering runs</p>
          {history.map((h) => (
            <div key={h.run_id} className="note" style={{ margin: "3px 0" }}>
              {new Date(h.created * 1000).toLocaleString()} · {h.state} · {h.models.map((x) => x.label).join(", ")}{" "}
              <button className="ghost" disabled={h.state !== "done"} onClick={async () => {
                setRun(h.run_id); setStatus({ state: h.state });
                const r = await call(`/api/run/results?id=${data.id}&run=${h.run_id}`); setRes(r); setView(r.best || Object.keys(r.models)[0]);
              }}>Open</button>
            </div>
          ))}
        </section>
      )}
    </section>
  );
}
