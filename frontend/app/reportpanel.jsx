"use client";

// Report tab (plan §15): the PDF report, the Excel tables and the JSON experiment
// record of this project, for a chosen training run.
import { useEffect, useState } from "react";
import { call, download } from "./apiclient";
import Icon from "./icons";

const DONE = ["train", "tune", "cluster"];

export default function ReportPanel({ data, refreshKey, onDownloaded, onEthics }) {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const [stats, setStats] = useState(0);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const [ethics, setEthics] = useState(data.ethics || "");
  const [saved, setSaved] = useState("");

  useEffect(() => {
    call(`/api/runs/summary?id=${data.id}`)
      .then((r) => setRuns(r.filter((x) => DONE.includes(x.kind) && x.state === "done")))
      .catch(() => setRuns([]));
    call(`/api/stats/history?id=${data.id}`).then((h) => setStats(h.length)).catch(() => {});
  }, [data.id, refreshKey]);

  async function get(kind, name) {
    setBusy(kind); setErr("");
    try { await download(`/api/report/${kind}?id=${data.id}${run ? `&run=${run}` : ""}`, name); onDownloaded?.(); }
    catch (e) { setErr(e.message); } finally { setBusy(""); }
  }

  async function saveEthics() {
    setErr(""); setSaved("");
    try {
      const r = await call(`/api/projects/${data.id}`, { ethics }, "PATCH");
      onEthics?.(r.ethics); setSaved(r.ethics ? "Saved. It appears in the PDF report." : "Cleared.");
    } catch (e) { setErr(e.message); }
  }

  const chosen = runs.find((r) => r.run_id === run) || runs[0];
  const items = [
    ["Dataset description and data-quality summary", true],
    ["Cleaning and preprocessing operations", data.can_reset !== undefined],
    ["Exploratory analysis (statistics and charts)", true],
    [chosen ? `Models, metrics and graphs of the run of ${new Date(chosen.created * 1000).toLocaleString()}` : "Models — no finished training run yet", !!chosen],
    [`Statistical tests (${stats} saved)`, stats > 0],
    ["Explainability — after you press Explain on the best model", !!chosen],
    ["Interpretation (labelled automated), limitations, reproducibility details", true],
  ];
  return (
    <section>
      <p className="sectitle">Report</p>
      <div className="panel" style={{ padding: 16 }}>
        <p style={{ marginTop: 0 }}>A research-ready report of this project, built from what you have run — nothing is re-trained.
          Automated interpretations are labelled and need your review before you use them.</p>
        <ul className="check-list">
          {items.map(([t, ok]) => (
            <li key={t} className={ok ? "check-item is-on" : "check-item"}>
              <span className="check-mark" aria-hidden="true">{ok && <Icon name="check" size={12} strokeWidth={3} />}</span>
              <span>{t}</span><span className="sr-only">{ok ? "(included)" : "(not yet)"}</span>
            </li>))}
        </ul>
        {runs.length > 1 && (
          <label className="pp-field" style={{ maxWidth: 420 }}><span>Training run to report</span>
            <select value={run} onChange={(e) => setRun(e.target.value)}>
              <option value="">latest finished run</option>
              {runs.map((r) => <option key={r.run_id} value={r.run_id}>
                {new Date(r.created * 1000).toLocaleString()} · {r.kind} · {r.models.filter((m) => !m.baseline).map((m) => m.label).join(", ")}</option>)}
            </select>
          </label>
        )}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
          <button className="primary" disabled={!!busy} onClick={() => get("pdf", "report.pdf")}>{busy === "pdf" ? "Building PDF…" : "Download PDF report"}</button>
          <button className="ghost" disabled={!!busy} onClick={() => get("excel", "report.xlsx")}>{busy === "excel" ? "Building…" : "Excel tables"}</button>
          <button className="ghost" disabled={!!busy} onClick={() => get("json", "experiment.json")}>{busy === "json" ? "Building…" : "JSON experiment record"}</button>
        </div>
        <p className="note" style={{ marginBottom: 0 }}>
          Graphs: every chart on the EDA, Train and Statistics tabs has its own PNG / SVG download. The trained model file and the
          fitted preprocessing pipeline download from the Train and Preprocessing tabs.
        </p>
        {err && <div className="error" style={{ marginTop: 8 }}>{err}</div>}
      </div>

      <div className="panel ethics-panel">
        <label htmlFor="ethics-note"><b>Ethics approval and consent</b> (plan §24, optional)</label>
        <p className="note" style={{ marginTop: 2 }}>For data about people: the approving committee and reference number, and how
          consent was obtained. It is printed in the report&apos;s dataset description.</p>
        <textarea id="ethics-note" rows={3} maxLength={2000} value={ethics}
                  onChange={(e) => { setEthics(e.target.value); setSaved(""); }}
                  placeholder="e.g. Approved by the Institutional Ethics Committee, ref. IEC/2026/041; written informed consent from all participants." />
        <div className="ethics-actions">
          <button type="button" className="btn btn-secondary btn-sm" disabled={ethics === (data.ethics || "")} onClick={saveEthics}>Save</button>
          {saved && <span className="note" role="status" style={{ margin: 0 }}>{saved}</span>}
        </div>
      </div>
    </section>
  );
}
