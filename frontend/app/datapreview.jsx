"use client";

import { useState } from "react";
import { call } from "./apiclient";

// The data the models learn from: a readable page of preprocessed rows (built on
// request, nothing extra stored) and a per-column summary of what the model
// actually receives. The model matrix itself is never shown as rows.
export default function DataPreview({ dsId }) {
  const [open, setOpen] = useState(false);
  const [split, setSplit] = useState("all");
  const [page, setPage] = useState(null);
  const [err, setErr] = useState("");
  const LIMIT = 50; // our starting point: rows per page

  async function load(offset = 0, which = split) {
    setErr("");
    try { setPage(await call(`/api/train/data?id=${dsId}&split=${which}&offset=${offset}&limit=${LIMIT}`)); }
    catch (e) { setErr(e.message); }
  }

  return (
    <section>
      <p className="sectitle">
        Training data{" "}
        <button className="ghost" onClick={() => { setOpen(!open); if (!open && !page) load(0); }}>{open ? "Hide" : "Show"}</button>
      </p>
      {open && (
        <div className="panel tablescroll" style={{ padding: 12, marginBottom: 12 }}>
          {err && <div className="error">{err}</div>}
          {page && (
            <>
              <p style={{ margin: "0 0 6px" }}>
                What the models receive: <b>{page.summary.model_features.toLocaleString()} features</b> from{" "}
                {page.rows_train.toLocaleString()} training rows ({page.rows_test.toLocaleString()} test rows are locked away for the final score).
              </p>
              <table style={{ marginBottom: 10 }}>
                <thead><tr><th>Column</th><th>Steps</th><th>Model features</th></tr></thead>
                <tbody>
                  {page.summary.columns.map((c) => (
                    <tr key={c.column}><td>{c.column}</td><td>{c.steps.join(" → ")}</td><td>{c.features.toLocaleString()}</td></tr>
                  ))}
                </tbody>
              </table>
              {page.summary.notes.map((n, i) => <p key={i} className="note">{n}</p>)}
              <p className="note">
                Rows below show each column after preprocessing (text stays readable; its word/letter features are counted above, not listed).
              </p>
              <div className="builder-row" style={{ alignItems: "center" }}>
                <label className="pp-field" style={{ maxWidth: 140 }}><span>Rows</span>
                  <select value={split} onChange={(e) => { setSplit(e.target.value); load(0, e.target.value); }}>
                    <option value="all">train + test</option><option value="train">train</option><option value="test">test</option>
                  </select>
                </label>
                <span className="note">
                  {page.total ? page.offset + 1 : 0}–{Math.min(page.offset + page.limit, page.total)} of {page.total.toLocaleString()}
                </span>
                <button className="ghost" disabled={page.offset === 0} onClick={() => load(Math.max(0, page.offset - LIMIT))}>Previous</button>
                <button className="ghost" disabled={page.offset + page.limit >= page.total} onClick={() => load(page.offset + LIMIT)}>Next</button>
              </div>
              <table>
                <thead><tr>{page.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                <tbody>
                  {page.rows.map((r, i) => (
                    <tr key={i}>
                      {page.columns.map((c) => {
                        const v = r[c];
                        const text = v === null ? "" : typeof v === "number" ? String(Number(v.toFixed(4))) : String(v);
                        return <td key={c} title={text} style={{ maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{text}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}
    </section>
  );
}
