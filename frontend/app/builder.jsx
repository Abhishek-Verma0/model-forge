"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const KINDS = ["histogram", "density", "box", "violin", "scatter", "line", "bar", "pie", "tsne"];
const NEEDS_Y = ["scatter", "line"];   // plotted against a second column

// User-driven charts: pick a type + column(s) + optional color/group, and the
// backend renders that exact chart (matplotlib/seaborn) as a PNG we <img>.
export default function ChartBuilder({ data }) {
  const cols = data.profile.column_names;
  const id = data.id;
  const [kind, setKind] = useState("histogram");
  const [x, setX] = useState(cols[0] || "");
  const [y, setY] = useState(cols[1] || cols[0] || "");
  const [hue, setHue] = useState("");
  const [err, setErr] = useState(false);

  if (!id) return null; // backend hasn't added the dataset store / id yet

  const needsY = NEEDS_Y.includes(kind);
  const params = new URLSearchParams({ id, kind, x });
  if (needsY) params.set("y", y);
  if (hue) params.set("hue", hue);
  const src = `${API}/api/chart?${params.toString()}`;

  async function download(fmt) {
    const res = await fetch(`${src}&fmt=${fmt}`);
    if (!res.ok) return;
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = `${kind}_${x}${needsY ? "_" + y : ""}.${fmt}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section>
      <p className="sectitle">Build a chart</p>
      <div className="panel" style={{ padding: 16 }}>
        <div className="builder-row">
          <Field label="Chart">
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              {KINDS.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </Field>
          <Field label={needsY ? "X" : "Column"}>
            <select value={x} onChange={(e) => setX(e.target.value)}>
              {cols.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </Field>
          {needsY && (
            <Field label="Y">
              <select value={y} onChange={(e) => setY(e.target.value)}>
                {cols.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </Field>
          )}
          <Field label="Color / group by">
            <select value={hue} onChange={(e) => setHue(e.target.value)}>
              <option value="">(none)</option>
              {cols.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </Field>
          <button className="ghost" onClick={() => download("png")}>Download PNG</button>
          <button className="ghost" onClick={() => download("svg")}>Download SVG</button>
        </div>

        <img
          key={src}
          src={src}
          alt="chart"
          className="eda-img"
          style={{ marginTop: 12, maxWidth: 680 }}
          hidden={err}
          onLoad={() => setErr(false)}
          onError={() => setErr(true)}
        />
        {err && (
          <p className="target-note" style={{ marginTop: 12 }}>
            Couldn’t render this combination — use a numeric column for histogram / box / violin /
            scatter, and a low-cardinality column for bar / pie.
          </p>
        )}
      </div>
    </section>
  );
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
