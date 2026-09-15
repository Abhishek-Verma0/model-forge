"use client";

import { useState } from "react";
import Image from "next/image";

import { API } from "./apiclient";
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

  const params = new URLSearchParams({ id, kind, x });
  if (NEEDS_Y.includes(kind)) params.set("y", y);
  if (hue) params.set("hue", hue);
  const src = `${API}/api/chart?${params.toString()}`;

  function download(fmt) {
    const p = new URLSearchParams(params);
    p.set("fmt", fmt);
    const a = document.createElement("a");
    a.href = `${API}/api/chart?${p.toString()}`;
    a.download = `chart_${kind}_${x}.${fmt}`;
    a.click();
  }

  return (
    <section>
      <p className="sectitle">Chart builder</p>
      <div className="panel" style={{ padding: 16 }}>
        <div className="builder-row">
          <Field label="Chart type">
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              {KINDS.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </Field>
          <Field label="X axis">
            <select value={x} onChange={(e) => setX(e.target.value)}>
              {cols.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </Field>
          {NEEDS_Y.includes(kind) && (
            <Field label="Y axis">
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

        <div className="builder-img-wrap" style={{ position: "relative", width: "100%", maxWidth: 680, marginTop: 12 }}>
          <Image
            key={src}
            src={src}
            alt="chart"
            width={680}
            height={450}
            sizes="(max-width: 768px) 100vw, 680px"
            className="eda-img"
            style={{ width: "100%", height: "auto", display: err ? "none" : "block" }}
            unoptimized
            onLoad={() => setErr(false)}
            onError={() => setErr(true)}
          />
        </div>
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
