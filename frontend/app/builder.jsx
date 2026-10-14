"use client";

import { useEffect, useState } from "react";

import { blobUrl, download } from "./apiclient";
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
  const [src, setSrc] = useState("");

  const params = new URLSearchParams({ id, kind, x });
  if (NEEDS_Y.includes(kind)) params.set("y", y);
  if (hue) params.set("hue", hue);
  const query = params.toString();

  // the chart route needs the session token, which an <img src> can't send: fetch it, show the blob
  useEffect(() => {
    if (!id) return undefined;
    let url = "", live = true;
    blobUrl(`/api/chart?${query}`)
      .then((u) => { url = u; if (live) { setSrc(u); setErr(false); } })
      .catch(() => { if (live) { setSrc(""); setErr(true); } });
    return () => { live = false; if (url) URL.revokeObjectURL(url); };
  }, [id, query]);

  if (!id) return null;

  function save(fmt) {
    download(`/api/chart?${query}&fmt=${fmt}`, `chart_${kind}_${x}.${fmt}`).catch(() => setErr(true));
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
          <button className="ghost" onClick={() => save("png")}>Download PNG</button>
          <button className="ghost" onClick={() => save("svg")}>Download SVG</button>
        </div>

        <div className="builder-img-wrap" style={{ position: "relative", width: "100%", maxWidth: 680, marginTop: 12 }}>
          {src && !err && (
            // eslint-disable-next-line @next/next/no-img-element -- a blob: URL from an authenticated fetch
            <img src={src} alt={`${kind} chart`} className="eda-img" style={{ width: "100%", height: "auto", display: "block" }} />
          )}
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
