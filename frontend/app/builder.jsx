"use client";

import { useEffect, useState } from "react";

import { blobUrl, download } from "./apiclient";
const NEEDS_Y = ["scatter", "line"];   // plotted against a second column
const NO_HUE = ["pie", "length", "words"]; // one column only
const LABEL = { length: "text length", words: "common words", tsne: "t-SNE map" };

// User-driven charts: pick a type + column(s) + optional color/group, and the
// backend renders that exact chart (matplotlib/seaborn) as a PNG we <img>.
// Each chart lists only the columns it can show (data.charts.builder, from the
// audit -- eda/plot.py FITS): an ID or a many-valued text column is never drawn
// one bar per value (that took 152 s on 12,000 ids, measured 2026-10-21).
export default function ChartBuilder({ data }) {
  const fits = data.charts?.builder || { x: {}, y: [], hue: [] };
  const kinds = Object.keys(fits.x).filter((k) => (k === "tsne" ? fits.y.length >= 2
    : fits.x[k].length > 0 && (!NEEDS_Y.includes(k) || fits.y.length > 0)));
  const id = data.id;
  const [kind, setKind] = useState("");
  const [x, setX] = useState("");
  const [y, setY] = useState("");
  const [hue, setHue] = useState("");
  const [err, setErr] = useState("");
  const [src, setSrc] = useState("");

  // a choice the table no longer allows (e.g. a column dropped by cleaning) falls back to the first that fits
  const k = kinds.includes(kind) ? kind : kinds[0] || "";
  const xs = fits.x[k] || [];
  const xv = xs.includes(x) ? x : xs[0] || "";
  const yv = fits.y.includes(y) ? y : fits.y.find((c) => c !== xv) || fits.y[0] || "";
  const hv = !NO_HUE.includes(k) && fits.hue.includes(hue) ? hue : "";
  const ready = !!k && (k === "tsne" || !!xv) && (!NEEDS_Y.includes(k) || !!yv);

  const params = new URLSearchParams({ id, kind: k });
  if (k !== "tsne") params.set("x", xv);
  if (NEEDS_Y.includes(k)) params.set("y", yv);
  if (hv) params.set("hue", hv);
  const query = params.toString();

  // the chart route needs the session token, which an <img src> can't send: fetch it, show the blob
  useEffect(() => {
    if (!id || !ready) return undefined;
    let url = "", live = true;
    blobUrl(`/api/chart?${query}`)
      .then((u) => { url = u; if (live) { setSrc(u); setErr(""); } })
      .catch((e) => { if (live) { setSrc(""); setErr(e.message); } });
    return () => { live = false; if (url) URL.revokeObjectURL(url); };
  }, [id, query, ready]);

  if (!id) return null;

  function save(fmt) {
    download(`/api/chart?${query}&fmt=${fmt}`, `chart_${k}_${xv || "map"}.${fmt}`).catch((e) => setErr(e.message));
  }

  return (
    <section>
      <p className="sectitle">Chart builder</p>
      <div className="panel" style={{ padding: 16 }}>
        {!kinds.length ? (
          <p className="target-note" style={{ margin: 0 }}>
            {data.charts?.builder ? "No column of this table suits a chart." : "Reload the page to use the chart builder."}
          </p>
        ) : (
          <>
            <div className="builder-row">
              <Field label="Chart type">
                <select value={k} onChange={(e) => setKind(e.target.value)}>
                  {kinds.map((o) => (
                    <option key={o} value={o}>{LABEL[o] || o}</option>
                  ))}
                </select>
              </Field>
              {k !== "tsne" && (
                <Field label="X axis">
                  <select value={xv} onChange={(e) => setX(e.target.value)}>
                    {xs.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </Field>
              )}
              {NEEDS_Y.includes(k) && (
                <Field label="Y axis">
                  <select value={yv} onChange={(e) => setY(e.target.value)}>
                    {fits.y.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </Field>
              )}
              {!NO_HUE.includes(k) && fits.hue.length > 0 && (
                <Field label="Color / group by">
                  <select value={hv} onChange={(e) => setHue(e.target.value)}>
                    <option value="">(none)</option>
                    {fits.hue.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </Field>
              )}
              <button className="ghost" onClick={() => save("png")}>Download PNG</button>
              <button className="ghost" onClick={() => save("svg")}>Download SVG</button>
            </div>

            <div className="builder-img-wrap" style={{ position: "relative", width: "100%", maxWidth: 680, marginTop: 12 }}>
              {src && !err && (
                // eslint-disable-next-line @next/next/no-img-element -- a blob: URL from an authenticated fetch
                <img src={src} alt={`${k} chart`} className="eda-img" style={{ width: "100%", height: "auto", display: "block" }} />
              )}
            </div>
            {err && <p className="target-note" style={{ marginTop: 12 }}>{err}</p>}
            <p className="note" style={{ marginTop: 8 }}>
              Each chart lists only the columns it can show. Text ID columns are not charted; text with many different
              values shows its 10 most common values, and free text its length or most used words.
            </p>
          </>
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
