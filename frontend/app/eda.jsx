"use client";

import Image from "next/image";
import { useState } from "react";

import { download } from "./apiclient";

// Backend-rendered EDA charts (matplotlib/seaborn) arrive as base64 PNG data
// URIs in data.eda = [{key, title, image}]. We show them; a download asks the
// backend to draw that chart again -- PNG at print resolution or SVG (vector).
export default function EdaGallery({ data }) {
  const eda = data.eda || [];
  const [err, setErr] = useState("");
  if (!eda.length) return null;

  function save(c, fmt) {
    setErr("");
    const name = c.title.replace(/\W+/g, "_");
    // charts from before keys existed: the on-screen PNG is all there is
    if (!c.key) {
      Object.assign(document.createElement("a"), { href: c.image, download: `${name}.png` }).click();
      return;
    }
    const q = new URLSearchParams({ id: data.id, key: c.key, fmt });
    download(`/api/eda/chart?${q}`, `${name}.${fmt}`).catch((e) => setErr(e.message));
  }

  return (
    <section>
      <p className="sectitle">EDA charts</p>
      {err && <div className="error">{err}</div>}
      <div className="eda-grid">
        {eda.map((c, i) => (
          <div className="chart" key={c.key || i}>
            <div className="chart-title" title={c.title}>
              {c.title}
            </div>
            <div className="eda-img-wrap">
              <Image
                src={c.image}
                alt={c.title}
                width={640}
                height={420}
                sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                className="eda-img"
                style={{ width: "100%", height: "auto" }}
                unoptimized
              />
            </div>
            <div className="eda-dls">
              <button className="ghost eda-dl" onClick={() => save(c, "png")} title="PNG at 300 dpi">Download PNG</button>
              {c.key && <button className="ghost eda-dl" onClick={() => save(c, "svg")}>Download SVG</button>}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
