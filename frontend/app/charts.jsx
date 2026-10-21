"use client";

import { Fragment } from "react";

export default function Charts({ data }) {
  const { profile, report } = data;
  const charts = data.charts || {};
  const info = profile.columns_info;
  const semType = {};
  for (const c of report.checks.column_types.columns) semType[c.column] = c.detected_type;
  const catCols = profile.column_names.filter(
    (n) => ["boolean", "categorical"].includes(semType[n]) && info[n].top_values
  );

  return (
    <>
      {catCols.length > 0 && (
        <Section title="Categorical counts">
          <div className="chart-grid">
            {catCols.map((n) => (
              <CatBar key={n} name={n} topValues={info[n].top_values} />
            ))}
          </div>
        </Section>
      )}

      {charts.correlation && (
        <Section title="Correlation (numeric)">
          <div className="panel" style={{ padding: 16, overflowX: "auto" }}>
            <Heatmap correlation={charts.correlation} />
          </div>
        </Section>
      )}
    </>
  );
}

function Section({ title, children }) {
  return (
    <section>
      <p className="sectitle">{title}</p>
      {children}
    </section>
  );
}

function CatBar({ name, topValues }) {
  const max = Math.max(...topValues.map((t) => t.count), 1);
  return (
    <div className="chart">
      <div className="chart-title" title={name}>
        {name}
      </div>
      {topValues.map((t, i) => (
        <div className="hbar" key={i}>
          <div className="hbar-label" title={String(t.value)}>
            {String(t.value)}
          </div>
          <div className="hbar-track">
            <div className="hbar-fill" style={{ width: (t.count / max) * 100 + "%" }} />
          </div>
          <div className="hbar-num">{t.count.toLocaleString()}</div>
        </div>
      ))}
    </div>
  );
}

// WCAG 2 relative luminance of an [r, g, b] colour (0-255)
const luminance = (rgb) => {
  const [r, g, b] = rgb.map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
// Black, not --text #1F1C18: as a cell's tint goes from pale to full, it passes the shade where white
// and dark text read equally well. There the best contrast is sqrt(1.05 / (dark + 0.05)): 4.10:1
// with #1F1C18, 4.58:1 with black -- so with black every r from -1 to 1 meets WCAG AA (4.5:1).
const TEXT_DARK = [0, 0, 0];
const contrast = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);

function Heatmap({ correlation }) {
  const { columns, matrix } = correlation;
  // blue = +, red = −, stronger = more opaque over the white panel
  const base = (v) => (v >= 0 ? [79, 70, 229] : [220, 38, 38]);
  const color = (v) => {
    if (v == null) return "var(--border)";
    return `rgba(${base(v).join(",")},${Math.min(Math.abs(v), 1)})`;
  };
  // the number takes whichever of white or dark text contrasts more with its cell: near 0 the
  // cell is almost white, and the old always-white numbers vanished there
  const ink = (v) => {
    if (v == null) return undefined;
    const a = Math.min(Math.abs(v), 1);
    const cell = luminance(base(v).map((c) => 255 - a * (255 - c)));   // the tint as drawn over #FFFFFF
    return contrast(cell, 1) >= contrast(cell, luminance(TEXT_DARK)) ? "#fff" : "#000";
  };
  return (
    <div
      className="heatmap"
      style={{ gridTemplateColumns: `110px repeat(${columns.length}, minmax(44px, 1fr))` }}
    >
      <div />
      {columns.map((c) => (
        <div key={c} className="hm-col" title={c}>
          {c}
        </div>
      ))}
      {matrix.map((row, i) => (
        <Fragment key={i}>
          <div className="hm-row" title={columns[i]}>
            {columns[i]}
          </div>
          {row.map((v, j) => (
            <div
              key={j}
              className="hm-cell"
              style={{ background: color(v), color: ink(v) }}
              title={`${columns[i]} × ${columns[j]}: ${v ?? "—"}`}
            >
              {v == null ? "" : v.toFixed(2)}
            </div>
          ))}
        </Fragment>
      ))}
    </div>
  );
}
