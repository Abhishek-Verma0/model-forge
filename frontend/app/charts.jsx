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

function Heatmap({ correlation }) {
  const { columns, matrix } = correlation;
  const color = (v) => {
    if (v == null) return "var(--border)";
    const a = Math.min(Math.abs(v), 1);
    return v >= 0 ? `rgba(79,70,229,${a})` : `rgba(220,38,38,${a})`; // blue = +, red = −
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
              style={{ background: color(v) }}
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
