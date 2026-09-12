"use client";

import { useMemo, useState } from "react";

export default function DatasetPreview({ data, onProceed, onClean, target, setTarget }) {
  const { profile, report, sample = [], filename } = data;
  const info = profile?.columns_info || {};
  const checks = report?.checks || {};

  // Find categorical/boolean columns suitable for class distribution preview
  const classCols = useMemo(() => {
    return (profile?.column_names || []).filter((col) => {
      const type = info[col]?.type;
      const unique = info[col]?.unique_values || 0;
      // Suitable for donut if boolean or categorical with 2 to 10 unique values
      return (type === "categorical" || type === "boolean" || unique <= 10) && unique > 1;
    });
  }, [profile, info]);

  // Default selected column for the donut chart: target or the best class column
  const [selectedCol, setSelectedCol] = useState(() => {
    if (target && classCols.includes(target)) return target;
    // Prefer column named "churn", "target", "label", "outcome" or first class col
    const preferred = classCols.find((c) =>
      /churn|target|label|outcome|class|status|default/i.test(c)
    );
    return preferred || classCols[0] || profile?.column_names?.[0] || "";
  });

  // Calculate distribution for selected column
  const distribution = useMemo(() => {
    if (!selectedCol || !sample.length) return [];
    const counts = {};
    let total = 0;
    for (const row of sample) {
      const rawVal = row[selectedCol];
      const val = rawVal === null || rawVal === undefined ? "(missing)" : String(rawVal);
      counts[val] = (counts[val] || 0) + 1;
      total++;
    }
    const colors = [
      "#C26E38", // warm bronze
      "#D97706", // amber
      "#2D6A4F", // deep sage green
      "#4F46E5", // indigo
      "#BE185D", // berry
      "#0891B2", // cyan
      "#854D0E", // dark gold
      "#6B7280", // slate
    ];
    let startAngle = 0;
    return Object.entries(counts).map(([label, count], i) => {
      const pct = total > 0 ? (count / total) * 100 : 0;
      const angle = total > 0 ? (count / total) * 360 : 0;
      const item = {
        label,
        count,
        pct: pct.toFixed(1),
        color: colors[i % colors.length],
        startAngle,
        angle,
      };
      startAngle += angle;
      return item;
    });
  }, [selectedCol, sample]);

  // Search filter for sample table
  const [searchQuery, setSearchQuery] = useState("");

  const filteredSample = useMemo(() => {
    if (!searchQuery.trim()) return sample;
    const q = searchQuery.toLowerCase();
    return sample.filter((row) =>
      Object.values(row).some((val) => String(val).toLowerCase().includes(q))
    );
  }, [sample, searchQuery]);

  // Calculate type breakdown
  const typeCounts = useMemo(() => {
    const counts = { numeric: 0, categorical: 0, boolean: 0, other: 0 };
    for (const col of profile?.column_names || []) {
      const t = info[col]?.type;
      if (t === "numeric") counts.numeric++;
      else if (t === "categorical") counts.categorical++;
      else if (t === "boolean") counts.boolean++;
      else counts.other++;
    }
    return counts;
  }, [profile, info]);

  // Helper for SVG donut slice
  function getCoordinatesForPercent(percent) {
    const x = Math.cos(2 * Math.PI * percent);
    const y = Math.sin(2 * Math.PI * percent);
    return [x, y];
  }

  return (
    <div className="preview-view-container">
      {/* Header */}
      <div className="preview-view-header">
        <div>
          <h1 className="preview-title">Dataset Preview</h1>
          <p className="preview-subtitle">
            Inspect dataset schema summary, target class balance, and live sample rows before running transformations.
          </p>
        </div>
      </div>

      {/* Top 2-Column Grid matching reference */}
      <div className="preview-top-grid">
        {/* Left Card: Dataset Summary */}
        <div className="preview-card summary-card">
          <div className="preview-card-header">
            <div className="card-title-group">
              <span className="card-icon">📋</span>
              <h3 className="card-heading">Dataset Summary</h3>
            </div>
            <span className="filename-tag">{filename}</span>
          </div>

          <div className="summary-list">
            <div className="summary-row">
              <span className="summary-label">Dataset Name</span>
              <span className="summary-value font-bold">{filename.replace(/\.[^/.]+$/, "")}</span>
            </div>

            <div className="summary-row">
              <span className="summary-label">Total Records (Rows)</span>
              <span className="summary-value font-bold">{profile?.rows?.toLocaleString() || "—"}</span>
            </div>

            <div className="summary-row">
              <span className="summary-label">Feature Columns</span>
              <div className="summary-value column-breakdown">
                <span className="col-total">{profile?.columns || 0} columns</span>
                <span className="col-chips">
                  {typeCounts.numeric > 0 && <span className="type-badge num">{typeCounts.numeric} Numeric</span>}
                  {typeCounts.categorical > 0 && <span className="type-badge cat">{typeCounts.categorical} Text</span>}
                  {typeCounts.boolean > 0 && <span className="type-badge bool">{typeCounts.boolean} Bool</span>}
                </span>
              </div>
            </div>

            <div className="summary-row">
              <span className="summary-label">Missing Cell Rate</span>
              <span className="summary-value">
                {checks.missing_values?.summary?.total_missing_cells === 0 ? (
                  <span className="clean-badge">✓ 0% Clean (0 nulls)</span>
                ) : (
                  <span className="warn-badge">
                    {checks.missing_values?.summary?.total_missing_cells} missing cells
                  </span>
                )}
              </span>
            </div>

            <div className="summary-row">
              <span className="summary-label">Duplicate Rows</span>
              <span className="summary-value">
                {profile?.duplicate_rows === 0 ? (
                  <span className="clean-badge">✓ None (0%)</span>
                ) : (
                  <span className="warn-badge">{profile?.duplicate_rows} duplicates</span>
                )}
              </span>
            </div>

            <div className="summary-row">
              <span className="summary-label">Recommended Task</span>
              <span className="summary-value task-type-badge">
                {selectedCol && info[selectedCol]?.type === "numeric"
                  ? "📈 Regression"
                  : "🎯 Classification"}
              </span>
            </div>
          </div>
        </div>

        {/* Right Card: Class / Target Distribution matching reference Donut */}
        <div className="preview-card distribution-card">
          <div className="preview-card-header">
            <div className="card-title-group">
              <span className="card-icon">🍩</span>
              <h3 className="card-heading">Class & Feature Distribution</h3>
            </div>
            {classCols.length > 1 && (
              <div className="col-select-wrap">
                <label htmlFor="dist-col-select" className="sr-only">Select Column</label>
                <select
                  id="dist-col-select"
                  className="dist-col-dropdown"
                  value={selectedCol}
                  onChange={(e) => setSelectedCol(e.target.value)}
                >
                  {classCols.map((c) => (
                    <option key={c} value={c}>
                      {c} {c === target ? "(Target)" : ""}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="donut-chart-container">
            {/* SVG Donut Visual */}
            <div className="donut-visual-wrap">
              <svg viewBox="0 0 160 160" className="donut-svg">
                {distribution.length > 0 ? (
                  (() => {
                    let cumulativePercent = 0;
                    return distribution.map((slice, idx) => {
                      const percent = parseFloat(slice.pct) / 100;
                      // Handle single 100% slice edge case
                      if (distribution.length === 1) {
                        return (
                          <circle
                            key={idx}
                            cx="80"
                            cy="80"
                            r="55"
                            fill="none"
                            stroke={slice.color}
                            strokeWidth="24"
                          />
                        );
                      }
                      const [startX, startY] = getCoordinatesForPercent(cumulativePercent);
                      cumulativePercent += percent;
                      const [endX, endY] = getCoordinatesForPercent(cumulativePercent);
                      const largeArcFlag = percent > 0.5 ? 1 : 0;
                      const pathData = [
                        `M ${80 + 55 * startX} ${80 + 55 * startY}`,
                        `A 55 55 0 ${largeArcFlag} 1 ${80 + 55 * endX} ${80 + 55 * endY}`,
                      ].join(" ");

                      return (
                        <path
                          key={idx}
                          d={pathData}
                          fill="none"
                          stroke={slice.color}
                          strokeWidth="24"
                          strokeLinecap="butt"
                        />
                      );
                    });
                  })()
                ) : (
                  <circle cx="80" cy="80" r="55" fill="none" stroke="#E8DFD1" strokeWidth="24" />
                )}
              </svg>
              <div className="donut-center-info">
                <span className="donut-center-col">{selectedCol}</span>
                <span className="donut-center-sub">{distribution.length} Classes</span>
              </div>
            </div>

            {/* Donut Legend */}
            <div className="donut-legend-list">
              {distribution.map((slice, i) => (
                <div key={i} className="legend-item-row">
                  <div className="legend-label-group">
                    <span className="legend-dot" style={{ backgroundColor: slice.color }}></span>
                    <span className="legend-class-name">{slice.label}</span>
                  </div>
                  <div className="legend-metric-group">
                    <span className="legend-count">{slice.count}</span>
                    <span className="legend-pct">({slice.pct}%)</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Lower Section: Interactive Sample Data Preview Grid matching reference */}
      <div className="sample-table-card">
        <div className="sample-table-header">
          <div className="table-header-left">
            <h3 className="table-title">Sample Data Records</h3>
            <span className="table-pill">First {sample.length} Rows</span>
          </div>

          <div className="table-header-right">
            <div className="table-search-box">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <input
                type="text"
                className="table-search-input"
                placeholder="Search sample rows…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              {searchQuery && (
                <button
                  type="button"
                  className="btn-clear-search"
                  onClick={() => setSearchQuery("")}
                >
                  ✕
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Scrollable Data Table */}
        <div className="sample-table-scroll-wrapper">
          <table className="sample-grid-table">
            <thead>
              <tr>
                <th className="th-row-num">#</th>
                {(profile?.column_names || []).map((col) => {
                  const type = info[col]?.type;
                  const isTarget = col === target;
                  return (
                    <th key={col} className={`th-col-header ${isTarget ? "is-target" : ""}`}>
                      <div className="col-header-inner">
                        <span className="col-name">{col}</span>
                        <span className={`col-type-pill ${type || "cat"}`}>
                          {type === "numeric" ? "#" : type === "boolean" ? "✓" : "A"}
                        </span>
                      </div>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {filteredSample.length > 0 ? (
                filteredSample.map((row, idx) => (
                  <tr key={idx}>
                    <td className="td-row-num">{idx + 1}</td>
                    {(profile?.column_names || []).map((col) => {
                      const val = row[col];
                      const isTarget = col === target;
                      return (
                        <td key={col} className={`td-data-cell ${isTarget ? "is-target" : ""}`}>
                          {val === null || val === undefined ? (
                            <span className="null-val-chip">null</span>
                          ) : typeof val === "boolean" ? (
                            <span className={`bool-val ${val ? "true" : "false"}`}>
                              {val ? "true" : "false"}
                            </span>
                          ) : typeof val === "number" ? (
                            Number.isInteger(val) ? val.toLocaleString() : val.toFixed(2)
                          ) : (
                            String(val)
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={(profile?.column_names?.length || 1) + 1} className="no-match-cell">
                    No rows match your search query "{searchQuery}"
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Action Bar Footer */}
      <div className="preview-action-bar">
        <button id="btn-proceed-audit" type="button" className="btn-proceed-audit" onClick={onProceed}>
          <span>Proceed to Quality Audit</span>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        </button>

        <button id="btn-clean-jump" type="button" className="btn-clean-jump" onClick={onClean}>
          <span>Jump to Cleaning Studio</span>
        </button>
      </div>
    </div>
  );
}
