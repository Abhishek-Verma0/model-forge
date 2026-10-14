"use client";

// Small SVG charts shared by every results view (no chart library), each inside a
// Figure with SVG / PNG download buttons (plan §8.3, §15: graphs as PNG or SVG).
import { useRef } from "react";
import { saveBlob } from "./apiclient";

export const COLORS = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2", "#dc2626", "#4b5563", "#be185d",
                       "#65a30d", "#0d9488"];
const PNG_SCALE = 3; // our starting point: PNG pixels per SVG unit (print quality)

function standalone(svg) {
  const copy = svg.cloneNode(true);
  copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  copy.setAttribute("color", "#1f1c18");          // currentColor -> dark text outside the page
  return new XMLSerializer().serializeToString(copy);
}

export function downloadSvg(svg, name) {
  saveBlob(new Blob([standalone(svg)], { type: "image/svg+xml" }), `${name}.svg`);
}

export function downloadPng(svg, name) {
  const { width, height } = svg.viewBox.baseVal;
  const img = new Image();
  img.onload = () => {
    const canvas = Object.assign(document.createElement("canvas"), { width: width * PNG_SCALE, height: height * PNG_SCALE });
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((b) => saveBlob(b, `${name}.png`), "image/png");
  };
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(standalone(svg));
}

const slug = (s) => String(s || "chart").replace(/[^\w]+/g, "_").replace(/^_|_$/g, "").slice(0, 60) || "chart";

export function Figure({ title, width, height, children, note, name }) {
  const ref = useRef(null);
  const file = slug(name || title);
  return (
    <div className="panel" style={{ padding: 12, marginTop: 8, overflowX: "auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <b style={{ fontSize: 14 }}>{title}</b>
        <span style={{ display: "flex", gap: 6 }}>
          <button className="ghost" type="button" onClick={() => downloadSvg(ref.current, file)}>SVG</button>
          <button className="ghost" type="button" onClick={() => downloadPng(ref.current, file)}>PNG</button>
        </span>
      </div>
      <svg ref={ref} viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", maxWidth: width }} role="img"
           aria-label={title} fontFamily="Plus Jakarta Sans, system-ui, sans-serif">
        <rect x="0" y="0" width={width} height={height} fill="transparent" />
        {children}
      </svg>
      {note && <p className="note" style={{ margin: "4px 0 0" }}>{note}</p>}
    </div>
  );
}

export function niceTicks(lo, hi, n = 5) {
  if (!isFinite(lo) || !isFinite(hi)) return [];
  if (lo === hi) return [lo];
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => raw <= s) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

const tickText = (v) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(1) : String(Number(v.toPrecision(4))));

function extent(values) {
  const v = values.filter((x) => x !== null && x !== undefined && isFinite(x));
  if (!v.length) return [0, 1];
  let lo = Math.min(...v), hi = Math.max(...v);
  if (lo === hi) { lo -= 1; hi += 1; }
  return [lo, hi];
}

// Axes frame: returns scale functions and draws the axes, ticks and labels.
function axesFor({ width, height, xDomain, yDomain, left = 54, bottom = 38, top = 10, right = 14 }) {
  const x = (v) => left + ((v - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * (width - left - right);
  const y = (v) => height - bottom - ((v - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * (height - top - bottom);
  return { x, y, left, bottom, top, right };
}

function Axes({ a, width, height, xDomain, yDomain, xLabel, yLabel, xTicks, yTicks = true }) {
  const xt = xTicks || niceTicks(xDomain[0], xDomain[1]);
  const yt = yTicks ? niceTicks(yDomain[0], yDomain[1]) : [];
  return (
    <g fontSize="10" fill="currentColor">
      <line x1={a.left} y1={height - a.bottom} x2={width - a.right} y2={height - a.bottom} stroke="currentColor" opacity="0.35" />
      <line x1={a.left} y1={a.top} x2={a.left} y2={height - a.bottom} stroke="currentColor" opacity="0.35" />
      {xt.map((t) => (
        <g key={`x${t}`}>
          <line x1={a.x(t)} x2={a.x(t)} y1={height - a.bottom} y2={height - a.bottom + 4} stroke="currentColor" opacity="0.35" />
          <text x={a.x(t)} y={height - a.bottom + 15} textAnchor="middle">{tickText(t)}</text>
        </g>
      ))}
      {yt.map((t) => (
        <g key={`y${t}`}>
          <line x1={a.left} x2={width - a.right} y1={a.y(t)} y2={a.y(t)} stroke="currentColor" opacity="0.07" />
          <text x={a.left - 5} y={a.y(t) + 3} textAnchor="end">{tickText(t)}</text>
        </g>
      ))}
      {xLabel && <text x={(a.left + width - a.right) / 2} y={height - 4} textAnchor="middle">{xLabel}</text>}
      {yLabel && <text x={12} y={(a.top + height - a.bottom) / 2} textAnchor="middle"
                       transform={`rotate(-90 12 ${(a.top + height - a.bottom) / 2})`}>{yLabel}</text>}
    </g>
  );
}

function Legend({ items, x, y }) {
  return (
    <g fontSize="10" fill="currentColor">
      {items.map((it, i) => (
        <g key={it.label} transform={`translate(${x}, ${y + i * 14})`}>
          <rect width="10" height="3" y="-4" fill={it.color} />
          <text x="14" y="0">{it.label}</text>
        </g>
      ))}
    </g>
  );
}

// series: [{label, x: [], y: [], sd?: [], dashed?, color?}]
export function LinePlot({ title, series, xLabel, yLabel, width = 600, height = 280, note, yDomain, xDomain, name, points = true,
                          legendItems }) {
  const xs = series.flatMap((s) => s.x), ys = series.flatMap((s) => s.y.map((v, i) => [v, v - (s.sd?.[i] || 0), v + (s.sd?.[i] || 0)]).flat());
  const xd = xDomain || extent(xs), yd = yDomain || extent(ys);
  const a = axesFor({ width, height, xDomain: xd, yDomain: yd });
  return (
    <Figure title={title} width={width} height={height} note={note} name={name}>
      <Axes a={a} width={width} height={height} xDomain={xd} yDomain={yd} xLabel={xLabel} yLabel={yLabel} />
      {series.map((s, k) => {
        const color = s.color || COLORS[k % COLORS.length];
        const pts = s.x.map((xv, i) => [xv, s.y[i]]).filter(([, yv]) => yv !== null && yv !== undefined && isFinite(yv));
        const band = s.sd ? s.x.map((xv, i) => [xv, s.y[i] + (s.sd[i] || 0)]).concat(
          s.x.map((xv, i) => [xv, s.y[i] - (s.sd[i] || 0)]).reverse()).filter(([, v]) => isFinite(v)) : null;
        return (
          <g key={s.label}>
            {band && <polygon points={band.map(([xv, yv]) => `${a.x(xv)},${a.y(yv)}`).join(" ")} fill={color} opacity="0.12" />}
            <polyline points={pts.map(([xv, yv]) => `${a.x(xv)},${a.y(yv)}`).join(" ")} fill="none" stroke={color}
                      strokeWidth="1.8" strokeDasharray={s.dashed ? "5 4" : undefined} />
            {points && pts.length <= 60 && pts.map(([xv, yv], i) => <circle key={i} cx={a.x(xv)} cy={a.y(yv)} r="2.4" fill={color} />)}
          </g>
        );
      })}
      <Legend items={legendItems || series.map((s, k) => ({ label: s.label, color: s.color || COLORS[k % COLORS.length] }))}
              x={a.left + 10} y={a.top + 10} />
    </Figure>
  );
}

// items: [{label, value, error?}] -> horizontal bars, largest first as given
export function BarPlot({ title, items, xLabel, width = 600, note, color = COLORS[0], name, labelWidth = 190 }) {
  const rowH = 20, top = 8, bottom = 36;
  const height = top + bottom + rowH * Math.max(items.length, 1);
  const vals = items.flatMap((d) => [d.value, d.value + (d.error || 0), d.value - (d.error || 0), 0]);
  const [lo, hi] = extent(vals);
  const a = axesFor({ width, height, xDomain: [lo, hi], yDomain: [0, 1], left: labelWidth, bottom, top });
  return (
    <Figure title={title} width={width} height={height} note={note} name={name}>
      <Axes a={a} width={width} height={height} xDomain={[lo, hi]} yDomain={[0, 1]} xLabel={xLabel} yTicks={false} />
      {items.map((d, i) => {
        const yy = top + i * rowH;
        const x0 = a.x(Math.max(lo, 0)), x1 = a.x(d.value);
        return (
          <g key={`${d.label}${i}`}>
            <text x={labelWidth - 6} y={yy + rowH / 2 + 3} fontSize="10" textAnchor="end" fill="currentColor">
              {String(d.label).length > 30 ? String(d.label).slice(0, 29) + "…" : d.label}
            </text>
            <rect x={Math.min(x0, x1)} y={yy + 3} width={Math.max(Math.abs(x1 - x0), 1)} height={rowH - 6} fill={d.color || color} opacity="0.85" rx="2" />
            {d.error > 0 && <line x1={a.x(d.value - d.error)} x2={a.x(d.value + d.error)} y1={yy + rowH / 2} y2={yy + rowH / 2}
                                  stroke="currentColor" strokeWidth="1.2" />}
            <title>{`${d.label}: ${d.value}${d.error ? ` ± ${d.error}` : ""}`}</title>
          </g>
        );
      })}
    </Figure>
  );
}

// points: [{x, y, color?, r?, title?}]; line: [[x1, y1], [x2, y2]] (dashed reference)
export function ScatterPlot({ title, points, xLabel, yLabel, line, width = 520, height = 340, note, legend, name, labels }) {
  const xd = extent(points.map((p) => p.x).concat(line ? line.map((l) => l[0]) : []));
  const yd = extent(points.map((p) => p.y).concat(line ? line.map((l) => l[1]) : []));
  const a = axesFor({ width, height, xDomain: xd, yDomain: yd });
  return (
    <Figure title={title} width={width} height={height} note={note} name={name}>
      <Axes a={a} width={width} height={height} xDomain={xd} yDomain={yd} xLabel={xLabel} yLabel={yLabel} />
      {line && <line x1={a.x(line[0][0])} y1={a.y(line[0][1])} x2={a.x(line[1][0])} y2={a.y(line[1][1])}
                     stroke="currentColor" opacity="0.45" strokeDasharray="5 4" />}
      {points.map((p, i) => (
        <circle key={i} cx={a.x(p.x)} cy={a.y(p.y)} r={p.r || 2.6} fill={p.color || COLORS[0]} opacity={p.r ? 0.7 : 0.6}>
          {p.title && <title>{p.title}</title>}
        </circle>
      ))}
      {labels && points.filter((p) => p.label).map((p, i) => (
        <text key={`l${i}`} x={a.x(p.x)} y={a.y(p.y) - (p.r || 3) - 2} fontSize="9" textAnchor="middle" fill="currentColor">{p.label}</text>
      ))}
      {legend && <Legend items={legend} x={width - 130} y={a.top + 10} />}
    </Figure>
  );
}

export function Histogram({ title, edges, counts, xLabel, width = 520, height = 260, note, name }) {
  const xd = [edges[0], edges[edges.length - 1]], yd = [0, Math.max(...counts, 1)];
  const a = axesFor({ width, height, xDomain: xd, yDomain: yd });
  return (
    <Figure title={title} width={width} height={height} note={note} name={name}>
      <Axes a={a} width={width} height={height} xDomain={xd} yDomain={yd} xLabel={xLabel} yLabel="rows" />
      {counts.map((c, i) => (
        <rect key={i} x={a.x(edges[i])} y={a.y(c)} width={Math.max(a.x(edges[i + 1]) - a.x(edges[i]) - 1, 1)}
              height={a.y(0) - a.y(c)} fill={COLORS[3]} opacity="0.8" />
      ))}
    </Figure>
  );
}
