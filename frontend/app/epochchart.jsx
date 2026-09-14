"use client";

// Epoch curves for one model: one colour per fit (fold 1..k, final), solid =
// training loss, dashed = validation loss on held-back rows. Inline SVG, no library.
const COLORS = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2", "#dc2626", "#4b5563"];
const W = 560, H = 220, PAD = 36;

export default function EpochChart({ data, title }) {
  const fits = Object.entries(data?.fits || {}).filter(([, pts]) => pts.length);
  if (!fits.length) return <p className="note">No epoch data yet.</p>;
  const vals = fits.flatMap(([, pts]) => pts.flatMap(([, a, b]) => [a, b])).filter((v) => v != null && isFinite(v));
  const total = data.total || Math.max(...fits.flatMap(([, pts]) => pts.map(([e]) => e)));
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = (e) => PAD + ((e - 1) / Math.max(total - 1, 1)) * (W - 2 * PAD);
  const y = (v) => H - PAD - ((v - lo) / (hi - lo || 1)) * (H - 2 * PAD);
  const line = (pts, i) => pts.filter((p) => p[i] != null).map((p) => `${x(p[0])},${y(p[i])}`).join(" ");

  return (
    <div className="panel" style={{ padding: 12, marginTop: 8, overflowX: "auto" }}>
      <p style={{ margin: "0 0 4px", fontWeight: 600 }}>{title} — {data.metric || "loss"} per epoch</p>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W }} role="img" aria-label={`${title} epoch curves`}>
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="currentColor" opacity="0.3" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="currentColor" opacity="0.3" />
        <text x={PAD} y={H - 10} fontSize="10" fill="currentColor">1</text>
        <text x={W - PAD} y={H - 10} fontSize="10" fill="currentColor" textAnchor="end">{total} epochs</text>
        <text x={4} y={PAD} fontSize="10" fill="currentColor">{hi.toPrecision(3)}</text>
        <text x={4} y={H - PAD} fontSize="10" fill="currentColor">{lo.toPrecision(3)}</text>
        {fits.map(([name, pts], k) => (
          <g key={name}>
            <polyline points={line(pts, 1)} fill="none" stroke={COLORS[k % COLORS.length]} strokeWidth="1.6" />
            <polyline points={line(pts, 2)} fill="none" stroke={COLORS[k % COLORS.length]} strokeWidth="1.6" strokeDasharray="4 3" />
          </g>
        ))}
      </svg>
      <p className="note" style={{ margin: 0 }}>
        {fits.map(([name, pts], k) => (
          <span key={name} style={{ color: COLORS[k % COLORS.length], marginRight: 10 }}>
            ■ {name} ({pts.length}{pts[pts.length - 1]?.[2] != null ? `, last val ${pts[pts.length - 1][2].toPrecision(3)}` : ""})
          </span>
        ))}
        <span>solid = training, dashed = validation</span>
      </p>
    </div>
  );
}
