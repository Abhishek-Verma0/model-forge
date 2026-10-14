"use client";

// Epoch curves for one model: one colour per fit (fold 1..k, final), solid =
// training loss, dashed = validation loss on held-back rows. Downloadable.
import { COLORS, LinePlot } from "./chartkit";

export default function EpochChart({ data, title }) {
  const fits = Object.entries(data?.fits || {}).filter(([, pts]) => pts.length);
  if (!fits.length) return <p className="note">No epoch data yet.</p>;
  const series = fits.flatMap(([name, pts], k) => {
    const color = COLORS[k % COLORS.length];
    const s = [{ label: `${name} training`, x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), color }];
    if (pts.some((p) => p[2] != null)) s.push({ label: `${name} validation`, x: pts.map((p) => p[0]), y: pts.map((p) => p[2]), color, dashed: true });
    return s;
  });
  return (
    <LinePlot title={`${title} — ${data.metric || "loss"} per epoch`} xLabel="epoch" yLabel={data.metric || "loss"}
              series={series} points={false} height={300} width={640}
              legendItems={fits.map(([name], k) => ({ label: name, color: COLORS[k % COLORS.length] }))}
              note="Solid = training rows, dashed = held-back validation rows. A rising dashed line while the solid one falls = overfitting." />
  );
}
