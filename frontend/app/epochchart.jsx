"use client";

// Epoch curves for one model: one colour per fit (fold 1..k, final), solid =
// training rows, dashed = validation rows (held back). Loss for every epoch-based
// model; accuracy too for classification neural networks (plan §11.1). Downloadable.
import { COLORS, LinePlot } from "./chartkit";

function curves(byFit) {
  const fits = Object.entries(byFit || {}).filter(([, pts]) => pts.length);
  const series = fits.flatMap(([name, pts], k) => {
    const color = COLORS[k % COLORS.length];
    const s = [{ label: `${name} training`, x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), color }];
    if (pts.some((p) => p[2] != null)) s.push({ label: `${name} validation`, x: pts.map((p) => p[0]), y: pts.map((p) => p[2]), color, dashed: true });
    return s;
  });
  return { fits, series, legend: fits.map(([name], k) => ({ label: name, color: COLORS[k % COLORS.length] })) };
}

export default function EpochChart({ data, title }) {
  const loss = curves(data?.fits);
  if (!loss.fits.length) return <p className="note">No epoch data yet.</p>;
  const acc = curves(data?.accuracy);
  return (
    <>
      <LinePlot title={`${title} — ${data.metric || "loss"} per epoch`} xLabel="epoch" yLabel={data.metric || "loss"}
                series={loss.series} points={false} height={300} width={640} legendItems={loss.legend}
                note="Solid = training rows, dashed = held-back validation rows. A rising dashed line while the solid one falls = overfitting." />
      {acc.fits.length > 0 && (
        <LinePlot title={`${title} — accuracy per epoch`} xLabel="epoch" yLabel="accuracy" yDomain={[0, 1]}
                  series={acc.series} points={false} height={260} width={640} legendItems={acc.legend}
                  note="Share of rows classified correctly after each epoch. Solid = training rows (a fixed sample of up to 2,000), dashed = held-back validation rows." />
      )}
    </>
  );
}
