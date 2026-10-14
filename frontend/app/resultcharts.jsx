"use client";

// Test-set charts of one model, from its saved predictions (/api/run/curves):
// classification -> confusion matrix, ROC, precision-recall, calibration;
// regression -> actual vs predicted, residuals, error distribution, Q-Q plot.
// Every chart downloads as SVG or PNG.
import { COLORS, Figure, Histogram, LinePlot, ScatterPlot } from "./chartkit";

function Matrix({ classes, cm, title }) {
  const n = classes.length;
  const cell = n <= 6 ? 54 : n <= 12 ? 34 : 22;
  const lab = 96, top = 30, size = lab + n * cell + 10;
  const rowMax = cm.map((row) => Math.max(1, ...row));
  return (
    <Figure title={`${title} — confusion matrix`} width={size} height={size + 6}
            note="Rows = actual, columns = predicted. Green diagonal = correct; off-diagonal = mistakes.">
      <text x={lab + (n * cell) / 2} y={12} fontSize="11" textAnchor="middle" fill="currentColor">predicted →</text>
      {classes.map((c, j) => (
        <text key={j} x={lab + j * cell + cell / 2} y={top - 4} fontSize="10" textAnchor="middle" fill="currentColor">{String(c).slice(0, 8)}</text>
      ))}
      {classes.map((c, i) => (
        <text key={i} x={lab - 6} y={top + i * cell + cell / 2 + 3} fontSize="10" textAnchor="end" fill="currentColor">{String(c).slice(0, 12)}</text>
      ))}
      {cm.map((row, i) => row.map((v, j) => {
        const t = v / rowMax[i];
        return (
          <g key={`${i}-${j}`}>
            <rect x={lab + j * cell} y={top + i * cell} width={cell - 1} height={cell - 1}
                  fill={i === j ? `rgba(22,163,74,${0.15 + 0.75 * t})` : `rgba(220,38,38,${0.08 + 0.62 * t})`} />
            <text x={lab + j * cell + cell / 2} y={top + i * cell + cell / 2 + 3} fontSize={n <= 12 ? 11 : 9}
                  textAnchor="middle" fill="currentColor">{v}</text>
          </g>
        );
      }))}
    </Figure>
  );
}

const grid = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 12 };

export default function ResultCharts({ data, title }) {
  if (!data) return null;
  if (data.task === "classification") {
    return (
      <div style={grid}>
        <Matrix classes={data.classes} cm={data.confusion} title={title} />
        {data.roc.length > 0 && (
          <LinePlot title={`${title} — ROC curve`} xLabel="false positive rate" yLabel="true positive rate" points={false}
                    xDomain={[0, 1]} yDomain={[0, 1]}
                    series={[...data.roc.map((s) => ({ label: `${s.label} (AUC ${s.auc.toFixed(3)})`, x: s.fpr, y: s.tpr })),
                             { label: "chance", x: [0, 1], y: [0, 1], dashed: true, color: "#999999" }]} />
        )}
        {data.pr.length > 0 && (
          <LinePlot title={`${title} — precision-recall`} xLabel="recall" yLabel="precision" points={false}
                    xDomain={[0, 1]} yDomain={[0, 1]}
                    series={data.pr.map((s) => ({ label: `${s.label} (AP ${s.ap.toFixed(3)})`, x: s.recall, y: s.precision }))} />
        )}
        {data.calibration?.length > 0 && (
          <LinePlot title={`${title} — calibration`} xLabel="mean predicted probability" yLabel="observed frequency"
                    xDomain={[0, 1]} yDomain={[0, 1]}
                    note="On the diagonal = the predicted probabilities can be read as real chances. Empty probability bins are left out."
                    series={[...data.calibration.map((s) => ({ label: `${s.label} (Brier ${s.brier.toFixed(3)})`, x: s.predicted, y: s.observed })),
                             { label: "perfect", x: [0, 1], y: [0, 1], dashed: true, color: "#999999" }]} />
        )}
        {!data.roc.length && <p className="note">This model has no probability or score output, so only the confusion matrix is shown.</p>}
      </div>
    );
  }
  if (data.task === "regression") {
    if (!data.actual) return <p className="note">Too few test rows to plot.</p>;
    const lo = Math.min(...data.actual, ...data.predicted), hi = Math.max(...data.actual, ...data.predicted);
    const qx = data.qq.theoretical;
    return (
      <div style={grid}>
        <ScatterPlot title={`${title} — actual vs predicted`} xLabel="actual" yLabel="predicted" line={[[lo, lo], [hi, hi]]}
                     points={data.actual.map((x, i) => ({ x, y: data.predicted[i] }))}
                     note={`On the dashed line = perfect prediction${data.shown < data.n ? ` · ${data.shown} of ${data.n} test rows shown` : ""}.`} />
        <ScatterPlot title={`${title} — residuals`} xLabel="predicted" yLabel="residual (actual − predicted)"
                     line={[[Math.min(...data.predicted), 0], [Math.max(...data.predicted), 0]]}
                     points={data.predicted.map((x, i) => ({ x, y: data.residual[i], color: COLORS[2] }))}
                     note={`Mean residual ${data.residual_mean.toFixed(3)}, sd ${data.residual_sd.toFixed(3)}. A pattern here = something the model misses.`} />
        <Histogram title={`${title} — error distribution`} edges={data.histogram.edges} counts={data.histogram.counts} xLabel="residual" />
        <ScatterPlot title={`${title} — normal Q-Q plot of residuals`} xLabel="normal quantiles" yLabel="residual quantiles"
                     line={[[qx[0], data.qq.slope * qx[0] + data.qq.intercept], [qx[qx.length - 1], data.qq.slope * qx[qx.length - 1] + data.qq.intercept]]}
                     points={qx.map((x, i) => ({ x, y: data.qq.sample[i] }))}
                     note={`On the line = normally distributed errors (r = ${data.qq.r.toFixed(3)}).`} />
      </div>
    );
  }
  return null;
}
