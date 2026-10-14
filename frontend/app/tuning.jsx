"use client";

// Hyperparameter search settings (plan §10): random or grid search over ranges that
// are our starting point (shown, editable). TuneTrials lists what each trial scored.
import { fmt } from "./apiclient";

const POINTS = 3; // the backend's default grid: low, middle, high

function gridValues(s) {
  if (s.kind === "choice") return s.choices;
  const vals = [];
  for (let i = 0; i < POINTS; i++) {
    const t = i / (POINTS - 1);
    const v = s.log && s.low > 0 ? s.low * (s.high / s.low) ** t : s.low + (s.high - s.low) * t;
    vals.push(s.kind === "int" ? Math.round(v) : Number(v.toPrecision(6)));
  }
  return [...new Set(vals)];
}

export function gridCount(space, grid) {
  return Object.entries(space || {}).reduce((n, [p, s]) => n * ((grid?.[p] || gridValues(s)).length || 1), 1);
}

export default function TuneSettings({ models, picked, spaces, value, onChange, disabled }) {
  const { method, nTrials, space, grid } = value;
  const set = (patch) => onChange({ ...value, ...patch });
  const label = (k) => models.find((m) => m.key === k)?.label || k;
  const paramLabel = (k, p) => models.find((m) => m.key === k)?.params.find((x) => x.name === p)?.label || p;
  const eff = (k) => Object.fromEntries(Object.entries(spaces[k]?.space || {}).map(([p, s]) => [p, { ...s, ...(space[k]?.[p] || {}) }]));

  return (
    <div className="panel" style={{ padding: 14, marginBottom: 12 }}>
      <p style={{ margin: "0 0 6px", fontWeight: 600 }}>Hyperparameter search</p>
      <p className="note" style={{ margin: "0 0 8px" }}>
        Every candidate is cross-validated on the same folds; the best settings per model are then trained and scored once on
        the locked test set. The ranges below are our starting point — change any. Epochs and layer sizes are not searched
        (early stopping decides epochs).
      </p>
      <div className="builder-row">
        <label className="pp-field" style={{ maxWidth: 180 }}><span>Method</span>
          <select value={method} disabled={disabled} onChange={(e) => set({ method: e.target.value })}>
            <option value="random">random search</option><option value="grid">grid search</option>
          </select>
        </label>
        {method === "random" && (
          <label className="pp-field" style={{ maxWidth: 140 }}><span>Trials per model</span>
            <input type="number" min="1" max="500" value={nTrials} disabled={disabled} onChange={(e) => set({ nTrials: e.target.value })} />
          </label>
        )}
      </div>
      {picked.map((k) => {
        const sp = eff(k);
        const n = gridCount(sp, grid[k]);
        return (
          <details key={k} style={{ marginTop: 8 }} open={picked.length <= 2}>
            <summary><b>{label(k)}</b>{" "}
              <span className="note">{Object.keys(sp).length ? (method === "grid" ? `${n.toLocaleString()} combination(s) x folds` :
                `${nTrials} trial(s) x folds`) : "nothing to search - trained with its parameters"}</span>
            </summary>
            {Object.entries(sp).map(([p, s]) => (
              <div key={p} style={{ display: "grid", gridTemplateColumns: "minmax(140px, 200px) 1fr", gap: 10, alignItems: "center",
                                    padding: "4px 0", borderTop: "1px solid rgba(0,0,0,0.06)" }}>
                <span>{paramLabel(k, p)}</span>
                {s.kind === "choice" ? (
                  <span style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {(spaces[k].space[p].choices).map((c) => (
                      <label key={c} style={{ fontSize: 13 }}>
                        <input type="checkbox" disabled={disabled} checked={s.choices.includes(c)}
                               onChange={() => set({ space: { ...space, [k]: { ...space[k], [p]: { kind: "choice",
                                 choices: s.choices.includes(c) ? s.choices.filter((x) => x !== c) : [...s.choices, c] } } } })} /> {c}
                      </label>
                    ))}
                  </span>
                ) : method === "grid" ? (
                  <input disabled={disabled} placeholder={gridValues(s).join(", ")} defaultValue={(grid[k]?.[p] || []).join(", ")}
                         onChange={(e) => {
                           const vals = e.target.value.split(",").map((x) => x.trim()).filter(Boolean).map(Number).filter((x) => !Number.isNaN(x));
                           set({ grid: { ...grid, [k]: { ...grid[k], [p]: vals.length ? vals : undefined } } });
                         }} title="values to try, comma-separated (empty = the low / middle / high shown)" />
                ) : (
                  <span style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                    <input type="number" style={{ width: 100 }} disabled={disabled} value={s.low}
                           onChange={(e) => set({ space: { ...space, [k]: { ...space[k], [p]: { ...s, low: Number(e.target.value) } } } })} />
                    to
                    <input type="number" style={{ width: 100 }} disabled={disabled} value={s.high}
                           onChange={(e) => set({ space: { ...space, [k]: { ...space[k], [p]: { ...s, high: Number(e.target.value) } } } })} />
                    {s.log && <span className="note">log scale</span>}
                  </span>
                )}
              </div>
            ))}
          </details>
        );
      })}
    </div>
  );
}

export function TuneTrials({ tuning, metrics }) {
  if (!tuning?.models) return null;
  const meta = Object.fromEntries(metrics.map((m) => [m.key, m]));
  return (
    <div className="panel tablescroll" style={{ padding: 12, marginTop: 8 }}>
      <p style={{ margin: "0 0 6px" }}><b>Search trials</b> — {tuning.method} search, scored by {meta[tuning.primary_metric]?.label} (CV mean).
        The CV score of the winner is slightly optimistic (the same folds chose it); its test score is the honest number.</p>
      {Object.entries(tuning.models).map(([k, t]) => (
        <details key={k} open style={{ marginTop: 6 }}>
          <summary><b>{t.label}</b> — {t.trials.length} trial(s){t.best ? `, best ${fmt(t.best.cv_mean)}` : ""}</summary>
          <table>
            <thead><tr><th>#</th><th>Settings</th><th>CV mean ± sd</th><th>Seconds</th></tr></thead>
            <tbody>
              {t.trials.map((x, i) => (
                <tr key={i} style={t.best === x || JSON.stringify(t.best?.params) === JSON.stringify(x.params) ? { background: "rgba(22,163,74,0.08)" } : undefined}>
                  <td>{i + 1}</td>
                  <td className="note">{Object.entries(x.params).map(([p, v]) => `${p}=${typeof v === "number" ? Number(v.toPrecision(4)) : v}`).join(" · ")}</td>
                  <td>{x.error ? <span className="note">{x.error}</span> : `${fmt(x.cv_mean)} ± ${fmt(x.cv_sd)}`}</td>
                  <td>{x.seconds}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      ))}
    </div>
  );
}
