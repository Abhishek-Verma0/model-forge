"use client";

// Common target-column names, checked before falling back to a shape heuristic.
// Our starting point: only picks the pre-selected suggestion; the user always chooses.
const NAME_HINTS = [
  "target", "label", "class", "outcome", "result", "churn", "survived",
  "default", "fraud", "fallen", "fall", "diagnosis", "disease", "y",
];

// A best-guess target the user can override. Not ML magic -- a cheap heuristic:
// skip IDs / constants, prefer a name hint, else the last low-cardinality
// categorical/boolean column, else just the last column.
export function suggestTarget(profile, report, semType, info) {
  const idCols = new Set(report.checks.id_columns.columns.map((c) => c.column));
  const cols = profile.column_names.filter((n) => !idCols.has(n) && !info[n].is_constant);

  const byName = cols.find((n) => NAME_HINTS.some((h) => n.toLowerCase().includes(h)));
  if (byName) return byName;

  const catish = cols.filter(
    (n) => ["boolean", "categorical"].includes(semType[n]) && info[n].missing_percent < 50
  );
  if (catish.length) return catish[catish.length - 1];

  return cols[cols.length - 1] || profile.column_names[profile.column_names.length - 1];
}

export default function TargetPicker({ data, target, setTarget }) {
  const { profile, report } = data;
  const info = profile.columns_info;
  const semType = {};
  for (const c of report.checks.column_types.columns) semType[c.column] = c.detected_type;

  const suggested = suggestTarget(profile, report, semType, info);

  const t = info[target];
  const type = semType[target];
  const problem = !target
    ? "none"
    : type === "numeric" ? "regression" : type === "boolean" || type === "categorical" ? "classification" : "unusual";

  return (
    <section>
      <p className="sectitle">Target &amp; task</p>
      <div className="panel" style={{ padding: 16 }}>
        <div className="target-row">
          <label>
            <b>Which column do you want to predict?</b>&nbsp;
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">— choose target —</option>
              {profile.column_names.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          {problem !== "none" && <span className="tag">{problem}</span>}
          <span className="target-note">suggested: {suggested}</span>
        </div>

        {problem === "none" && (
          <p className="target-note">
            Pick your outcome column above — it drives the AI’s cleaning &amp; preprocessing suggestions,
            the chat, and the train/test split.
          </p>
        )}
        {problem === "classification" && <ClassBalance info={t} />}
        {problem === "regression" && <RegressionInfo info={t} />}
        {problem === "unusual" && (
          <p className="target-note">
            “{target}” is {type} — not a typical target. Pick a categorical column for classification, or a
            numeric one for regression.
          </p>
        )}
      </div>
    </section>
  );
}

function ClassBalance({ info }) {
  const tv = info.top_values || [];
  if (!tv.length) return <p className="target-note">No class values to show.</p>;

  const counts = tv.map((c) => c.count);
  const max = Math.max(...counts, 1);
  const ratio = Math.min(...counts) > 0 ? Math.max(...counts) / Math.min(...counts) : Infinity;
  const imbalanced = ratio >= 3;
  const partial = info.unique_values > tv.length;

  return (
    <div style={{ marginTop: 10 }}>
      <div className="target-note" style={{ marginBottom: 8 }}>
        {info.unique_values} class{info.unique_values === 1 ? "" : "es"}
        {partial ? " (top 10 shown)" : ""} · imbalance{" "}
        {ratio === Infinity ? "—" : ratio.toFixed(1) + ":1"}{" "}
        {imbalanced ? (
          <span style={{ color: "var(--warn)" }}>· imbalanced</span>
        ) : (
          <span style={{ color: "var(--good)" }}>· balanced</span>
        )}
      </div>
      {tv.map((c, i) => (
        <div className="hbar" key={i}>
          <div className="hbar-label" title={String(c.value)}>
            {String(c.value)}
          </div>
          <div className="hbar-track">
            <div className="hbar-fill" style={{ width: (c.count / max) * 100 + "%" }} />
          </div>
          <div className="hbar-num">
            {c.count.toLocaleString()} ({c.percent}%)
          </div>
        </div>
      ))}
    </div>
  );
}

function RegressionInfo({ info }) {
  const s = info.statistics;
  if (!s) return <p className="target-note">No statistics for this column.</p>;
  const f = (v) => (typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 3 }) : v);
  return (
    <div className="target-note" style={{ marginTop: 6 }}>
      Regression target — range {f(s.min)} … {f(s.max)}, mean {f(s.mean)}, std {f(s.std)}.
      {info.unique_values <= 20 && (
        <div style={{ marginTop: 4 }}>
          note: only {info.unique_values} distinct values — could be treated as classification instead.
        </div>
      )}
    </div>
  );
}
