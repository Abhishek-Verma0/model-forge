"use client";

// Statistics tab (plan §13): descriptive statistics, frequency tables, normality,
// group comparisons, paired tests, correlation, chi-square, regression and
// questionnaire reliability. Every result keeps its assumptions, effect size and an
// automated decision the researcher must confirm; saved results feed the hypothesis
// decision table and the PDF report.
import { useEffect, useState } from "react";
import { call, fmt, fmtP, pText } from "./apiclient";
import { ScatterPlot } from "./chartkit";

const ANALYSES = [
  ["describe", "Descriptive statistics"], ["frequencies", "Frequency tables (demographics)"],
  ["normality", "Normality (Shapiro-Wilk, Q-Q)"], ["compare", "Compare groups (t-test / ANOVA / rank tests)"],
  ["paired", "Paired measurements (before / after)"], ["correlation", "Correlation (Pearson / Spearman)"],
  ["association", "Association of two categories (chi-square)"], ["regression", "Regression (linear / logistic)"],
  ["reliability", "Questionnaire reliability (Cronbach's alpha)"],
];
const TESTS = {
  compare2: [["auto", "recommend for me"], ["welch_t", "Welch's t-test"], ["student_t", "Student's t-test"], ["mann_whitney", "Mann-Whitney U"]],
  compareK: [["auto", "recommend for me"], ["anova", "one-way ANOVA"], ["welch_anova", "Welch's ANOVA"], ["kruskal", "Kruskal-Wallis"]],
  paired: [["auto", "recommend for me"], ["paired_t", "paired t-test"], ["wilcoxon", "Wilcoxon signed-rank"]],
};

function Pick({ label, options, value, onChange, empty = "choose…" }) {
  return (
    <label className="pp-field" style={{ minWidth: 180 }}><span>{label}</span>
      <select value={value || ""} onChange={(e) => onChange(e.target.value)}>
        <option value="">{empty}</option>
        {options.map((c) => <option key={c} value={c}>{c}</option>)}
      </select>
    </label>
  );
}

function Multi({ label, options, value, onChange }) {
  return (
    <div style={{ marginTop: 6 }}>
      <span className="note">{label} <button className="pp-link" type="button" onClick={() => onChange(value.length ? [] : options)}>
        {value.length ? "clear" : "all"}</button></span>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "2px 14px", maxHeight: 150, overflow: "auto" }}>
        {options.map((c) => (
          <label key={c} style={{ fontSize: 13 }}>
            <input type="checkbox" checked={value.includes(c)}
                   onChange={() => onChange(value.includes(c) ? value.filter((x) => x !== c) : [...value, c])} /> {c}
          </label>
        ))}
      </div>
    </div>
  );
}

function Effect({ e }) {
  if (!e || e.value === null || e.value === undefined) return <span className="note">n/a</span>;
  return <>{e.name} = <b>{fmt(e.value)}</b> ({e.magnitude})</>;
}

function TestResult({ r }) {
  return (
    <>
      <p style={{ margin: "4px 0" }}>
        <b>{r.test}</b>: {r.statistic_name} = {fmt(r.statistic)}{r.df !== null && r.df !== undefined ? `, df = ${Array.isArray(r.df) ? r.df.join(", ") : fmt(r.df, 2)}` : ""},
        {pText(r.p)} · <Effect e={r.effect} />
        {r.ci && <> · {Math.round(r.ci.level * 100)}% CI of the {r.ci.of}: {fmt(r.ci.low)} to {fmt(r.ci.high)}</>}
      </p>
      <p style={{ margin: "2px 0" }}>H0: {r.h0}. <b>{r.decision}</b>.</p>
      {r.reason && <p className="note" style={{ margin: "2px 0" }}>Why this test: {r.reason}.</p>}
      {r.method && <p className="note" style={{ margin: "2px 0" }}>p-value: {r.method}.</p>}
    </>
  );
}

function Result({ r }) {
  const a = r.analysis;
  if (a === "describe") return (
    <div className="tablescroll"><table>
      <thead><tr><th>Column</th><th>n</th><th>missing</th><th>mean</th><th>sd</th><th>se</th><th>median</th><th>mode</th>
        <th>min</th><th>max</th><th>range</th><th>Q1</th><th>Q3</th><th>skew</th><th>kurtosis</th><th>95% CI of mean</th></tr></thead>
      <tbody>{r.rows.map((x) => x.kind === "numeric" ? (
        <tr key={x.column}><td>{x.column}</td><td>{x.n}</td><td>{x.missing}</td><td>{fmt(x.mean)}</td><td>{fmt(x.sd)}</td><td>{fmt(x.se)}</td>
          <td>{fmt(x.median)}</td><td>{fmt(x.mode)}{x.modes > 1 ? "*" : ""}</td><td>{fmt(x.min)}</td><td>{fmt(x.max)}</td><td>{fmt(x.range)}</td>
          <td>{fmt(x.q1)}</td><td>{fmt(x.q3)}</td><td>{fmt(x.skewness)}</td><td>{fmt(x.kurtosis)}</td><td>{fmt(x.ci_low)} to {fmt(x.ci_high)}</td></tr>
      ) : (
        <tr key={x.column}><td>{x.column}</td><td>{x.n}</td><td>{x.missing}</td><td colSpan={13} className="note">
          categorical · {x.distinct} values · mode “{x.mode}” ({x.mode_percent}%) — see Frequency tables</td></tr>
      ))}</tbody></table>
      <p className="note">Kurtosis is excess kurtosis (0 = normal). * = several values share the top count (the first is shown).</p>
    </div>
  );
  if (a === "frequencies") return r.tables.map((t) => (
    <div key={t.column} className="tablescroll" style={{ marginBottom: 8 }}>
      <b>{t.column}</b> <span className="note">({t.distinct} values{t.shown < t.distinct ? `, top ${t.shown} shown` : ""}; {t.missing} missing)</span>
      <table><thead><tr><th>Value</th><th>Count</th><th>% of all</th><th>Valid %</th><th>Cumulative %</th></tr></thead>
        <tbody>{t.rows.map((x) => <tr key={x.value}><td>{x.value}</td><td>{x.count}</td><td>{x.percent}</td><td>{x.valid_percent}</td><td>{x.cumulative}</td></tr>)}</tbody></table>
    </div>
  ));
  if (a === "normality") return (
    <>
      <div className="tablescroll"><table>
        <thead><tr><th>Column</th><th>Group</th><th>n</th><th>W</th><th>p</th><th>skew</th><th>kurtosis</th><th>Looks normal?</th></tr></thead>
        <tbody>{r.rows.map((x, i) => (
          <tr key={i}><td>{x.column}</td><td>{x.group || "all"}</td><td>{x.n}</td><td>{fmt(x.shapiro?.W)}</td><td>{fmtP(x.shapiro?.p)}</td>
            <td>{fmt(x.skewness)}</td><td>{fmt(x.kurtosis)}</td><td>{x.normal === null ? "n/a" : x.normal ? "no evidence against" : "no"}{x.shapiro?.note ? ` (${x.shapiro.note})` : ""}</td></tr>
        ))}</tbody></table></div>
      <p className="note">{r.note}</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 10 }}>
        {r.rows.filter((x) => x.qq).slice(0, 8).map((x, i) => {
          const t = x.qq.theoretical;
          return <ScatterPlot key={i} width={420} height={280} title={`Q-Q: ${x.column}${x.group ? ` (${x.group})` : ""}`}
                              xLabel="normal quantiles" yLabel="sample quantiles" points={t.map((v, j) => ({ x: v, y: x.qq.sample[j] }))}
                              line={[[t[0], x.qq.slope * t[0] + x.qq.intercept], [t[t.length - 1], x.qq.slope * t[t.length - 1] + x.qq.intercept]]} />;
        })}
      </div>
    </>
  );
  if (a === "compare") return (
    <>
      <TestResult r={r} />
      <div className="tablescroll"><table><thead><tr><th>Group</th><th>n</th><th>mean</th><th>sd</th><th>median</th><th>Shapiro p</th></tr></thead>
        <tbody>{r.groups.map((g, i) => <tr key={g.group}><td>{g.group}</td><td>{g.n}</td><td>{fmt(g.mean)}</td><td>{fmt(g.sd)}</td>
          <td>{fmt(g.median)}</td><td>{fmtP(r.assumptions.normality[i]?.p)}</td></tr>)}</tbody></table></div>
      <p className="note">Levene (equal variances): F = {fmt(r.assumptions.levene.F)}, {pText(r.assumptions.levene.p)} — {r.assumptions.levene.equal_variances ? "no evidence of unequal variances" : "variances differ"}.</p>
      {r.post_hoc && (
        <div className="tablescroll"><b>Post-hoc: {r.post_hoc.test}</b>
          <table><thead><tr><th>Pair</th><th>Difference</th><th>p (adjusted)</th><th>95% CI</th></tr></thead>
            <tbody>{r.post_hoc.pairs.map((x) => <tr key={x.a + x.b}><td>{x.a} − {x.b}</td><td>{fmt(x.difference)}</td><td>{fmtP(x.p)}</td>
              <td>{fmt(x.low)} to {fmt(x.high)}</td></tr>)}</tbody></table></div>
      )}
    </>
  );
  if (a === "paired") return (
    <>
      <TestResult r={r} />
      <p className="note">n = {r.n} pairs · mean difference {fmt(r.mean_difference)} · normality of differences {pText(r.assumptions.normality_of_differences?.p)}
        {r.zero_differences_dropped ? ` · ${r.zero_differences_dropped} zero difference(s) dropped (as R does)` : ""}</p>
    </>
  );
  if (a === "correlation") return (
    <div className="tablescroll">
      <table><thead><tr><th>Pair</th><th>Method</th><th>n</th><th>r</th><th>p</th><th>95% CI</th><th>Strength</th><th>Decision</th></tr></thead>
        <tbody>{r.pairs.map((x) => <tr key={x.a + x.b}><td>{x.a} – {x.b}</td>{x.error ? <td colSpan={7} className="note">{x.error}</td> : <>
          <td>{x.method}</td><td>{x.n}</td><td>{fmt(x.r)}</td><td>{fmtP(x.p)}</td><td>{x.ci ? `${fmt(x.ci.low)} to ${fmt(x.ci.high)}` : "—"}</td>
          <td>{x.magnitude}</td><td>{x.decision}</td></>}</tr>)}</tbody></table>
      {r.reason && <p className="note">Method: {r.reason}.</p>}
    </div>
  );
  if (a === "association") return (
    <>
      <TestResult r={r} />
      {r.warnings.map((w, i) => <p key={i} className="error" style={{ margin: "4px 0" }}>{w}</p>)}
      {r.fisher && <p>Fisher&apos;s exact test: odds ratio {fmt(r.fisher.odds_ratio)}, {pText(r.fisher.p)} — <b>{r.fisher.decision}</b>.</p>}
      <div className="tablescroll"><table>
        <thead><tr><th>observed (expected)</th>{r.table.cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>{r.table.rows.map((rw, i) => <tr key={rw}><td>{rw}</td>{r.table.counts[i].map((v, j) => <td key={j}>{v} ({r.table.expected[i][j]})</td>)}</tr>)}</tbody>
      </table></div>
    </>
  );
  if (a === "regression") {
    const stat = r.kind === "linear" ? "t" : "z";
    return (
      <>
        <div className="tablescroll"><table>
          <thead><tr><th>Term</th><th>Estimate</th><th>SE</th><th>{stat}</th><th>p</th><th>95% CI</th>{r.kind === "logistic" && <th>Odds ratio (95% CI)</th>}</tr></thead>
          <tbody>{r.coefficients.map((c) => <tr key={c.term}><td>{c.term}</td><td>{fmt(c.estimate, 4)}</td><td>{fmt(c.se, 4)}</td><td>{fmt(c[stat])}</td>
            <td>{fmtP(c.p)}</td><td>{fmt(c.low, 4)} to {fmt(c.high, 4)}</td>
            {r.kind === "logistic" && <td>{fmt(c.odds_ratio)} ({fmt(c.or_low)} to {fmt(c.or_high)})</td>}</tr>)}</tbody></table></div>
        <p>{r.kind === "linear"
          ? <>R² = {fmt(r.r2)}, adjusted R² = {fmt(r.adj_r2)}, F({r.f_df.join(", ")}) = {fmt(r.f, 2)}, {pText(r.p)} · residual sd {fmt(r.sigma)}</>
          : <>P({r.outcome} = {r.event}) · McFadden pseudo R² = {fmt(r.pseudo_r2)}, likelihood-ratio χ²({r.lr_df}) = {fmt(r.lr_chi2, 2)}, {pText(r.p)}</>}
          {" "}· n = {r.n}{r.dropped_rows ? ` (${r.dropped_rows} incomplete row(s) left out)` : ""}</p>
        <p>H0: {r.h0}. <b>{r.decision}</b>.</p>
        {r.vif && Object.keys(r.vif).length > 0 && <p className="note">VIF: {Object.entries(r.vif).map(([k, v]) => `${k} ${fmt(v, 2)}`).join(" · ")}</p>}
        {(r.notes || []).map((n, i) => <p key={i} className="error" style={{ margin: "4px 0" }}>{n}</p>)}
        {r.residual_normality && <p className="note">Residual normality: Shapiro-Wilk {pText(r.residual_normality.p)}.</p>}
      </>
    );
  }
  if (a === "reliability") return (
    <>
      <p>Cronbach&apos;s alpha = <b>{fmt(r.alpha)}</b> ({r.label}) · standardised {fmt(r.standardized_alpha)} · {r.k} items · {r.n} respondents
        {r.dropped_rows ? ` (${r.dropped_rows} with a missing answer left out)` : ""} · mean inter-item r {fmt(r.mean_inter_item_r)}</p>
      {r.warnings.map((w, i) => <p key={i} className="error" style={{ margin: "4px 0" }}>{w}</p>)}
      <div className="tablescroll"><table>
        <thead><tr><th>Item</th><th>Reversed</th><th>Mean</th><th>SD</th><th>Item-total r (corrected)</th><th>Alpha if deleted</th></tr></thead>
        <tbody>{r.items.map((x) => <tr key={x.item}><td>{x.item}</td><td>{x.reversed ? "yes" : ""}</td><td>{fmt(x.mean)}</td><td>{fmt(x.sd)}</td>
          <td>{fmt(x.item_total_r)}</td><td>{fmt(x.alpha_if_deleted)}</td></tr>)}</tbody></table></div>
      <p className="note">Labels (George &amp; Mallery 2003): ≥ .9 excellent, .8 good, .7 acceptable, .6 questionable, .5 poor.</p>
    </>
  );
  return null;
}

export default function Statistics({ data, onCleaned }) {
  const info = data.profile.columns_info;
  const cols = data.profile.column_names;
  const numeric = cols.filter((c) => info[c]?.type === "numeric");
  const categorical = cols.filter((c) => info[c]?.type !== "numeric" || info[c]?.unique_values <= 10);
  const [analysis, setAnalysis] = useState("describe");
  const [p, setP] = useState({ columns: [], predictors: [], items: [], reverse: [] });
  const [alpha, setAlpha] = useState(0.05);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [history, setHistory] = useState([]);
  const [decisions, setDecisions] = useState(null);
  const [likert, setLikert] = useState(null);
  const [comp, setComp] = useState({ name: "", method: "mean", low: "", high: "" });
  const set = (k, v) => setP((x) => ({ ...x, [k]: v }));

  async function load() {
    try {
      const [h, d] = await Promise.all([call(`/api/stats/history?id=${data.id}`), call(`/api/stats/decisions?id=${data.id}`)]);
      setHistory(h); setDecisions(d);
    } catch (e) { setErr(e.message); }
  }
  useEffect(() => { load(); call(`/api/stats/likert?id=${data.id}`).then(setLikert).catch(() => {}); }, [data.id]); // eslint-disable-line react-hooks/exhaustive-deps

  async function run() {
    setBusy(true); setErr("");
    try { await call("/api/stats/run", { id: data.id, analysis, params: p, alpha: Number(alpha) }); await load(); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  async function remove(id) {
    try { await call(`/api/stats?id=${data.id}&stat=${id}`, undefined, "DELETE"); load(); } catch (e) { setErr(e.message); }
  }

  async function makeComposite() {
    setErr("");
    const ops = [];
    const lo = comp.low === "" ? null : Number(comp.low), hi = comp.high === "" ? null : Number(comp.high);
    if (p.reverse.length && (lo === null || hi === null)) { setErr("Give the scale's lowest and highest answer to reverse items."); return; }
    for (const c of p.reverse) ops.push({ op: "reverse_score", column: c, low: lo, high: hi });
    if (comp.name.trim()) ops.push({ op: "composite", columns: p.items, to: comp.name.trim(), method: comp.method });
    if (!ops.length) { setErr("Choose items to reverse and/or name the composite score."); return; }
    try { onCleaned(await call("/api/clean", { id: data.id, ops, filename: data.filename })); setP((x) => ({ ...x, reverse: [] })); }
    catch (e) { setErr(e.message); }
  }

  const form = {
    describe: <Multi label="Columns" options={cols} value={p.columns} onChange={(v) => set("columns", v)} />,
    frequencies: <Multi label="Categorical columns" options={categorical} value={p.columns} onChange={(v) => set("columns", v)} />,
    normality: <><Multi label="Numeric columns" options={numeric} value={p.columns} onChange={(v) => set("columns", v)} />
      <Pick label="Separately for each group of (optional)" options={categorical} value={p.by} onChange={(v) => set("by", v)} empty="— all rows —" /></>,
    compare: <div className="builder-row">
      <Pick label="Outcome (numeric)" options={numeric} value={p.outcome} onChange={(v) => set("outcome", v)} />
      <Pick label="Groups" options={categorical} value={p.group} onChange={(v) => set("group", v)} />
      <label className="pp-field"><span>Test</span><select value={p.test || "auto"} onChange={(e) => set("test", e.target.value)}>
        {(p.group && info[p.group]?.unique_values === 2 ? TESTS.compare2 : p.group && info[p.group]?.unique_values > 2 ? TESTS.compareK : TESTS.compare2.slice(0, 1))
          .map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label></div>,
    paired: <div className="builder-row">
      <Pick label="First measurement" options={numeric} value={p.before} onChange={(v) => set("before", v)} />
      <Pick label="Second measurement" options={numeric} value={p.after} onChange={(v) => set("after", v)} />
      <label className="pp-field"><span>Test</span><select value={p.test || "auto"} onChange={(e) => set("test", e.target.value)}>
        {TESTS.paired.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label></div>,
    correlation: <><Multi label="Numeric columns (every pair is tested)" options={numeric} value={p.columns} onChange={(v) => set("columns", v)} />
      <label className="pp-field" style={{ maxWidth: 220 }}><span>Method</span><select value={p.method || "auto"} onChange={(e) => set("method", e.target.value)}>
        <option value="auto">recommend for me</option><option value="pearson">Pearson</option><option value="spearman">Spearman</option></select></label></>,
    association: <div className="builder-row">
      <Pick label="Rows" options={categorical} value={p.row} onChange={(v) => set("row", v)} />
      <Pick label="Columns" options={categorical} value={p.col} onChange={(v) => set("col", v)} /></div>,
    regression: <>
      <div className="builder-row">
        <label className="pp-field"><span>Kind</span><select value={p.kind || "linear"} onChange={(e) => set("kind", e.target.value)}>
          <option value="linear">linear (numeric outcome)</option><option value="logistic">logistic (two-valued outcome)</option></select></label>
        <Pick label="Outcome" options={(p.kind || "linear") === "linear" ? numeric : cols.filter((c) => info[c]?.unique_values === 2)}
              value={p.outcome} onChange={(v) => set("outcome", v)} />
        {p.kind === "logistic" && p.outcome && (
          <Pick label="Event (the outcome value modelled)" options={(info[p.outcome]?.top_values || []).map((t) => t.value)}
                value={p.event} onChange={(v) => set("event", v)} empty="second value (default)" />
        )}
      </div>
      <Multi label="Predictors" options={cols.filter((c) => c !== p.outcome)} value={p.predictors} onChange={(v) => set("predictors", v)} />
    </>,
    reliability: <>
      {likert?.scales?.length > 0 && (
        <p className="note">Possible Likert scales: {likert.scales.map((s) => (
          <button key={s.range} className="pp-link" type="button" onClick={() => set("items", s.columns)}>
            {s.columns.join(", ")} ({s.range})</button>))} — {likert.note}</p>
      )}
      <Multi label="Items of one scale" options={numeric} value={p.items} onChange={(v) => set("items", v)} />
      <Multi label="Reverse-worded items (scored low + high − answer)" options={p.items} value={p.reverse} onChange={(v) => set("reverse", v)} />
      <div className="builder-row">
        <label className="pp-field" style={{ maxWidth: 140 }}><span>Scale lowest answer</span>
          <input type="number" value={comp.low} onChange={(e) => { setComp({ ...comp, low: e.target.value }); set("low", e.target.value === "" ? null : Number(e.target.value)); }} /></label>
        <label className="pp-field" style={{ maxWidth: 140 }}><span>Scale highest answer</span>
          <input type="number" value={comp.high} onChange={(e) => { setComp({ ...comp, high: e.target.value }); set("high", e.target.value === "" ? null : Number(e.target.value)); }} /></label>
      </div>
      <div className="panel" style={{ padding: 10, marginTop: 8 }}>
        <b>Save to the data</b> <span className="note">(a cleaning step: kept, replayed on new rows, usable in every tab)</span>
        <div className="builder-row" style={{ alignItems: "flex-end" }}>
          <label className="pp-field"><span>Composite score name (optional)</span>
            <input value={comp.name} placeholder="e.g. wellbeing" onChange={(e) => setComp({ ...comp, name: e.target.value })} /></label>
          <label className="pp-field" style={{ maxWidth: 130 }}><span>Combine by</span><select value={comp.method} onChange={(e) => setComp({ ...comp, method: e.target.value })}>
            <option value="mean">mean of answered items</option><option value="sum">sum (all items answered)</option></select></label>
          <button className="ghost" type="button" onClick={makeComposite}>Apply reverse scoring / composite</button>
        </div>
      </div>
    </>,
  }[analysis];

  return (
    <section>
      <p className="note" style={{ marginTop: 0 }}>Runs on the current (cleaned) table. The app recommends a test from the variable types, the number of
        groups and normality; you remain responsible for confirming the assumptions and interpreting the result.</p>
      <div className="panel" style={{ padding: 14, marginBottom: 12 }}>
        <div className="builder-row">
          <label className="pp-field" style={{ minWidth: 300 }}><span>Analysis</span>
            <select value={analysis} onChange={(e) => setAnalysis(e.target.value)}>
              {ANALYSES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
            </select>
          </label>
          <label className="pp-field" style={{ maxWidth: 160 }}><span>Significance level (α)</span>
            <input type="number" step="0.01" min="0.001" max="0.2" value={alpha} onChange={(e) => setAlpha(e.target.value)} /></label>
        </div>
        {form}
        <button className="primary" style={{ marginTop: 10 }} disabled={busy} onClick={run}>{busy ? "Running…" : "Run analysis"}</button>
      </div>
      {err && <div className="error">{err}</div>}

      {decisions?.rows?.length > 0 && (
        <div className="panel tablescroll" style={{ padding: 12, marginBottom: 12 }}>
          <b>Hypothesis decision table</b>
          <table><thead><tr><th>Hypothesis</th><th>Test</th><th>p</th><th>Effect size</th><th>Decision</th></tr></thead>
            <tbody>{decisions.rows.map((x, i) => <tr key={i}><td>{x.hypothesis}</td><td>{x.test}</td><td>{fmtP(x.p)}</td><td>{x.effect}</td><td>{x.decision}</td></tr>)}</tbody>
          </table>
          <p className="note">{decisions.review}</p>
        </div>
      )}
      {history.map((r) => (
        <div key={r.id} className="panel" style={{ padding: 12, marginBottom: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
            <b>{r.title}</b>
            <span className="note">{new Date(r.created * 1000).toLocaleString()}{" "}
              <button className="ghost" onClick={() => remove(r.id)}>Remove</button></span>
          </div>
          <Result r={r} />
          {r.review && <p className="note" style={{ margin: "6px 0 0" }}>{r.review}</p>}
        </div>
      ))}
    </section>
  );
}
