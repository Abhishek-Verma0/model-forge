"use client";

import { useEffect, useRef, useState } from "react";

import Charts from "./charts";
import TargetPicker from "./target";
import EdaGallery from "./eda";
import ChartBuilder from "./builder";
import Clean from "./clean";
import Preprocess from "./preprocess";
import ChatPanel from "./chat";
import Landing from "./landing";
import DatasetUpload from "./upload";
import DatasetPreview from "./preview";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const MAX_MB = 200;
const ALLOWED = [".csv", ".xlsx", ".xls"];

// What each detected issue means for preprocessing. Read-only for now — the
// backend has no "apply" endpoint yet, so these are recommendations, not buttons.
const ISSUE_META = {
  missing_values: { label: "Missing values", rec: "Impute (mean / median / mode / KNN) or drop" },
  duplicate_rows: { label: "Duplicate rows", rec: "Preview and remove duplicates" },
  outliers: { label: "Outliers", rec: "IQR clip, z-score, or winsorize" },
  constant_columns: { label: "Constant columns", rec: "Drop (zero variance, no signal)" },
  id_columns: { label: "Possible ID columns", rec: "Exclude from modeling (leaks / no signal)" },
  high_cardinality: { label: "High-cardinality columns", rec: "Hash / target-encode, or drop" },
  value_issues: { label: "Value issues", rec: "Convert numbers-stored-as-text, review negatives/zeros" },
  consistency_issues: { label: "Inconsistent categories", rec: "Normalize spellings (Male / male / \" Male\")" },
};

const VALUE_ISSUE_LABEL = {
  negative_values: "Negative values",
  zero_values: "Zero values",
  numeric_stored_as_text: "Numbers stored as text (ratio)",
  whitespace_values: "Leading/trailing whitespace",
};

const num = (v) =>
  typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 3 }) : v ?? "—";

export default function Page() {
  const [viewMode, setViewMode] = useState("landing");
  const [activeTab, setActiveTab] = useState("upload");
  const [file, setFile] = useState(null);
  const [datasetName, setDatasetName] = useState("");
  const [taskType, setTaskType] = useState("auto");
  const [target, setTarget] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  function pick(f) {
    setError("");
    if (!f) return;
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED.includes(ext)) {
      setError(`Unsupported file type "${ext}". Use CSV or Excel (${ALLOWED.join(", ")}).`);
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`File is ${(f.size / 1024 / 1024).toFixed(0)} MB — the limit is ${MAX_MB} MB.`);
      return;
    }
    setFile(f);
    if (!datasetName) {
      const base = f.name.replace(/\.[^/.]+$/, "");
      setDatasetName(base.replace(/[-_]/g, " "));
    }
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/api/analyze`, { method: "POST", body: form });
      if (!res.ok) {
        let detail = `Server error ${res.status}`;
        try {
          const j = await res.json();
          if (j.detail) detail = j.detail;
        } catch {}
        throw new Error(detail);
      }
      const data = await res.json();
      setResult(data);
      setActiveTab("preview");
    } catch (e) {
      setError(
        e instanceof TypeError
          ? `Cannot reach the backend at ${API}. Is it running? (uvicorn app:app --reload)`
          : e.message
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setDatasetName("");
    setTarget("");
    setError("");
    setActiveTab("upload");
  }

  if (viewMode === "landing") {
    return <Landing onLaunchStudio={() => setViewMode("studio")} />;
  }

  return (
    <div className="studio-app-shell">
      {/* Sidebar matching reference dashboard */}
      <aside className="studio-sidebar">
        <div className="sidebar-brand" onClick={() => setViewMode("landing")}>
          <div className="brand-icon-box small">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M12 2v8M4.93 10.93 9.17 15.17M2 18h20M20 18v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2M7 14h10l-2-4H9l-2 4Z" />
            </svg>
          </div>
          <div className="sidebar-brand-text">
            <span className="sidebar-brand-name">Model Forge</span>
            <span className="sidebar-brand-tag">Studio</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="sidebar-nav-section">INGESTION</div>
          <button
            type="button"
            className={`sidebar-nav-btn ${activeTab === "upload" ? "active" : ""}`}
            onClick={() => setActiveTab("upload")}
          >
            <span className="nav-btn-icon">📂</span>
            <span className="nav-btn-text">Upload Dataset</span>
            {file && <span className="nav-pill-dot green" title="Dataset selected"></span>}
          </button>

          <button
            type="button"
            className={`sidebar-nav-btn ${activeTab === "preview" ? "active" : ""}`}
            onClick={() => result && setActiveTab("preview")}
            disabled={!result}
          >
            <span className="nav-btn-icon">👁️</span>
            <span className="nav-btn-text">Dataset Preview</span>
            {!result && <span className="nav-lock-badge">🔒</span>}
          </button>

          <div className="sidebar-nav-section">ML PIPELINE</div>
          <button
            type="button"
            className={`sidebar-nav-btn ${activeTab === "insights" ? "active" : ""}`}
            onClick={() => result && setActiveTab("insights")}
            disabled={!result}
          >
            <span className="nav-btn-icon">🛡️</span>
            <span className="nav-btn-text">Quality Audit</span>
            {!result && <span className="nav-lock-badge">🔒</span>}
          </button>

          <button
            type="button"
            className={`sidebar-nav-btn ${activeTab === "cleaning" ? "active" : ""}`}
            onClick={() => result && setActiveTab("cleaning")}
            disabled={!result}
          >
            <span className="nav-btn-icon">🧹</span>
            <span className="nav-btn-text">Data Cleaning</span>
            {!result && <span className="nav-lock-badge">🔒</span>}
          </button>

          <button
            type="button"
            className={`sidebar-nav-btn ${activeTab === "eda" ? "active" : ""}`}
            onClick={() => result && setActiveTab("eda")}
            disabled={!result}
          >
            <span className="nav-btn-icon">📊</span>
            <span className="nav-btn-text">Visual EDA</span>
            {!result && <span className="nav-lock-badge">🔒</span>}
          </button>

          <button
            type="button"
            className={`sidebar-nav-btn ${activeTab === "preprocessing" ? "active" : ""}`}
            onClick={() => result && setActiveTab("preprocessing")}
            disabled={!result}
          >
            <span className="nav-btn-icon">⚙️</span>
            <span className="nav-btn-text">ML Preprocessing</span>
            {!result && <span className="nav-lock-badge">🔒</span>}
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-profile-card">
            <div className="profile-avatar">👤</div>
            <div className="profile-meta">
              <span className="profile-name">ML Workspace</span>
              <span className="profile-status">● Session Active</span>
            </div>
          </div>

          <button
            type="button"
            className="btn-sidebar-back"
            onClick={() => setViewMode("landing")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
            <span>Back to Home</span>
          </button>
        </div>
      </aside>

      {/* Main Studio Content Area */}
      <main className="studio-main-viewport">
        {/* Top Header */}
        <header className="studio-header-strip">
          <div className="header-breadcrumbs">
            <span className="bc-home" onClick={() => setViewMode("landing")}>Model Forge</span>
            <span className="bc-divider">/</span>
            <span className="bc-active">
              {activeTab === "upload" || !result
                ? "Dataset Ingestion"
                : `${result.filename} · ${
                    activeTab === "preview"
                      ? "Dataset Preview"
                      : activeTab === "insights"
                      ? "Quality Audit"
                      : activeTab === "cleaning"
                      ? "Data Cleaning"
                      : activeTab === "eda"
                      ? "Visual EDA"
                      : "ML Preprocessing"
                  }`}
            </span>
          </div>

          <div className="header-status-area">
            {result && (
              <div className="loaded-dataset-chip">
                <span>📄 {result.filename}</span>
                <span className="chip-count">({result.profile?.rows} rows)</span>
              </div>
            )}
            <div className="server-status-pill">
              <span className="status-dot"></span>
              <span>Local API (8000)</span>
            </div>
          </div>
        </header>

        {/* Content Container */}
        <div className="studio-view-canvas">
          {activeTab === "upload" || !result ? (
            <DatasetUpload
              file={file}
              setFile={pick}
              datasetName={datasetName}
              setDatasetName={setDatasetName}
              taskType={taskType}
              setTaskType={setTaskType}
              loading={loading}
              error={error}
              onAnalyze={analyze}
              onReset={reset}
            />
          ) : activeTab === "preview" ? (
            <DatasetPreview
              data={result}
              onProceed={() => setActiveTab("insights")}
              onClean={() => setActiveTab("cleaning")}
              target={target}
              setTarget={setTarget}
            />
          ) : (
            <div className="wrap wide" style={{ padding: "16px 0 60px" }}>
              <Report
                data={result}
                setData={setResult}
                onReset={reset}
                tab={activeTab}
                setTab={setActiveTab}
                target={target}
                setTarget={setTarget}
              />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------

function Report({ data, setData, onReset, tab: activeTabProp, setTab: setActiveTabProp, target: targetProp, setTarget: setTargetProp }) {
  const { profile, report, filename } = data;
  const info = profile.columns_info;
  const checks = report.checks;

  // detector's semantic type per column
  const semType = {};
  for (const c of checks.column_types.columns) semType[c.column] = c.detected_type;

  const [localTarget, setLocalTarget] = useState("");
  const target = targetProp !== undefined ? targetProp : localTarget;
  const setTarget = setTargetProp || setLocalTarget;
  const task = !target
    ? ""
    : semType[target] === "numeric"
      ? "regression"
      : "classification"; // boolean/categorical (and anything else) -> classification

  const [localTab, setLocalTab] = useState("insights");
  const tab = activeTabProp || localTab;
  const setTab = setActiveTabProp || setLocalTab;

  // LLM plan (clean + preprocess suggestions). Optional overlay: on failure the
  // tabs fall back to rule-based defaults. Fetched once per dataset.
  const [plan, setPlan] = useState(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [planErr, setPlanErr] = useState("");
  const [livePlan, setLivePlan] = useState(null); // user's edited plan -> chat memory
  const [rev, setRev] = useState(0);               // bumps when data is cleaned, to reset child state

  useEffect(() => {
    let cancel = false;
    setPlanLoading(true);
    setPlanErr("");
    fetch(`${API}/api/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: data.id, target, task }),
    })
      .then(async (r) => {
        if (!r.ok) {
          let d = `Error ${r.status}`;
          try { const j = await r.json(); if (j.detail) d = j.detail; } catch {}
          throw new Error(d);
        }
        return r.json();
      })
      .then((p) => { if (!cancel) setPlan(p); })
      .catch((e) => { if (!cancel) setPlanErr(e instanceof TypeError ? "backend unreachable" : e.message); })
      .finally(() => { if (!cancel) setPlanLoading(false); });
    return () => { cancel = true; };
    // refetch when the user picks/changes the target so the AI refines its plan
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.id, target]);

  // Cleaning replaces the stored frame; refresh everything with the cleaned data.
  function onCleaned(fresh) {
    setData(fresh);
    setRev((r) => r + 1);
    setTab("eda");
  }

  const { score, grade, deductions } = report.quality_score;
  const numericCols = profile.column_names.filter((n) => info[n].type === "numeric");

  function download() {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename.replace(/\.[^.]+$/, "") + ".report.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <>
      <div className="brand" style={{ marginBottom: 8, justifyContent: "space-between", width: "100%" }}>
        <h1 style={{ fontSize: 18 }}>{filename}</h1>
        <span style={{ display: "flex", gap: 8 }}>
          <button className="ghost" style={{ padding: "4px 12px" }} onClick={download}>
            Download report.json
          </button>
          <button className="ghost" style={{ padding: "4px 12px" }} onClick={onReset}>
            ← New file
          </button>
        </span>
      </div>

      <div className="report-row">
        <div className="report-main">

      {/* Target: the user's choice, always visible -- it drives the AI plan,
          the chat, and preprocessing. */}
      <TargetPicker data={data} target={target} setTarget={setTarget} />

      {/* Tabs — flow order: audit -> clean -> explore -> preprocess */}
      <div className="tabs">
        <button className={tab === "insights" ? "tab on" : "tab"} onClick={() => setTab("insights")}>
          Insights
        </button>
        <button className={tab === "cleaning" ? "tab on" : "tab"} onClick={() => setTab("cleaning")}>
          Cleaning
        </button>
        <button className={tab === "eda" ? "tab on" : "tab"} onClick={() => setTab("eda")}>
          EDA
        </button>
        <button
          className={tab === "preprocessing" ? "tab on" : "tab"}
          onClick={() => setTab("preprocessing")}
        >
          Preprocessing
        </button>
      </div>

      {/* ===== INSIGHTS TAB ===== */}
      <div hidden={tab !== "insights"}>

      {/* Summary */}
      <section>
        <div className="cards">
          <Stat k="Rows" v={num(profile.rows)} />
          <Stat k="Columns" v={profile.columns} />
          <Stat
            k="Duplicate rows"
            v={`${num(profile.duplicate_rows)} (${checks.duplicate_rows.summary.duplicate_percent}%)`}
          />
          <Stat k="Missing cells" v={num(checks.missing_values.summary.total_missing_cells)} />
          <Stat k="Quality score" v={score + "/100"} color={scoreColor(score)}
                note={deductions.length
                  ? `${grade} — ${deductions.map((d) => `−${d.points} ${d.reason}`).join(", ")}`
                  : "nothing wrong found"} />
          <Stat
            k="Issue types"
            v={`${report.total_issue_types_found} / ${Object.keys(report.issues_found).length}`}
          />
        </div>
      </section>

      {/* Column types roll-up */}
      <Section title="Column types">
        <Table
          head={["Type", "Columns"]}
          rows={Object.entries(checks.column_types.summary.type_counts).map(([t, n]) => [t, num(n)])}
        />
      </Section>

      {/* Master column table */}
      <Section title={`Columns (${profile.columns})`}>
        <Table
          head={["Column", "Type", "Dtype", "Missing", "Unique", "Constant"]}
          rows={profile.column_names.map((n) => {
            const c = info[n];
            return [
              n,
              <span className="tag">{semType[n] || c.type}</span>,
              <span style={{ color: "var(--muted)" }}>{c.data_type}</span>,
              c.missing_values > 0 ? `${num(c.missing_values)} (${c.missing_percent}%)` : "—",
              num(c.unique_values),
              c.is_constant ? "yes" : "—",
            ];
          })}
        />
      </Section>

      {/* Descriptive stats — numeric columns */}
      {numericCols.length > 0 && (
        <Section title="Numeric statistics">
          <Table
            head={["Column", "Min", "Max", "Mean", "Median", "Std", "Q1", "Q3"]}
            rows={numericCols.map((n) => {
              const s = info[n].statistics;
              return [n, num(s.min), num(s.max), num(s.mean), num(s.median), num(s.std), num(s.q1), num(s.q3)];
            })}
          />
        </Section>
      )}

      {/* Per-issue detail tables */}
      {checks.missing_values.has_missing && (
        <Section title={`Missing values (${checks.missing_values.columns.length})`}>
          <Table
            head={["Column", "Missing", "%", "Nulls", "Empty strings"]}
            rows={checks.missing_values.columns.map((m) => [
              m.column,
              num(m.missing_count),
              m.missing_percent + "%",
              num(m.null_count),
              num(m.empty_string_count),
            ])}
          />
        </Section>
      )}

      {checks.outliers.has_outliers && (
        <Section title={`Outliers — IQR (${checks.outliers.columns.length})`}>
          <Table
            head={["Column", "Outliers", "%", "Lower", "Upper", "Min", "Max"]}
            rows={checks.outliers.columns.map((o) => [
              o.column,
              num(o.outlier_count),
              o.outlier_percent + "%",
              num(o.lower_bound),
              num(o.upper_bound),
              num(o.min_value),
              num(o.max_value),
            ])}
          />
        </Section>
      )}

      {checks.duplicate_rows.has_duplicates && (
        <Section title="Duplicate rows">
          <Table
            head={["Duplicate rows", "%", "Rows involved", "Unique rows"]}
            rows={[
              [
                num(checks.duplicate_rows.summary.duplicate_rows),
                checks.duplicate_rows.summary.duplicate_percent + "%",
                num(checks.duplicate_rows.summary.rows_involved_in_duplication),
                num(checks.duplicate_rows.summary.unique_rows),
              ],
            ]}
          />
        </Section>
      )}

      {checks.constant_columns.has_constant_columns && (
        <Section title={`Constant columns (${checks.constant_columns.columns.length})`}>
          <Table
            head={["Column", "Value", "All missing"]}
            rows={checks.constant_columns.columns.map((c) => [
              c.column,
              c.all_missing ? "—" : c.constant_value,
              c.all_missing ? "yes" : "—",
            ])}
          />
        </Section>
      )}

      {checks.id_columns.has_id_columns && (
        <Section title={`Possible ID columns (${checks.id_columns.columns.length})`}>
          <Table
            head={["Column", "Distinct", "Unique ratio", "Fully unique"]}
            rows={checks.id_columns.columns.map((c) => [
              c.column,
              num(c.distinct_values),
              c.unique_ratio,
              c.is_fully_unique ? "yes" : "—",
            ])}
          />
        </Section>
      )}

      {checks.high_cardinality.has_high_cardinality && (
        <Section title={`High-cardinality columns (${checks.high_cardinality.columns.length})`}>
          <Table
            head={["Column", "Distinct", "Unique ratio"]}
            rows={checks.high_cardinality.columns.map((c) => [
              c.column,
              num(c.distinct_values),
              c.unique_ratio,
            ])}
          />
        </Section>
      )}

      {checks.value_issues.has_value_issues && (
        <Section title={`Value issues (${checks.value_issues.columns.length})`}>
          <Table
            head={["Column", "Issue", "Value"]}
            rows={checks.value_issues.columns.flatMap((c) =>
              Object.entries(c.issues).map(([k, v]) => [c.column, VALUE_ISSUE_LABEL[k] || k, num(v)])
            )}
          />
        </Section>
      )}

      {checks.consistency_issues.has_consistency_issues && (
        <Section title={`Inconsistent categories (${checks.consistency_issues.columns.length})`}>
          <Table
            head={["Column", "Groups", "Examples"]}
            rows={checks.consistency_issues.columns.map((c) => [
              c.column,
              num(c.inconsistent_groups),
              Object.values(c.examples)
                .map((sp) => sp.join(" / "))
                .join("     ·     "),
            ])}
          />
        </Section>
      )}

      {/* Recommendations */}
      <Section title="Recommended preprocessing">
        <div className="panel">
          {Object.entries(report.issues_found).filter(([, f]) => f).length === 0 && (
            <p style={{ padding: "12px 16px", color: "var(--good)", margin: 0 }}>
              No data-quality issues detected.
            </p>
          )}
          {Object.entries(report.issues_found)
            .filter(([, found]) => found)
            .map(([key]) => (
              <div className="issue" key={key}>
                <div className="dot" style={{ background: "var(--warn)" }} />
                <div className="body">
                  <strong>
                    {ISSUE_META[key].label} — {issueDetail(key, checks)}
                  </strong>
                  <p>{ISSUE_META[key].rec}</p>
                </div>
              </div>
            ))}
        </div>
        <p className="note">
          Recommendations are read-only for now — applying them (impute, encode, scale, balance) is
          the next backend milestone.
        </p>
      </Section>

      {/* Raw JSON */}
      <Section title="Raw report">
        <details className="rawjson">
          <summary>Show raw JSON (exact values)</summary>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </details>
      </Section>
      </div>

      {/* ===== CLEANING TAB ===== */}
      <div hidden={tab !== "cleaning"}>
        <Clean
          key={"clean:" + rev}
          data={data}
          target={target}
          plan={plan?.clean}
          planLoading={planLoading}
          planErr={planErr}
          onCleaned={onCleaned}
        />
      </div>

      {/* ===== EDA TAB ===== */}
      <div hidden={tab !== "eda"}>
        <Charts data={data} />
        <EdaGallery data={data} />
        <ChartBuilder data={data} />
      </div>

      {/* ===== PREPROCESSING TAB ===== */}
      <div hidden={tab !== "preprocessing"}>
        <Preprocess
          key={target + ":" + rev}
          data={data}
          target={target}
          task={task}
          plan={plan?.preprocess}
          aiPipeline={plan?.pipeline}
          planLoading={planLoading}
          planErr={planErr}
          onPlan={setLivePlan}
        />
      </div>

        </div>{/* /report-main */}

        <aside className="chat-dock">
          <ChatPanel data={data} target={target} task={task} plan={livePlan} />
        </aside>
      </div>{/* /report-row */}
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

function Table({ head, rows }) {
  return (
    <div className="panel tablescroll">
      <table>
        <thead>
          <tr>
            {head.map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td key={j}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Stat({ k, v, color, note }) {
  return (
    <div className="card" title={note || undefined}>
      <div className="k">{k}</div>
      <div className="v" style={color ? { color } : undefined}>
        {v}
      </div>
      {note && <div className="note" style={{ fontSize: 11, marginTop: 2 }}>{note}</div>}
    </div>
  );
}

// ---- helpers ----

// Score comes from the backend (detector.quality_score) so report.json and any
// future PDF cite the same number the screen shows. Thresholds match its grades.
function scoreColor(s) {
  return s >= 85 ? "var(--good)" : s >= 70 ? "var(--warn)" : "var(--bad)";
}

function issueDetail(key, checks) {
  switch (key) {
    case "missing_values":
      return `${checks.missing_values.columns.length} column(s)`;
    case "duplicate_rows":
      return `${num(checks.duplicate_rows.summary.duplicate_rows)} row(s)`;
    case "outliers":
      return `${checks.outliers.columns.length} column(s)`;
    case "constant_columns":
      return `${checks.constant_columns.columns.length} column(s)`;
    case "id_columns":
      return checks.id_columns.columns.map((c) => c.column).join(", ") || "—";
    case "high_cardinality":
      return `${checks.high_cardinality.columns.length} column(s)`;
    case "value_issues":
      return `${checks.value_issues.columns.length} column(s)`;
    case "consistency_issues":
      return `${checks.consistency_issues.columns.length} column(s)`;
    default:
      return "";
  }
}
