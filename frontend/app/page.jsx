"use client";

import { useEffect, useRef, useState } from "react";

import Admin from "./admin";
import Bibliometrics from "./biblio";
import Charts from "./charts";
import Dashboard from "./dashboard";
import Projects from "./projects";
import ReportPanel from "./reportpanel";
import Statistics from "./stats";
import TargetPicker, { derivedTask } from "./target";
import EdaGallery from "./eda";
import ChartBuilder from "./builder";
import Clean from "./clean";
import Preprocess from "./preprocess";
import Train from "./train";
import ChatPanel from "./chat";
import Landing from "./landing";
import DatasetUpload from "./upload";
import DatasetPreview from "./preview";
import Icon from "./icons";
import { STEPS, StepFooter, StepHeader, StepRail, pipelineState, stepLabel } from "./workflow";
import { API, call } from "./apiclient";
import { AuthProvider, useAuth } from "./authcontext";
import AuthModal from "./authmodal";

// What each detected issue means for preprocessing. Read-only for now — the
// backend has no "apply" endpoint yet, so these are recommendations, not buttons.
const ISSUE_META = {
  missing_values: { label: "Missing values", rec: "Impute (mean / median / mode / KNN) or drop" },
  duplicate_rows: { label: "Duplicate rows", rec: "Preview and remove duplicates" },
  outliers: { label: "Outliers", rec: "IQR clip, z-score, or winsorize" },
  constant_columns: { label: "Constant columns", rec: "Drop (zero variance, no signal)" },
  id_columns: { label: "Possible ID columns", rec: "Exclude from modeling (leaks / no signal)" },
  high_cardinality: { label: "High-cardinality columns", rec: "Free text: encode as text (word + letter TF-IDF) in Preprocessing; many-valued labels: drop or group them" },
  value_issues: { label: "Value issues", rec: "Convert numbers stored as text; review negatives / zeros (zeros can be marked as missing on the Cleaning tab)" },
  consistency_issues: { label: "Inconsistent categories", rec: "Normalize spellings (Male / male / \" Male\")" },
};

const VALUE_ISSUE_LABEL = {
  negative_values: "Negative values",
  zero_values: "Zero values",
  numeric_stored_as_text: "Numbers stored as text (ratio)",
  whitespace_values: "Leading/trailing whitespace",
};

// Pages that are not pipeline steps, for the header.
const PAGE_LABEL = { dashboard: "Dashboard", projects: "Projects", upload: "New dataset", biblio: "Bibliometrics",
                     admin: "Accounts", statistics: "Statistics" };
const PIPELINE = STEPS.map((s) => s.key);
const CHAT_DOCK_MIN = 1360; // px: below this the AI panel starts closed so the step keeps its width

const num = (v) =>
  typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 3 }) : v ?? "—";

function StudioApp() {
  const { user, logout, openAuthModal, isLoading } = useAuth();
  const [viewMode, setViewMode] = useState("landing");
  const [activeTab, setActiveTab] = useState("dashboard");
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [datasetName, setDatasetName] = useState("");
  // true once the user types a name for the upload: then choosing another file keeps it;
  // otherwise the name follows the chosen file (it kept the previous dataset's name before)
  const nameTyped = useRef(false);
  const [taskType, setTaskType] = useState("auto");
  const [taskChoice, setTaskChoice] = useState("auto"); // the Target picker's task (auto = from the target's type)
  const [biblioId, setBiblioId] = useState(null);
  const [target, setTarget] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  // What the user has finished in the open project (drives Next and the sidebar track).
  // prepFor: the target the stored preprocessing was fitted for ("" = clustering).
  const [progress, setProgress] = useState({});
  const [chatOpen, setChatOpen] = useState(true);
  // Upload limits come from the backend (MAX_UPLOAD_MB in backend/.env), never a copy here.
  // If they can't be loaded the quick check is skipped -- the backend still enforces them.
  const [limits, setLimits] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/upload/limits`).then((r) => (r.ok ? r.json() : null)).then(setLimits).catch(() => {});
    setChatOpen(window.innerWidth >= CHAT_DOCK_MIN);
  }, []);

  // Signing in (or a session restored on load) opens the studio. The same account coming
  // back after an expired session keeps its place; another account starts on a clean dashboard.
  const signedIn = useRef(false);
  const lastAccount = useRef(null);
  useEffect(() => {
    if (isLoading) return;
    if (user && !signedIn.current) {
      if (lastAccount.current !== null && lastAccount.current !== user.id) {
        setResult(null); setProgress({}); setTarget(""); setTaskChoice("auto"); setFile(null);
        setDatasetName(""); nameTyped.current = false; setBiblioId(null); setError("");
      }
      if (lastAccount.current !== user.id) setActiveTab("dashboard");
      lastAccount.current = user.id;
      setViewMode("studio");
    }
    signedIn.current = !!user;
  }, [user, isLoading]);

  // Enforce mandatory registration/login for accessing the studio
  useEffect(() => {
    if (!isLoading && !user && viewMode === "studio") {
      setViewMode("landing");
      openAuthModal("login");
    }
  }, [user, isLoading, viewMode, openAuthModal]);

  // Looking at a step is what finishes Preview and Explore.
  useEffect(() => {
    const key = activeTab === "preview" ? "preview" : activeTab === "eda" ? "eda" : null;
    if (key && result && !progress[key]) setProgress((p) => ({ ...p, [key]: true }));
  }, [activeTab, result, progress]);

  const clustering = taskChoice === "clustering";
  const hasTarget = !!target || clustering;
  // preprocessing (and the runs made from it) only count for the target it was fitted for
  const prepValid = !!progress.preprocess && (progress.prepFor ?? "") === (clustering ? "" : target);
  const { done, locked } = pipelineState({ ...progress, preprocess: prepValid, train: prepValid && progress.train }, hasTarget);

  function pick(f) {
    setError("");
    if (!f) return;
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (limits && !limits.allowed.includes(ext)) {
      setError(`Unsupported file type "${ext}". Use CSV or Excel (${limits.allowed.join(", ")}).`);
      return;
    }
    if (limits && f.size > limits.max_mb * 1024 * 1024) {
      setError(`File is ${(f.size / 1024 / 1024).toFixed(0)} MB — the limit is ${limits.max_mb} MB.`);
      return;
    }
    setFile(f);
    if (!nameTyped.current || !datasetName.trim()) {
      nameTyped.current = false;
      setDatasetName(f.name.replace(/\.[^/.]+$/, "").replace(/[-_]/g, " "));   // "nvda_stock-clean.csv" -> "nvda stock clean"
    }
  }

  function typeName(value) {
    nameTyped.current = true;
    setDatasetName(value);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await call("/api/analyze", form);
      if (datasetName.trim() && datasetName.trim() !== data.title) {
        try { data.title = (await call(`/api/projects/${data.id}`, { title: datasetName.trim() }, "PATCH")).title; } catch {}
      }
      nameTyped.current = false;   // a typed name was for this upload; the next file names itself again
      setResult(data);
      setProgress({});
      setTarget("");
      setTaskChoice(taskType === "auto" ? "auto" : taskType);
      switchTab("preview");
    } catch (e) {
      setError(e.message.startsWith("Cannot reach") ? `${e.message} Is it running? (uvicorn app:app --reload)` : e.message);
    } finally {
      setLoading(false);
    }
  }

  // a sample dataset bundled with scikit-learn, created on the server
  async function loadSample(key, task) {
    setError("");
    setLoading(true);
    try {
      const data = await call(`/api/samples/${key}`, {});
      setResult(data);
      setProgress({});
      setTarget(data.suggested?.target || "");
      setTaskChoice(task === "classification" || task === "regression" ? "auto" : task);
      switchTab("preview");
    } catch (e) {
      setError(e.message);
      switchTab("upload");
    } finally {
      setLoading(false);
    }
  }

  // reopen a saved project where it was left: its current table + last target / task
  async function openProject(p) {
    setError("");
    if (p.kind === "bibliometric") { setBiblioId(p.id); switchTab("biblio"); return; }
    try {
      const data = await call(`/api/projects/${p.id}`);
      setResult(data);
      setTarget(data.restore?.target || "");
      setTaskChoice(data.restore?.task || "auto");
      // a stored preprocessing means the steps before it were finished last time
      setProgress(data.restore
        ? { preview: true, clean: true, eda: true, preprocess: true, prepFor: data.restore.target || "", train: p.runs > 0 }
        : { preview: true });
      switchTab(data.restore ? (p.runs > 0 ? "training" : "preprocessing") : "insights");
    } catch (e) {
      setError(e.message);
      switchTab("upload");
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setProgress({});
    setDatasetName("");
    nameTyped.current = false;
    setTarget("");
    setTaskChoice("auto");
    setError("");
    switchTab("upload");
  }

  function switchTab(t) {
    setActiveTab(t);
    setMobileNavOpen(false);
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.scrollTo({ top: 0, behavior: still ? "auto" : "smooth" });
  }

  function goStep(key) {
    if (!result || locked[key]) return;
    switchTab(key);
  }

  const mark = (key, value = true) => setProgress((p) => ({ ...p, [key]: value }));

  if (viewMode === "landing") {
    return (
      <>
        <Landing
          onLaunchStudio={() => {
            if (!user) {
              openAuthModal("register");
            } else {
              setViewMode("studio");
            }
          }}
        />
        <AuthModal />
      </>
    );
  }

  const isStep = PIPELINE.includes(activeTab);
  const stepNow = STEPS.find((s) => s.key === activeTab);
  const task = clustering ? "clustering" : "";
  const nextStep = STEPS.find((s) => !done[s.key]) || STEPS[STEPS.length - 1];
  const navItem = (key, icon, label, opts = {}) => (
    <button type="button" className={`nav-item ${activeTab === key ? "is-active" : ""}`}
            aria-current={activeTab === key ? "page" : undefined}
            disabled={opts.disabled} title={opts.title} onClick={() => switchTab(key)}>
      <Icon name={icon} size={18} />
      <span>{label}</span>
    </button>
  );

  return (
    <div className="studio-app-shell">
      {mobileNavOpen && (
        <div className="studio-sidebar-backdrop" onClick={() => setMobileNavOpen(false)} aria-hidden="true" />
      )}

      <aside className={`sidebar ${mobileNavOpen ? "is-open" : ""}`} aria-label="Studio navigation">
        <div className="sb-top">
          <button type="button" className="sb-brand" onClick={() => switchTab("dashboard")}>
            <span className="sb-logo">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
                <path d="M12 2v8M4.93 10.93 9.17 15.17M2 18h20M20 18v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2M7 14h10l-2-4H9l-2 4Z" />
              </svg>
            </span>
            <span className="sb-brand-name">Model Forge</span>
          </button>
          <button type="button" className="icon-btn sb-close" onClick={() => setMobileNavOpen(false)} aria-label="Close navigation">
            <Icon name="x" size={18} />
          </button>
        </div>

        <nav className="sb-nav">
          <div className="sb-group">
            <p className="sb-label">Workspace</p>
            {navItem("dashboard", "dashboard", "Dashboard")}
            {navItem("projects", "folder", "Projects")}
            {navItem("upload", "upload", "New dataset")}
          </div>

          <div className="sb-group">
            <p className="sb-label">Pipeline</p>
            {result ? (
              <>
                <p className="sb-project" title={result.title || result.filename}>
                  <Icon name="table" size={15} />
                  <span>{result.title || result.filename}</span>
                </p>
                <StepRail current={isStep ? activeTab : null} done={done} locked={locked} task={task} onGo={goStep} />
              </>
            ) : (
              <div className="sb-empty">
                <p>Open a project or upload a dataset to start the guided pipeline.</p>
                <button type="button" className="btn btn-secondary btn-sm" onClick={() => switchTab("upload")}>
                  <Icon name="plus" size={15} /> New dataset
                </button>
              </div>
            )}
          </div>

          <div className="sb-group">
            <p className="sb-label">Tools</p>
            {navItem("statistics", "sigma", "Statistics", { disabled: !result, title: result ? undefined : "Open a dataset first" })}
            {navItem("biblio", "network", "Bibliometrics")}
          </div>

          {user?.role === "admin" && (
            <div className="sb-group">
              <p className="sb-label">Administration</p>
              {navItem("admin", "users", "Accounts")}
            </div>
          )}
        </nav>

        <div className="sb-foot">
          {user && (
            <div className="sb-user">
              <span className="avatar">
                {user.avatar_url ? <img src={user.avatar_url} alt="" /> : (user.full_name || user.email || "U")[0].toUpperCase()}
              </span>
              <span className="sb-user-text">
                <b>{user.full_name || user.email.split("@")[0]}</b>
                <span>{user.role === "admin" ? "Administrator" : user.email}</span>
              </span>
              <button type="button" className="icon-btn" title="Sign out" aria-label="Sign out"
                      onClick={() => { logout(); setViewMode("landing"); }}>
                <Icon name="logOut" size={17} />
              </button>
            </div>
          )}
          <button type="button" className="sb-home" onClick={() => { setViewMode("landing"); setMobileNavOpen(false); }}>
            <Icon name="home" size={15} /> Home page
          </button>
        </div>
      </aside>

      <main className="studio-main-viewport">
        <header className="topbar">
          <button type="button" className="icon-btn topbar-menu" onClick={() => setMobileNavOpen((o) => !o)}
                  aria-label="Open navigation" aria-expanded={mobileNavOpen}>
            <Icon name="menu" size={20} />
          </button>
          <nav className="crumbs" aria-label="Breadcrumb">
            <button type="button" onClick={() => switchTab("dashboard")}>Dashboard</button>
            {(isStep || activeTab === "statistics") && result ? (
              <>
                <span aria-hidden="true">/</span>
                <span className="crumb-project" title={result.title || result.filename}>{result.title || result.filename}</span>
                <span aria-hidden="true">/</span>
                <b>{isStep ? stepLabel(stepNow, task) : PAGE_LABEL.statistics}</b>
              </>
            ) : activeTab !== "dashboard" && (
              <>
                <span aria-hidden="true">/</span>
                <b>{PAGE_LABEL[activeTab] || stepNow?.label}</b>
              </>
            )}
          </nav>
          <div className="topbar-right">
            {result && (
              <span className="chip" title={result.filename}>
                <Icon name="table" size={14} />
                {result.profile?.rows?.toLocaleString()} × {result.profile?.columns}
              </span>
            )}
            {result && (target || clustering) && (
              <button type="button" className="chip chip-accent" onClick={() => goStep("insights")}
                      title="Change the target on the quality audit">
                <Icon name="target" size={14} />
                <span>{clustering ? "Clustering" : target}</span>
              </button>
            )}
          </div>
        </header>

        <div className={`studio-view-canvas ${isStep || activeTab === "statistics" ? "is-wide" : ""}`}>
          {activeTab === "dashboard" ? (
            <Dashboard
              user={user}
              samples={limits?.samples}
              current={result && { title: result.title || result.filename, done: STEPS.filter((s) => done[s.key]).length,
                                   total: STEPS.length, next: stepLabel(nextStep, task) }}
              onContinue={() => switchTab(nextStep.key)}
              onOpen={openProject}
              onNew={() => switchTab("upload")}
              onSample={loadSample}
              onGo={switchTab}
            />
          ) : activeTab === "projects" ? (
            <Projects onOpen={openProject} />
          ) : activeTab === "biblio" ? (
            <Bibliometrics id={biblioId} onOpened={(r) => setBiblioId(r?.id || null)} />
          ) : activeTab === "admin" && user?.role === "admin" ? (
            <Admin me={user} />
          ) : activeTab === "upload" || !result ? (
            <DatasetUpload
              file={file}
              setFile={pick}
              datasetName={datasetName}
              setDatasetName={typeName}
              taskType={taskType}
              setTaskType={setTaskType}
              loading={loading}
              error={error}
              onAnalyze={analyze}
              onReset={reset}
              limits={limits}
              onSample={loadSample}
            />
          ) : activeTab === "preview" ? (
            <>
              <StepHeader stepKey="preview" done={done} task={task} />
              <DatasetPreview data={result} target={target} setTarget={setTarget} />
              <StepFooter stepKey="preview" done={done} task={task} onGo={goStep} />
            </>
          ) : (
            <Report
              data={result}
              setData={setResult}
              tab={activeTab}
              setTab={goStep}
              target={target}
              setTarget={setTarget}
              taskChoice={taskChoice}
              setTaskChoice={setTaskChoice}
              done={done}
              mark={mark}
              chatOpen={chatOpen}
              setChatOpen={setChatOpen}
              onDashboard={() => switchTab("dashboard")}
            />
          )}
        </div>
      </main>
      <AuthModal />
    </div>
  );
}

// ---------------------------------------------------------------------------

function Report({ data, setData, tab, setTab, target, setTarget, taskChoice = "auto", setTaskChoice,
                  done, mark, chatOpen, setChatOpen, onDashboard }) {
  const { profile, report, filename } = data;
  const info = profile.columns_info;
  const checks = report.checks;

  // detector's semantic type per column
  const semType = {};
  for (const c of checks.column_types.columns) semType[c.column] = c.detected_type;

  // the user's choice wins; "auto" follows the target's type (numeric -> regression)
  const task = taskChoice === "clustering" ? "clustering"
    : !target ? ""
    : taskChoice !== "auto" ? taskChoice
    : derivedTask(semType, target);

  // LLM plan (clean + preprocess suggestions). Optional overlay: on failure the
  // tabs fall back to rule-based defaults. Fetched once per dataset.
  const [plan, setPlan] = useState(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [planErr, setPlanErr] = useState("");
  const [livePlan, setLivePlan] = useState(null); // user's edited plan -> chat memory
  const [rev, setRev] = useState(0);               // bumps when data is cleaned, to reset child state
  const [prepRev, setPrepRev] = useState(0);       // bumps after each preprocess, so Train re-reads options + AI picks
  const [trainResults, setTrainResults] = useState(null); // the run shown on the Train tab -> chat context
  const [chatAsk, setChatAsk] = useState(null);           // {text, n}: a question sent to the chat by a button

  useEffect(() => {
    let cancel = false;
    setPlan(null);          // never show or apply the previous target's plan while the new one loads
    setPlanLoading(true);
    setPlanErr("");
    call("/api/plan", { id: data.id, target, task: task === "clustering" ? "classification" : task })
      .then((p) => { if (!cancel) setPlan(p); })
      .catch((e) => { if (!cancel) setPlanErr(e.message); })
      .finally(() => { if (!cancel) setPlanLoading(false); });
    return () => { cancel = true; };
    // refetch when the user picks/changes the target so the AI refines its plan
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.id, target]);

  // The table changed (cleaning, or reverse scoring on Statistics): refresh everything,
  // and the stored preprocessing no longer matches the table -- it has to run again.
  function onDataChanged(fresh) {
    setData(fresh);
    setRev((r) => r + 1);
    mark("preprocess", false);
    mark("train", false);
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

  const isStep = tab !== "statistics";
  const chatToggle = (
    <button type="button" className="btn btn-secondary btn-sm chat-toggle" aria-pressed={chatOpen}
            aria-controls="ai-dock" onClick={() => setChatOpen((o) => !o)}>
      <Icon name="bot" size={16} /> {chatOpen ? "Hide assistant" : "AI assistant"}
    </button>
  );
  const footer = (key) => (
    <StepFooter stepKey={key} done={done} task={task} onGo={setTab}
                last={<button type="button" className="btn btn-primary btn-lg" onClick={onDashboard}>
                  <Icon name="dashboard" size={17} /> Back to dashboard</button>} />
  );

  return (
    <>
      {isStep ? (
        <StepHeader stepKey={tab} done={done} task={task}>
          {tab === "insights" && (
            <button type="button" className="btn btn-quiet btn-sm" onClick={download} title="The full audit as JSON">
              <Icon name="download" size={15} /> Audit JSON
            </button>
          )}
          {chatToggle}
        </StepHeader>
      ) : (
        <header className="step-head">
          <div className="step-head-row">
            <div className="step-head-text">
              <h1 className="step-title"><span className="step-title-icon"><Icon name="sigma" size={20} /></span>Statistics</h1>
              <p className="step-blurb">Descriptive and inferential tests on {data.title || filename}, each with its assumptions and effect size.</p>
            </div>
            <div className="step-head-actions">{chatToggle}</div>
          </div>
        </header>
      )}

      <div className={`report-row ${chatOpen ? "" : "chat-closed"}`}>
        <div className="report-main">

      {/* Compact active target indicator for downstream tabs */}
      {tab !== "insights" && (target || task === "clustering") && (
        <div className="target-bar">
          <Icon name="target" size={16} />
          <span>
            {task === "clustering" ? "No target (clustering)" : <>Target <b>{target}</b></>}
            <span className="pill pill-neutral">{task || (info[target]?.type === "numeric" ? "regression" : "classification")}</span>
          </span>
          <button type="button" className="link-btn" onClick={() => setTab("insights")}>Change</button>
        </div>
      )}

      {/* ===== INSIGHTS TAB ===== */}
      <div hidden={tab !== "insights"}>
        {/* plan §24: warn about possible personal identifiers before anything is processed */}
        {(checks.personal_data?.has_personal_data || checks.id_columns.has_id_columns) && (
          <div className="privacy-note" role="note">
            <Icon name="shield" size={18} />
            <div>
              <b>Possible personal identifiers</b>
              <p>
                {[...(checks.personal_data?.columns || []).map((p) => `${p.column} (${p.kind === "email" ? "e-mail addresses" : "phone numbers"})`),
                  ...checks.id_columns.columns.map((c) => `${c.column} (one value per row)`)].join(", ")}.
                {" "}Drop them on the Cleaning step unless the analysis needs them, and make sure their use is covered by consent
                or ethics approval (you can record it on the Report step).
              </p>
            </div>
          </div>
        )}
        {/* Target: selected during Quality Audit to guide cleaning, EDA, and preprocessing */}
        <TargetPicker data={data} target={target} setTarget={setTarget} taskChoice={taskChoice} setTaskChoice={setTaskChoice} />
        {data.load_notes?.length > 0 && (
          <div className="panel" style={{ padding: 12, marginBottom: 10 }}>
            <b>Reading the file</b>
            <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>{data.load_notes.map((n, i) => <li key={i} className="note">{n}</li>)}</ul>
          </div>
        )}

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
            head={["Column", "Min", "Max", "Mean", "Median", "Std", "Q1", "Q3", "Skewness", "Kurtosis"]}
            rows={numericCols.map((n) => {
              const s = info[n].statistics;
              return [n, num(s.min), num(s.max), num(s.mean), num(s.median), num(s.std), num(s.q1), num(s.q3),
                      num(s.skew), num(s.kurtosis)];
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
          Apply them on the Cleaning tab (whole-table fixes) and the Preprocessing tab (impute, encode, scale,
          outliers, imbalance — fitted on the training split only).
        </p>
      </Section>

      {/* Raw JSON */}
      <Section title="Raw report">
        <details className="rawjson">
          <summary>Show raw JSON (exact values)</summary>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </details>
      </Section>

        {footer("insights")}
      </div>

      {/* ===== CLEANING TAB ===== */}
      <div hidden={tab !== "cleaning"}>
        <Clean
          key={"clean:" + data.id}
          data={data}
          target={target}
          plan={plan?.clean}
          planLoading={planLoading}
          planErr={planErr}
          onCleaned={(fresh, how) => { onDataChanged(fresh); mark("clean", !how?.reset); }}
          onKept={() => mark("clean")}
          onProceedEDA={() => setTab("eda")}
        />
        {footer("cleaning")}
      </div>

      {/* ===== EDA TAB ===== */}
      <div hidden={tab !== "eda"}>
        <Charts data={data} />
        <EdaGallery data={data} />
        <ChartBuilder data={data} />
        {footer("eda")}
      </div>

      {/* ===== PREPROCESSING TAB ===== */}
      <div hidden={tab !== "preprocessing"}>
        <Preprocess
          key={target + ":" + task + ":" + rev}
          data={data}
          target={task === "clustering" ? "" : target}
          task={task}
          plan={plan?.preprocess}
          aiPipeline={plan?.pipeline}
          planLoading={planLoading}
          planErr={planErr}
          onPlan={setLivePlan}
          onDone={() => { setPrepRev((r) => r + 1); mark("preprocess"); mark("prepFor", task === "clustering" ? "" : target); mark("train", false); }}
        />
        {footer("preprocessing")}
      </div>

      {/* ===== TRAINING TAB ===== */}
      <div hidden={tab !== "training"}>
        <Train key={`train:${rev}:${prepRev}`} data={data} target={task === "clustering" ? "" : target} active={tab === "training"}
               onResults={setTrainResults} onAsk={(text) => setChatAsk((a) => ({ text, n: (a?.n || 0) + 1 }))}
               onTrained={() => mark("train")} />
        {footer("training")}
      </div>

      {/* ===== STATISTICS TAB ===== */}
      <div hidden={tab !== "statistics"}>
        {tab === "statistics" && <Statistics key={"stats:" + rev} data={data} onCleaned={onDataChanged} />}
      </div>

      {/* ===== REPORT TAB ===== */}
      <div hidden={tab !== "report"}>
        {tab === "report" && <ReportPanel data={data} refreshKey={prepRev + rev} onDownloaded={() => mark("report")}
                                          onEthics={(ethics) => setData({ ...data, ethics })} />}
        {footer("report")}
      </div>

        </div>{/* /report-main */}

        <aside className="chat-dock" id="ai-dock" hidden={!chatOpen} aria-label="AI assistant">
          <ChatPanel data={data} target={target} task={task} plan={livePlan} results={trainResults} asked={chatAsk} />
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

export default function Page() {
  return (
    <AuthProvider>
      <StudioApp />
    </AuthProvider>
  );
}
