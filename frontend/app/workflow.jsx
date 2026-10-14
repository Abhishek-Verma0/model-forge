"use client";

// The guided pipeline (plan §7 steps 2-10 as screens): the order, what "done" means for
// each step, the sidebar track, the step heading and the Back / Next footer. Next only
// appears once the step itself is done; the sidebar locks only hard prerequisites
// (preprocessing needs a target, training needs preprocessed data).
import Icon from "./icons";

export const STEPS = [
  { key: "preview", label: "Preview", icon: "eye", title: "Preview the data",
    blurb: "The columns, their types and the first rows, before anything changes." },
  { key: "insights", label: "Quality audit", icon: "shield", title: "Audit quality and choose the target",
    blurb: "What needs fixing, and which column to predict (or clustering)." },
  { key: "cleaning", label: "Cleaning", icon: "eraser", title: "Clean the data",
    blurb: "Fix types, spellings, placeholders and duplicates on the whole table." },
  { key: "eda", label: "Explore", icon: "chart", title: "Explore the data",
    blurb: "Distributions, relationships and your own charts, each downloadable." },
  { key: "preprocessing", label: "Preprocess", icon: "sliders", title: "Preprocess for modelling",
    blurb: "Impute, encode, scale and split, fitted on the training rows only." },
  { key: "training", label: "Train and evaluate", icon: "cpu", title: "Train, evaluate and explain",
    blurb: "Cross-validated models, their charts and what drives their predictions." },
  { key: "report", label: "Report", icon: "report", title: "Download the report",
    blurb: "The PDF report, Excel tables and the experiment record of this project." },
];

// What each step needs before Next appears (shown in place of the button).
const HINT = {
  insights: "Choose the target column (or clustering) above to continue.",
  cleaning: "Apply the cleaning, or keep the data as it is, to continue.",
  preprocessing: "Run preprocessing to continue.",
  training: "Finish a training run to continue to the report.",
};

/** done / locked per step from what the user has finished in this project. */
export function pipelineState(progress, hasTarget) {
  const done = {
    preview: !!progress.preview, insights: hasTarget, cleaning: !!progress.clean, eda: !!progress.eda,
    preprocessing: !!progress.preprocess, training: !!progress.train, report: !!progress.report,
  };
  const locked = {
    preprocessing: hasTarget ? "" : "Choose a target on the quality audit first",
    training: progress.preprocess ? "" : "Run preprocessing first",
  };
  return { done, locked };
}

export const stepLabel = (step, task) =>
  step.key === "training" && task === "clustering" ? "Cluster and evaluate" : step.label;

/** The numbered track in the sidebar; the line fills as steps are finished. */
export function StepRail({ current, done, locked, task, onGo }) {
  return (
    <ol className="rail">
      {STEPS.map((s, i) => {
        const isDone = done[s.key];
        const lock = locked[s.key];
        const state = s.key === current ? "current" : isDone ? "done" : lock ? "locked" : "todo";
        return (
          <li key={s.key} className={`rail-step is-${state}${isDone ? " is-complete" : ""}`}>
            <button type="button" className="rail-btn" disabled={!!lock} onClick={() => onGo(s.key)}
                    aria-current={s.key === current ? "step" : undefined} title={lock || undefined}>
              <span className="rail-node" aria-hidden="true">
                {isDone && s.key !== current ? <Icon name="check" size={13} strokeWidth={3} /> : lock ? <Icon name="lock" size={12} /> : i + 1}
              </span>
              <span className="rail-text">
                <span className="rail-label">{stepLabel(s, task)}</span>
                {lock && <span className="rail-sub">{lock}</span>}
              </span>
              <span className="sr-only">{isDone ? "(done)" : lock ? "(locked)" : ""}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

/** Heading of a pipeline step: where you are, what the step does, overall progress. */
export function StepHeader({ stepKey, done, task, children }) {
  const i = STEPS.findIndex((s) => s.key === stepKey);
  const step = STEPS[i];
  const finished = STEPS.filter((s) => done[s.key]).length;
  return (
    <header className="step-head">
      <div className="step-head-row">
        <div className="step-head-text">
          <p className="step-count">Step {i + 1} of {STEPS.length}</p>
          <h1 className="step-title">
            <span className="step-title-icon"><Icon name={step.icon} size={20} /></span>
            {step.key === "training" && task === "clustering" ? "Cluster, evaluate and explain" : step.title}
          </h1>
          <p className="step-blurb">{step.blurb}</p>
        </div>
        {children && <div className="step-head-actions">{children}</div>}
      </div>
      <div className="step-progress" role="progressbar" aria-label="Pipeline progress"
           aria-valuemin={0} aria-valuemax={STEPS.length} aria-valuenow={finished}>
        <span style={{ width: `${(finished / STEPS.length) * 100}%` }} />
      </div>
    </header>
  );
}

/** Back / Next for a pipeline step. Next is shown only when the step is done. */
export function StepFooter({ stepKey, done, task, onGo, last }) {
  const i = STEPS.findIndex((s) => s.key === stepKey);
  const prev = STEPS[i - 1];
  const next = STEPS[i + 1];
  return (
    <nav className="step-foot" aria-label="Pipeline navigation">
      {prev ? (
        <button type="button" className="btn btn-quiet" onClick={() => onGo(prev.key)}>
          <Icon name="arrowLeft" size={16} /> {stepLabel(prev, task)}
        </button>
      ) : <span />}
      {next ? (
        done[stepKey] || !HINT[stepKey] ? (   // Preview and Explore are done by looking at them
          <button type="button" className="btn btn-primary btn-lg" onClick={() => onGo(next.key)}>
            Next: {stepLabel(next, task)} <Icon name="arrowRight" size={17} />
          </button>
        ) : (
          <p className="step-foot-hint" role="status"><Icon name="lock" size={15} /> {HINT[stepKey]}</p>
        )
      ) : last}
    </nav>
  );
}
