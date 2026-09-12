"use client";

import { useRef, useState } from "react";

const SAMPLES = [
  {
    id: "churn",
    title: "Customer Churn",
    type: "Classification",
    filename: "customer_churn.csv",
    icon: "🏦",
    task: "classification",
    desc: "Predict customer churn with tenure & monthly charges",
  },
  {
    id: "housing",
    title: "California Housing",
    type: "Regression",
    filename: "california_housing.csv",
    icon: "🏠",
    task: "regression",
    desc: "Predict median house values across districts",
  },
  {
    id: "diabetes",
    title: "Healthcare Risk",
    type: "Classification",
    filename: "healthcare_diabetes.csv",
    icon: "🩺",
    task: "classification",
    desc: "Diagnostic metrics for diabetes outcome prediction",
  },
];

export default function DatasetUpload({
  file,
  setFile,
  datasetName,
  setDatasetName,
  taskType,
  setTaskType,
  loading,
  error,
  onAnalyze,
  onReset,
}) {
  const [dragOver, setDragOver] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const inputRef = useRef(null);

  function handleFileSelect(selectedFile) {
    if (!selectedFile) return;
    setFile(selectedFile);
    if (!datasetName) {
      // derive clean name from filename (remove extension)
      const baseName = selectedFile.name.replace(/\.[^/.]+$/, "");
      setDatasetName(baseName.replace(/[-_]/g, " "));
    }
  }

  async function loadSampleDataset(sample) {
    try {
      setSampleLoading(true);
      const res = await fetch(`/samples/${sample.filename}`);
      if (!res.ok) throw new Error("Could not load sample file");
      const blob = await res.blob();
      const sampleFile = new File([blob], sample.filename, { type: "text/csv" });
      setFile(sampleFile);
      setDatasetName(sample.title);
      setTaskType(sample.task);
    } catch (e) {
      console.error("Error loading sample:", e);
    } finally {
      setSampleLoading(false);
    }
  }

  const formatFileSize = (bytes) => {
    if (!bytes) return "0 B";
    const k = 1024;
    if (bytes < k * k) return (bytes / k).toFixed(1) + " KB";
    return (bytes / (k * k)).toFixed(2) + " MB";
  };

  return (
    <div className="upload-view-container">
      {/* View Header */}
      <div className="upload-view-header">
        <div className="upload-header-text">
          <h1 className="upload-title">Upload Your Dataset</h1>
          <p className="upload-subtitle">
            Upload CSV or Excel files to profile schema, audit data hygiene, and forge leak-free ML pipelines.
          </p>
        </div>
      </div>

      {/* Main Upload Dropzone */}
      <div
        className={`upload-dropzone ${dragOver ? "drag-active" : ""} ${file ? "has-file" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (e.dataTransfer.files?.[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          hidden
          onChange={(e) => handleFileSelect(e.target.files?.[0])}
        />

        {!file ? (
          <div className="dropzone-empty-state">
            <div className="dropzone-icon-circle" onClick={() => inputRef.current?.click()}>
              <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                <path d="M12 12v9" />
                <path d="m16 16-4-4-4 4" />
              </svg>
            </div>

            <h3 className="dropzone-prompt">Drag & Drop your dataset here</h3>
            <span className="dropzone-or">or</span>

            <button
              type="button"
              className="btn-browse-files"
              onClick={() => inputRef.current?.click()}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <span>Browse Files</span>
            </button>

            <div className="dropzone-limits">
              <span>Supported formats: <strong>CSV, XLSX, XLS</strong></span>
              <span className="limits-dot">·</span>
              <span>Max size: <strong>200 MB</strong></span>
            </div>
          </div>
        ) : (
          <div className="dropzone-file-state">
            <div className="selected-file-card">
              <div className="file-badge-icon">
                <span className="file-ext-label">{file.name.split(".").pop()?.toUpperCase()}</span>
              </div>
              <div className="file-info-group">
                <div className="file-name-row">
                  <span className="file-name-text">{file.name}</span>
                  <span className="file-ready-badge">Ready for Profiling</span>
                </div>
                <div className="file-meta-row">
                  <span>Size: {formatFileSize(file.size)}</span>
                  <span className="meta-sep">·</span>
                  <span>Type: {file.type || "Tabular Dataset"}</span>
                </div>
              </div>

              <div className="file-actions">
                <button
                  type="button"
                  className="btn-change-file"
                  onClick={() => inputRef.current?.click()}
                  title="Change file"
                >
                  Change
                </button>
                <button
                  type="button"
                  className="btn-remove-file"
                  onClick={onReset}
                  title="Remove file"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Dataset Configuration / Information Panel */}
      <div className="dataset-config-panel">
        <div className="panel-header-row">
          <div className="panel-title-group">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 20h9" />
              <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
            </svg>
            <h3 className="panel-title">Dataset Information</h3>
          </div>
          <span className="panel-note">Metadata & Modeling Target</span>
        </div>

        <div className="config-form-grid">
          {/* Dataset Name */}
          <div className="form-field">
            <label className="form-label" htmlFor="datasetName">
              Dataset Name
            </label>
            <input
              id="datasetName"
              type="text"
              className="form-input"
              placeholder="e.g. Customer Churn Analysis"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
            />
          </div>

          {/* Machine Learning Task Type */}
          <div className="form-field">
            <label className="form-label" htmlFor="taskType">
              Task Type
            </label>
            <select
              id="taskType"
              className="form-select"
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
            >
              <option value="auto">Auto Detect (Recommended)</option>
              <option value="classification">Classification (Binary / Multi-Class)</option>
              <option value="regression">Regression (Continuous Numeric)</option>
              <option value="unsupervised">Clustering / Unsupervised EDA</option>
            </select>
          </div>
        </div>

        {/* Fit-on-Train Guard & Feature Notice */}
        <div className="config-badge-strip">
          <div className="guard-badge">
            <span className="guard-icon">🛡️</span>
            <div>
              <strong>Strict Fit-on-Train Guard Active:</strong>
              <span> Scalers and encoders are strictly fitted on the training split to eliminate data leakage.</span>
            </div>
          </div>
        </div>
      </div>

      {/* 1-Click Sample Datasets Strip */}
      <div className="sample-datasets-section">
        <div className="sample-header-row">
          <span className="sample-label">Or explore with a 1-click sample dataset:</span>
          {sampleLoading && <span className="sample-loading-text">Loading sample…</span>}
        </div>
        <div className="sample-chips-row">
          {SAMPLES.map((sample) => (
            <button
              key={sample.id}
              type="button"
              className={`sample-chip ${file?.name === sample.filename ? "active" : ""}`}
              onClick={() => loadSampleDataset(sample)}
              disabled={loading || sampleLoading}
            >
              <span className="sample-icon">{sample.icon}</span>
              <div className="sample-chip-text">
                <span className="sample-chip-title">{sample.title}</span>
                <span className="sample-chip-tag">{sample.type}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Error Alert Display */}
      {error && (
        <div className="upload-error-alert">
          <div className="error-icon">⚠️</div>
          <div className="error-text">
            <strong>Analysis Failed:</strong> {error}
          </div>
        </div>
      )}

      {/* Submit Action Bar */}
      <div className="upload-action-row">
        <button
          id="btn-upload-proceed"
          type="button"
          className="btn-upload-proceed"
          onClick={onAnalyze}
          disabled={!file || loading}
        >
          {loading ? (
            <>
              <span className="spinner-dot"></span>
              <span>Profiling & Auditing Dataset…</span>
            </>
          ) : (
            <>
              <span>Upload and Proceed</span>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </>
          )}
        </button>

        {file && !loading && (
          <button type="button" className="btn-upload-clear" onClick={onReset}>
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
