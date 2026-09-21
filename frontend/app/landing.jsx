"use client";

import { useState } from "react";
import { useAuth } from "./authcontext";

export default function Landing({ onLaunchStudio }) {
  const { user, openAuthModal, logout } = useAuth();

  function handleLaunch() {
    if (!user) {
      openAuthModal("register");
    } else {
      onLaunchStudio();
    }
  }

  return (
    <div className="landing-root">
      {/* Top Navbar */}
      <header className="landing-header">
        <div className="landing-nav-inner">
          <div className="landing-brand" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
            <div className="brand-icon-box">
              {/* Modern Forge Anvil/Flame SVG Icon */}
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2v8" />
                <path d="m4.93 10.93 4.24 4.24" />
                <path d="M2 18h20" />
                <path d="M20 18v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2" />
                <path d="M7 14h10l-2-4H9l-2 4Z" />
              </svg>
            </div>
            <span className="brand-title">Model Forge</span>
            <span className="brand-badge">Studio</span>
          </div>

          <nav className="landing-nav-links">
            <a href="#hero" className="nav-link active">Home</a>
            <a href="#features" className="nav-link">Features</a>
            <a href="#pipeline" className="nav-link">Pipeline</a>
            <a href="#features" className="nav-link">About</a>
          </nav>

          <div className="landing-nav-cta" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {user ? (
              <>
                <div className="user-nav-profile">
                  <div className="user-avatar-circle">
                    {user.avatar_url ? (
                      <img src={user.avatar_url} alt={user.full_name || user.email} />
                    ) : (
                      <span>{(user.full_name || user.email || "U")[0].toUpperCase()}</span>
                    )}
                  </div>
                  <span className="user-nav-name">{user.full_name || user.email.split("@")[0]}</span>
                  <button type="button" className="btn-sign-out" onClick={logout} title="Sign Out">
                    Sign Out
                  </button>
                </div>
                <button
                  id="btn-get-started"
                  type="button"
                  className="btn-cream-primary"
                  onClick={onLaunchStudio}
                >
                  <span>Open Studio</span>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <path d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  className="btn-nav-signin"
                  onClick={() => openAuthModal("login")}
                >
                  Sign In
                </button>
                <button
                  id="btn-get-started"
                  type="button"
                  className="btn-cream-primary"
                  onClick={() => openAuthModal("register")}
                >
                  <span>Get Started</span>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <path d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                </button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section id="hero" className="landing-hero">
        <div className="hero-grid">
          {/* Left Column: Value Proposition & Copy */}
          <div className="hero-content">
            <div className="hero-pill">
              <span className="pill-dot"></span>
              <span>Intelligent Data & ML Preprocessing Studio</span>
            </div>

            <h1 className="hero-title">
              Forge Raw Data into <br />
              <span className="hero-highlight">Production-Ready</span> <br />
              Machine Learning
            </h1>

            <p className="hero-subtitle">
              Upload your dataset, let AI audit data quality and prevent leakage,
              engineer robust feature pipelines, and forge clean training sets — all in one unified studio.
            </p>

            <div className="hero-actions">
              <button
                id="btn-launch-studio"
                type="button"
                className="btn-hero-primary"
                onClick={handleLaunch}
              >
                <span>Launch Studio</span>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>
              <a href="#pipeline" className="btn-hero-secondary">
                <span>Explore Pipeline</span>
              </a>
            </div>

            {/* Quick Proof Badges */}
            <div className="hero-proof-bar">
              <div className="proof-item">
                <span className="proof-check">✓</span>
                <span>200 MB Ingestion</span>
              </div>
              <div className="proof-item">
                <span className="proof-check">✓</span>
                <span>Zero Data Leakage</span>
              </div>
              <div className="proof-item">
                <span className="proof-check">✓</span>
                <span>Ollama AI Copilot</span>
              </div>
            </div>
          </div>

          {/* Right Column: Warm Organic Backdrop + Bento Data Cards */}
          <div className="hero-visual">
            <div className="hero-blob-backdrop"></div>

            <div className="bento-container">
              {/* Card 1: Data Health & Quality Audit Score */}
              <div className="bento-card bento-health">
                <div className="card-top-row">
                  <div className="badge-tag green">Data Health Audit</div>
                  <span className="live-dot"></span>
                </div>
                <div className="health-score-row">
                  <div className="radial-score">
                    <svg viewBox="0 0 36 36" className="circular-chart">
                      <path
                        className="circle-bg"
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                      <path
                        className="circle"
                        strokeDasharray="98.4, 100"
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                    </svg>
                    <div className="score-number">98.4%</div>
                  </div>
                  <div className="health-metrics">
                    <div className="h-metric-item">
                      <span className="metric-val text-green">0</span>
                      <span className="metric-lbl">Missing Nulls</span>
                    </div>
                    <div className="h-metric-item">
                      <span className="metric-val text-green">Clean</span>
                      <span className="metric-lbl">0 Leaking IDs</span>
                    </div>
                    <div className="h-metric-item">
                      <span className="metric-val text-amber">14</span>
                      <span className="metric-lbl">Outliers Clipped</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Card 2: Gaussian Feature Distribution & Correlation Heat */}
              <div className="bento-card bento-chart">
                <div className="card-top-row">
                  <span className="card-mini-title">Feature Distribution</span>
                  <span className="card-pill-tag">income_log</span>
                </div>
                <div className="chart-preview">
                  <svg viewBox="0 0 200 70" className="gaussian-svg">
                    <defs>
                      <linearGradient id="warmGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor="#C26E38" stopOpacity="0.4" />
                        <stop offset="100%" stopColor="#C26E38" stopOpacity="0.0" />
                      </linearGradient>
                    </defs>
                    <path
                      d="M 5 65 Q 40 65 60 55 Q 85 40 100 12 Q 115 40 140 55 Q 160 65 195 65 Z"
                      fill="url(#warmGrad)"
                    />
                    <path
                      d="M 5 65 Q 40 65 60 55 Q 85 40 100 12 Q 115 40 140 55 Q 160 65 195 65"
                      fill="none"
                      stroke="#C26E38"
                      strokeWidth="2.5"
                    />
                    <line x1="100" y1="10" x2="100" y2="65" stroke="#70695E" strokeDasharray="3,3" strokeWidth="1.2" />
                  </svg>
                </div>
                <div className="chart-stat-row">
                  <span>μ = 45.2k · σ = 8.4</span>
                  <span className="stat-clean">Normal Skew (0.08)</span>
                </div>
              </div>

              {/* Card 3: Preprocessing Recipe Pipeline */}
              <div className="bento-card bento-recipe">
                <div className="card-top-row">
                  <div className="recipe-title-group">
                    <span className="recipe-icon">⚡</span>
                    <span className="card-mini-title">ML Pipeline Execution</span>
                  </div>
                  <span className="leak-badge">Fit on Train Only</span>
                </div>
                <div className="recipe-steps">
                  <div className="recipe-step">
                    <span className="step-num">1</span>
                    <span className="step-name">Median Impute</span>
                  </div>
                  <span className="step-arr">→</span>
                  <div className="recipe-step">
                    <span className="step-num">2</span>
                    <span className="step-name">RobustScaler</span>
                  </div>
                  <span className="step-arr">→</span>
                  <div className="recipe-step">
                    <span className="step-num">3</span>
                    <span className="step-name">Target Encode</span>
                  </div>
                </div>
              </div>

              {/* Card 4: AI Copilot Assistant Insight */}
              <div className="bento-card bento-copilot">
                <div className="copilot-header">
                  <div className="copilot-avatar">AI</div>
                  <span className="copilot-label">Model Forge Copilot</span>
                  <span className="copilot-model">Ollama</span>
                </div>
                <p className="copilot-text">
                  “Target column <strong>churn</strong> detected with 18% positive rate.
                  Stratified split and tree-based encodings configured automatically.”
                </p>
              </div>

              {/* Card 5: Ingestion Specs & Framework Readiness */}
              <div className="bento-card bento-frameworks">
                <div className="frameworks-meta">
                  <span className="fw-rows">125,000 Rows</span>
                  <span className="fw-dot">·</span>
                  <span className="fw-cols">34 Features</span>
                </div>
                <div className="framework-tags">
                  <span className="fw-tag">Scikit-Learn</span>
                  <span className="fw-tag">XGBoost</span>
                  <span className="fw-tag">PyTorch</span>
                  <span className="fw-tag">LightGBM</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4 Core Features Highlight Bar (matching user's reference layout) */}
      <section className="landing-highlights">
        <div className="highlights-container">
          {/* Item 1: Ingestion */}
          <div className="highlight-card">
            <div className="highlight-icon-wrap">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <h3 className="highlight-title">Easy Dataset Upload</h3>
            <p className="highlight-desc">Supports CSV and Excel up to 200MB with automatic schema detection.</p>
          </div>

          {/* Item 2: Preprocessing */}
          <div className="highlight-card">
            <div className="highlight-icon-wrap">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              </svg>
            </div>
            <h3 className="highlight-title">Automatic Preprocessing</h3>
            <p className="highlight-desc">AI-powered recommendations for nulls, outliers, scaling & encodings.</p>
          </div>

          {/* Item 3: Pipeline & Models */}
          <div className="highlight-card">
            <div className="highlight-icon-wrap">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="2" y="2" width="8" height="8" rx="2" />
                <rect x="14" y="2" width="8" height="8" rx="2" />
                <rect x="8" y="14" width="8" height="8" rx="2" />
                <line x1="6" y1="10" x2="12" y2="14" />
                <line x1="18" y1="10" x2="12" y2="14" />
              </svg>
            </div>
            <h3 className="highlight-title">ML-Ready Pipelines</h3>
            <p className="highlight-desc">Leak-free fit-on-train pipelines tailored for classification and regression.</p>
          </div>

          {/* Item 4: Evaluation & EDA */}
          <div className="highlight-card">
            <div className="highlight-icon-wrap">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="20" x2="18" y2="10" />
                <line x1="12" y1="20" x2="12" y2="4" />
                <line x1="6" y1="20" x2="6" y2="14" />
              </svg>
            </div>
            <h3 className="highlight-title">Detailed Evaluation</h3>
            <p className="highlight-desc">Automated EDA charts, correlation heatmaps and full data audit reports.</p>
          </div>
        </div>
      </section>

      {/* Interactive 4-Step Pipeline Walkthrough */}
      <section id="pipeline" className="landing-pipeline">
        <div className="section-header">
          <div className="section-pill">Step-by-Step Architecture</div>
          <h2 className="section-heading">How Model Forge Works</h2>
          <p className="section-sub">
            From raw, unvalidated CSV spreadsheets to pristine, production-grade tensors ready for model training.
          </p>
        </div>

        <div className="pipeline-steps-grid">
          <div className="step-card">
            <div className="step-number-badge">01</div>
            <h4>Ingest & Profile</h4>
            <p>
              Instantly inspect data types, row counts, memory footprint, null percentages, and cardinality metrics without writing boilerplate scripts.
            </p>
          </div>

          <div className="step-card">
            <div className="step-number-badge">02</div>
            <h4>Audit & Clean</h4>
            <p>
              Identify zero-variance constants, leaking ID keys, extreme outliers, and dirty string categories. Review changes in a live 50-row preview.
            </p>
          </div>

          <div className="step-card">
            <div className="step-number-badge">03</div>
            <h4>Feature Engineering</h4>
            <p>
              Select classification or regression targets. Apply Robust or MinMax scalers, target or one-hot encodings, strictly fitted on training splits.
            </p>
          </div>

          <div className="step-card">
            <div className="step-number-badge">04</div>
            <h4>Export & Train</h4>
            <p>
              Export cleaned, split CSVs directly for XGBoost, LightGBM, Scikit-Learn, or PyTorch models with reproducible preprocessing recipes.
            </p>
          </div>
        </div>
      </section>

      {/* Feature Deep Dive Section */}
      <section id="features" className="landing-features">
        <div className="section-header">
          <div className="section-pill">Engineered for Precision</div>
          <h2 className="section-heading">Everything Data Scientists Need</h2>
          <p className="section-sub">
            Eliminate repetitive Pandas pipelines and data leakage traps with automated intelligence.
          </p>
        </div>

        <div className="features-bento">
          <div className="feature-bento-large">
            <div className="bento-text">
              <span className="badge-tag amber">Hygiene First</span>
              <h3>Zero Data Leakage Guarantee</h3>
              <p>
                A common mistake in ML is fitting scalers and target encoders on the full dataset before splitting.
                Model Forge guarantees transformers are fit strictly on the train partition and safely applied to validation and test sets.
              </p>
            </div>
            <div className="bento-preview-box">
              <div className="code-mockup">
                <span className="code-comment"># Strict Leakage-Free Fit Pipeline</span>
                <code>pipeline.fit(X_train, y_train)</code>
                <code>X_train_trans = pipeline.transform(X_train)</code>
                <code>X_test_trans = pipeline.transform(X_test)  # No leakage!</code>
              </div>
            </div>
          </div>

          <div className="feature-bento-card">
            <div className="feature-icon-circle">🤖</div>
            <h3>Ollama Local AI Copilot</h3>
            <p>
              Run inference locally without sending confidential dataset rows to cloud providers. Ask questions about your features and receive actionable recommendations.
            </p>
          </div>

          <div className="feature-bento-card">
            <div className="feature-icon-circle">📊</div>
            <h3>Custom EDA & Chart Studio</h3>
            <p>
              Plot box plots, bivariate scatters, histograms, and correlation heatmaps generated on the fly via Matplotlib, Seaborn, and SVG charts.
            </p>
          </div>
        </div>
      </section>

      {/* Audience Tagline Strip (from reference mock-up) */}
      <section className="landing-audience">
        <div className="audience-container">
          <span className="audience-tag">Built for ML Engineers</span>
          <span className="audience-sep">|</span>
          <span className="audience-tag">Data Scientists</span>
          <span className="audience-sep">|</span>
          <span className="audience-tag">Researchers</span>
          <span className="audience-sep">|</span>
          <span className="audience-tag">AI Innovators</span>
        </div>
      </section>

      {/* Bottom CTA Banner */}
      <section className="landing-cta-banner">
        <div className="cta-banner-inner">
          <h2>Ready to Forge Your Next Dataset?</h2>
          <p>Launch the studio, drop your CSV or Excel file, and audit your data quality in seconds.</p>
          <button className="btn-hero-primary" onClick={handleLaunch}>
            <span>Launch Model Forge Studio</span>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <div className="brand-icon-box small">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="M12 2v8M4.93 10.93 9.17 15.17M2 18h20M20 18v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2M7 14h10l-2-4H9l-2 4Z" />
              </svg>
            </div>
            <span className="footer-title">Model Forge</span>
            <span className="footer-copy">© 2026 Model Forge. Intelligent Preprocessing & Feature Engineering Studio.</span>
          </div>
          <div className="footer-links">
            <a href="#hero">Back to Top</a>
            <a href="#features">Features</a>
            <a href="#pipeline">Pipeline</a>
            <span className="footer-status">● Local Server Ready</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
