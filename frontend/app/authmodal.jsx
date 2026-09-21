"use client";

import { useEffect, useRef, useState } from "react";
import { useAuth } from "./authcontext";

export default function AuthModal() {
  const {
    authModalOpen,
    authModalMode,
    setAuthModalMode,
    closeAuthModal,
    login,
    register,
    loginWithGoogle,
    googleClientId,
  } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showGoogleGuide, setShowGoogleGuide] = useState(false);

  const googleBtnRef = useRef(null);

  // Reset fields when opening modal or changing mode
  useEffect(() => {
    if (authModalOpen) {
      setError("");
      setShowGoogleGuide(false);
    }
  }, [authModalOpen, authModalMode]);

  // Load and render Google Identity Services button
  useEffect(() => {
    if (!authModalOpen) return;

    function initGoogleGsi() {
      if (typeof window === "undefined" || !window.google?.accounts?.id) return;

      if (googleClientId && googleClientId.trim().length > 0) {
        try {
          window.google.accounts.id.initialize({
            client_id: googleClientId.trim(),
            callback: async (response) => {
              if (response.credential) {
                setLoading(true);
                setError("");
                try {
                  await loginWithGoogle(response.credential);
                } catch (err) {
                  setError(err.message || "Google sign-in failed.");
                } finally {
                  setLoading(false);
                }
              }
            },
          });

          if (googleBtnRef.current) {
            googleBtnRef.current.innerHTML = "";
            window.google.accounts.id.renderButton(googleBtnRef.current, {
              type: "standard",
              theme: "outline",
              size: "large",
              text: authModalMode === "login" ? "signin_with" : "signup_with",
              shape: "pill",
              logo_alignment: "left",
              width: 320,
            });
          }
        } catch (err) {
          console.warn("Failed to initialize Google Sign-In button:", err);
        }
      }
    }

    // Check if script already on page
    const existingScript = document.getElementById("google-gsi-script");
    if (!existingScript) {
      const script = document.createElement("script");
      script.id = "google-gsi-script";
      script.src = "https://accounts.google.com/gsi/client";
      script.async = true;
      script.defer = true;
      script.onload = initGoogleGsi;
      document.body.appendChild(script);
    } else {
      initGoogleGsi();
    }
  }, [authModalOpen, authModalMode, googleClientId]);

  if (!authModalOpen) return null;

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");

    if (!email || !email.includes("@")) {
      setError("Please enter a valid email address.");
      return;
    }

    if (!password || password.length < 6) {
      setError("Password must be at least 6 characters long.");
      return;
    }

    if (authModalMode === "register") {
      if (password !== confirmPassword) {
        setError("Passwords do not match.");
        return;
      }
    }

    setLoading(true);
    try {
      if (authModalMode === "login") {
        await login(email, password);
      } else {
        await register(email, password, fullName);
      }
    } catch (err) {
      setError(err.message || "Authentication error occurred.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-overlay" onClick={closeAuthModal}>
      <div className="auth-modal" onClick={(e) => e.stopPropagation()}>
        {/* Close Button */}
        <button
          type="button"
          className="auth-modal-close"
          onClick={closeAuthModal}
          aria-label="Close modal"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>

        {/* Brand Header */}
        <div className="auth-header">
          <div className="auth-brand-badge">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v8" />
              <path d="m4.93 10.93 4.24 4.24" />
              <path d="M2 18h20" />
              <path d="M20 18v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2" />
              <path d="M7 14h10l-2-4H9l-2 4Z" />
            </svg>
          </div>
          <h2 className="auth-title">
            {authModalMode === "login" ? "Welcome back to Model Forge" : "Create your Model Forge Account"}
          </h2>
          <p className="auth-subtitle">
            {authModalMode === "login"
              ? "Sign in to access your models, datasets, and pipelines"
              : "Register to start training, auditing, and building ML workflows"}
          </p>
        </div>

        {/* Switcher Tabs */}
        <div className="auth-tabs">
          <button
            type="button"
            className={`auth-tab ${authModalMode === "login" ? "active" : ""}`}
            onClick={() => setAuthModalMode("login")}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`auth-tab ${authModalMode === "register" ? "active" : ""}`}
            onClick={() => setAuthModalMode("register")}
          >
            Create Account
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="auth-error-banner">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>{error}</span>
          </div>
        )}

        {/* Google Sign-In Button */}
        <div className="auth-social-area">
          {googleClientId ? (
            <div ref={googleBtnRef} className="google-btn-container" />
          ) : (
            <div className="google-setup-card">
              <button
                type="button"
                className="btn-google-custom"
                onClick={() => setShowGoogleGuide((prev) => !prev)}
              >
                <svg width="18" height="18" viewBox="0 0 24 24">
                  <path
                    fill="#4285F4"
                    d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.8-2.4 3.65v3h3.88c2.27-2.09 3.665-5.17 3.665-9.09z"
                  />
                  <path
                    fill="#34A853"
                    d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3c-1.08.72-2.45 1.16-4.05 1.16-3.12 0-5.77-2.1-6.72-4.93H1.25v3.09C3.28 21.43 7.35 24 12 24z"
                  />
                  <path
                    fill="#FBBC05"
                    d="M5.28 14.32c-.25-.72-.38-1.49-.38-2.32s.13-1.6.38-2.32V6.59H1.25C.45 8.18 0 9.97 0 12s.45 3.82 1.25 5.41l4.03-3.09z"
                  />
                  <path
                    fill="#EA4335"
                    d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.35 0 3.28 2.57 1.25 6.59l4.03 3.09c.95-2.83 3.6-4.93 6.72-4.93z"
                  />
                </svg>
                <span>{authModalMode === "login" ? "Sign in with Google" : "Sign up with Google"}</span>
              </button>

              {showGoogleGuide && (
                <div className="google-setup-guide">
                  <div className="guide-title">
                    <span>Google OAuth Setup Instructions</span>
                    <button type="button" onClick={() => setShowGoogleGuide(false)}>✕</button>
                  </div>
                  <p>To enable live Google Sign-In:</p>
                  <ol>
                    <li>Go to <a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noreferrer">Google Cloud Console &rarr; Credentials</a></li>
                    <li>Create an <b>OAuth 2.0 Client ID</b> (Web application).</li>
                    <li>Add <code>http://localhost:3000</code> to <i>Authorized JavaScript origins</i>.</li>
                    <li>Paste your Client ID in <code>backend/.env</code>: <br/><code>GOOGLE_CLIENT_ID=your-id.apps.googleusercontent.com</code></li>
                  </ol>
                  <p className="guide-note">Email/password registration below is active and connected directly to PostgreSQL.</p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="auth-divider">
          <span>or continue with email</span>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          {authModalMode === "register" && (
            <div className="auth-field">
              <label htmlFor="auth-name">Full Name</label>
              <input
                id="auth-name"
                type="text"
                placeholder="Alex Morgan"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                autoComplete="name"
              />
            </div>
          )}

          <div className="auth-field">
            <label htmlFor="auth-email">Email address</label>
            <input
              id="auth-email"
              type="email"
              placeholder="alex@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
          </div>

          <div className="auth-field">
            <div className="auth-field-row">
              <label htmlFor="auth-password">Password</label>
              {authModalMode === "login" && (
                <span className="auth-hint">Must be at least 6 characters</span>
              )}
            </div>
            <div className="auth-input-wrapper">
              <input
                id="auth-password"
                type={showPassword ? "text" : "password"}
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete={authModalMode === "login" ? "current-password" : "new-password"}
              />
              <button
                type="button"
                className="auth-eye-btn"
                onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9.88 9.88a3 3 0 1 0 4.24 4.24M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61M2 2l20 20" />
                  </svg>
                ) : (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                )}
              </button>
            </div>
          </div>

          {authModalMode === "register" && (
            <div className="auth-field">
              <label htmlFor="auth-confirm-password">Confirm Password</label>
              <input
                id="auth-confirm-password"
                type={showPassword ? "text" : "password"}
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                autoComplete="new-password"
              />
            </div>
          )}

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={loading}
          >
            {loading ? (
              <span className="auth-spinner"></span>
            ) : authModalMode === "login" ? (
              "Sign In to Model Forge"
            ) : (
              "Create Account"
            )}
          </button>
        </form>

        {/* Footer switch prompt */}
        <div className="auth-footer">
          {authModalMode === "login" ? (
            <p>
              Don&apos;t have an account?{" "}
              <button
                type="button"
                className="auth-link-btn"
                onClick={() => setAuthModalMode("register")}
              >
                Create an account
              </button>
            </p>
          ) : (
            <p>
              Already have an account?{" "}
              <button
                type="button"
                className="auth-link-btn"
                onClick={() => setAuthModalMode("login")}
              >
                Sign in
              </button>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
