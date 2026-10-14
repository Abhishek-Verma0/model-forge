"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { API, TOKEN_KEY } from "./apiclient";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authModalMode, setAuthModalMode] = useState("login"); // "login" | "register" | "forgot" | "reset"
  // The server decides: it only accepts Google tokens issued to ITS client id, so the
  // button shows only when the backend has one (GOOGLE_CLIENT_ID in backend/.env).
  const [googleClientId, setGoogleClientId] = useState("");
  const [passwordReset, setPasswordReset] = useState(false);
  const [resetToken, setResetToken] = useState("");

  // Load user session and backend auth config on mount
  useEffect(() => {
    async function initAuth() {
      // 1. Fetch server auth config (e.g. Google Client ID)
      try {
        const confRes = await fetch(`${API}/api/auth/config`);
        if (confRes.ok) {
          const conf = await confRes.json();
          setGoogleClientId(conf.google_client_id || "");
          setPasswordReset(!!conf.password_reset);
        }
      } catch (err) {
        console.warn("Could not load auth config from backend:", err);
      }

      // A password-reset link (?reset=<token>) opens the reset form; the token leaves the address bar.
      const link = typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("reset") : null;
      if (link) {
        setResetToken(link);
        setAuthModalMode("reset");
        setAuthModalOpen(true);
        window.history.replaceState(null, "", window.location.pathname);
      }

      // 2. Validate existing token in localStorage
      const savedToken = typeof window !== "undefined" ? localStorage.getItem(TOKEN_KEY) : null;
      if (savedToken) {
        try {
          const res = await fetch(`${API}/api/auth/me`, {
            headers: { Authorization: `Bearer ${savedToken}` },
          });
          if (res.ok) {
            const userData = await res.json();
            setUser(userData);
            setToken(savedToken);
          } else {
            // Token expired or invalid
            localStorage.removeItem(TOKEN_KEY);
            setUser(null);
            setToken(null);
          }
        } catch {
          // If server temporarily unreachable, keep token to try again or clear
          // but do not crash
        }
      }
      setIsLoading(false);
    }

    initAuth();
    // any request answered 401 (session expired or revoked) signs out everywhere
    const out = () => { localStorage.removeItem(TOKEN_KEY); setToken(null); setUser(null); };
    window.addEventListener("modelforge:signed-out", out);
    return () => window.removeEventListener("modelforge:signed-out", out);
  }, []);

  function openAuthModal(mode = "login") {
    setAuthModalMode(mode);
    setAuthModalOpen(true);
  }

  function closeAuthModal() {
    setAuthModalOpen(false);
  }

  function handleAuthSuccess(authData) {
    if (authData.access_token && authData.user) {
      localStorage.setItem(TOKEN_KEY, authData.access_token);
      setToken(authData.access_token);
      setUser(authData.user);
      setAuthModalOpen(false);
    }
  }

  function formatAuthError(detail, fallback) {
    if (!detail) return fallback;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => (d.msg ? d.msg.replace(/^Value error,\s*/i, "") : JSON.stringify(d)))
        .join(". ");
    }
    if (typeof detail === "object" && detail.msg) {
      return detail.msg.replace(/^Value error,\s*/i, "");
    }
    return String(detail);
  }

  async function login(email, password) {
    const res = await fetch(`${API}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(formatAuthError(data.detail, "Invalid email or password."));
    }
    handleAuthSuccess(data);
    return data.user;
  }

  async function register(email, password, fullName) {
    const res = await fetch(`${API}/api/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: email.trim().toLowerCase(),
        password,
        full_name: fullName ? fullName.trim() : undefined,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(formatAuthError(data.detail, "Failed to register account."));
    }
    handleAuthSuccess(data);
    return data.user;
  }

  async function loginWithGoogle(credential) {
    const res = await fetch(`${API}/api/auth/google`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credential }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(formatAuthError(data.detail, "Google authentication failed."));
    }
    handleAuthSuccess(data);
    return data.user;
  }

  async function forgotPassword(email) {
    const res = await fetch(`${API}/api/auth/forgot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: email.trim().toLowerCase() }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(formatAuthError(data.detail, "Could not send the reset email."));
    return data.detail;
  }

  async function resetPassword(password) {
    const res = await fetch(`${API}/api/auth/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: resetToken, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(formatAuthError(data.detail, "Could not reset the password."));
    setResetToken("");
    handleAuthSuccess(data);
    return data.user;
  }

  function logout() {
    if (typeof window !== "undefined") {
      localStorage.removeItem(TOKEN_KEY);
    }
    setToken(null);
    setUser(null);
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        isLoading,
        authModalOpen,
        authModalMode,
        openAuthModal,
        closeAuthModal,
        setAuthModalMode,
        login,
        register,
        loginWithGoogle,
        logout,
        googleClientId,
        passwordReset,
        forgotPassword,
        resetPassword,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
