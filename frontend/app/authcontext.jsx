"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { API } from "./apiclient";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authModalMode, setAuthModalMode] = useState("login"); // "login" | "register"
  const [googleClientId, setGoogleClientId] = useState(
    process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || ""
  );

  // Load user session and backend auth config on mount
  useEffect(() => {
    async function initAuth() {
      // 1. Fetch server auth config (e.g. Google Client ID)
      try {
        const confRes = await fetch(`${API}/api/auth/config`);
        if (confRes.ok) {
          const conf = await confRes.json();
          if (conf.google_client_id) {
            setGoogleClientId(conf.google_client_id);
          }
        }
      } catch (err) {
        console.warn("Could not load auth config from backend:", err);
      }

      // 2. Validate existing token in localStorage
      const savedToken = typeof window !== "undefined" ? localStorage.getItem("modelforge_token") : null;
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
            localStorage.removeItem("modelforge_token");
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
      localStorage.setItem("modelforge_token", authData.access_token);
      setToken(authData.access_token);
      setUser(authData.user);
      setAuthModalOpen(false);
    }
  }

  async function login(email, password) {
    const res = await fetch(`${API}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Failed to log in.");
    }
    handleAuthSuccess(data);
    return data.user;
  }

  async function register(email, password, fullName) {
    const res = await fetch(`${API}/api/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, full_name: fullName }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Failed to register.");
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
      throw new Error(data.detail || "Google authentication failed.");
    }
    handleAuthSuccess(data);
    return data.user;
  }

  function logout() {
    if (typeof window !== "undefined") {
      localStorage.removeItem("modelforge_token");
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
