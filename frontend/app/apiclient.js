// Shared fetch helper for the training / inference components.
export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function call(path, body, method) {
  const token = typeof window !== "undefined" ? localStorage.getItem("modelforge_token") : null;
  const headers = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  let init;
  if (body instanceof FormData) {
    init = { method: method || "POST", headers, body };
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    init = { method: method || "POST", headers, body: JSON.stringify(body) };
  } else if (method) {
    init = { method, headers };
  } else if (token) {
    init = { headers };
  }

  let res;
  try {
    res = await fetch(`${API}${path}`, init);
  } catch {
    throw new Error(`Cannot reach the backend at ${API}.`);
  }
  let data = null;
  try { data = await res.json(); } catch {}
  if (!res.ok) throw new Error(data?.detail || `Error ${res.status}`);
  return data;
}

export const fmt = (v, d = 3) => (v === null || v === undefined ? "n/a" : Number(v).toFixed(d));
export const dur = (s) => (s == null ? "" : s < 90 ? `${Math.round(s)} s` : `${(s / 60).toFixed(1)} min`);
