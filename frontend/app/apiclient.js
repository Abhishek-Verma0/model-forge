// Shared fetch helpers. Every data route needs the session token (the backend's
// owner guard), so nothing calls fetch directly: JSON goes through call(), files
// through download(), images through blobUrl(), the chat stream through stream().
export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export const TOKEN_KEY = "modelforge_token";

export function authHeaders(extra = {}) {
  const token = typeof window !== "undefined" ? localStorage.getItem(TOKEN_KEY) : null;
  return token ? { ...extra, Authorization: `Bearer ${token}` } : { ...extra };
}

async function send(path, init = {}) {
  try {
    return await fetch(`${API}${path}`, { ...init, headers: authHeaders(init.headers) });
  } catch (e) {
    if (e.name === "AbortError") throw e;   // a request the page cancelled on purpose (the chat replaces old questions)
    throw new Error(`Cannot reach the backend at ${API}.`);
  }
}

async function fail(res) {
  let detail = null;
  try { detail = (await res.json()).detail; } catch {}
  if (res.status === 401 && typeof window !== "undefined") window.dispatchEvent(new Event("modelforge:signed-out"));
  if (Array.isArray(detail)) detail = detail.map((d) => d.msg?.replace(/^Value error,\s*/i, "") || JSON.stringify(d)).join(". ");
  return new Error(detail || `Error ${res.status}`);
}

export async function call(path, body, method) {
  let init = {};
  if (body instanceof FormData) init = { method: method || "POST", body };
  else if (body !== undefined) init = { method: method || "POST", headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify(body) };
  else if (method) init = { method };
  const res = await send(path, init);
  if (!res.ok) throw await fail(res);
  try { return await res.json(); } catch { return null; }
}

// Save a backend file (PDF, CSV, joblib ...) under the name the server gives, else `fallback`.
export async function download(path, fallback, body) {
  const res = await send(path, body instanceof FormData ? { method: "POST", body } : {});
  if (!res.ok) throw await fail(res);
  const name = /filename="?([^";]+)"?/.exec(res.headers.get("content-disposition") || "")?.[1] || fallback;
  saveBlob(await res.blob(), name);
}

export function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  Object.assign(document.createElement("a"), { href: url, download: name }).click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// An image from an authenticated route, as an object URL for <img src>.
export async function blobUrl(path) {
  const res = await send(path);
  if (!res.ok) throw await fail(res);
  return URL.createObjectURL(await res.blob());
}

// A streamed response (the chat): the raw Response, after the status check.
export async function stream(path, body, signal) {
  const res = await send(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify(body), signal });
  if (!res.ok) throw await fail(res);
  return res;
}

export const fmt = (v, d = 3) => (v === null || v === undefined ? "n/a" : Number(v).toFixed(d));
export const fmtP = (p) => (p === null || p === undefined ? "n/a" : p < 0.0001 ? "< 0.0001" : Number(p).toFixed(4));
// in a sentence: "p < 0.0001" / "p = 0.0123"
export const pText = (p) => (fmtP(p).startsWith("<") ? `p ${fmtP(p)}` : `p = ${fmtP(p)}`);
export const dur = (s) => (s == null ? "" : s < 90 ? `${Math.round(s)} s` : `${(s / 60).toFixed(1)} min`);
