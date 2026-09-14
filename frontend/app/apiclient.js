// Shared fetch helper for the training / inference components.
export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function call(path, body, method) {
  const init = body instanceof FormData
    ? { method: "POST", body }
    : body !== undefined
      ? { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : method ? { method } : undefined;
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
