"use client";

// Administrator page (plan §8.1 "role-based user and administrator access"): the
// accounts and their state. Administrators manage accounts, never other users' data.
import { useEffect, useState } from "react";
import { call } from "./apiclient";

export default function Admin({ me }) {
  const [users, setUsers] = useState(null);
  const [err, setErr] = useState("");
  const load = () => call("/api/admin/users").then(setUsers).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);

  async function setActive(u, active) {
    if (!active && !window.confirm(`Deactivate ${u.email}? They will be signed out and can't sign in until reactivated.`)) return;
    try { await call(`/api/admin/users/${u.id}/active`, { active }); load(); } catch (e) { setErr(e.message); }
  }

  return (
    <div className="page">
      <header className="page-head">
        <h1>Accounts</h1>
        <p>Administrators are listed in ADMIN_EMAILS on the server. Datasets stay private to their owner, so this page never shows them.</p>
      </header>
      {err && <div className="error">{err}</div>}
      {users && (
        <div className="panel tablescroll" style={{ padding: 12 }}>
          <table>
            <thead><tr><th>Email</th><th>Name</th><th>Sign-in</th><th>Role</th><th>Created</th><th>State</th><th /></tr></thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id}>
                  <td>{u.email}</td><td>{u.full_name}</td><td>{u.auth_provider}</td><td>{u.role}</td>
                  <td>{u.created_at ? new Date(u.created_at).toLocaleDateString() : "—"}</td>
                  <td>{u.is_active ? <span className="pill pill-good">Active</span> : <span className="pill pill-bad">Deactivated</span>}</td>
                  <td>{u.id !== me?.id && (
                    <button className="btn btn-secondary btn-sm" onClick={() => setActive(u, !u.is_active)}>{u.is_active ? "Deactivate" : "Reactivate"}</button>
                  )}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
