"use client";

// My projects (plan §7 step 1, §8.1, §24): every dataset or publication collection
// this account uploaded -- reopen, rename, delete (with everything made from it).
import { useEffect, useState } from "react";
import { call } from "./apiclient";
import { ago, stage } from "./dashboard";
import Icon from "./icons";

const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "—");

export default function Projects({ onOpen, refreshKey }) {
  const [list, setList] = useState(null);
  const [err, setErr] = useState("");
  const [editing, setEditing] = useState(null); // {id, title}

  async function load() {
    try { setList(await call("/api/projects")); setErr(""); } catch (e) { setErr(e.message); }
  }
  useEffect(() => { load(); }, [refreshKey]);

  async function rename() {
    try { await call(`/api/projects/${editing.id}`, { title: editing.title }, "PATCH"); setEditing(null); load(); }
    catch (e) { setErr(e.message); }
  }

  async function remove(p) {
    if (!window.confirm(`Delete “${p.title}” and everything made from it (cleaned data, runs, models, analyses)? This can't be undone.`)) return;
    try { await call(`/api/projects/${p.id}`, undefined, "DELETE"); load(); } catch (e) { setErr(e.message); }
  }

  return (
    <div className="page">
      <header className="page-head">
        <h1>Projects</h1>
        <p>Each upload is a private workspace that only your account can open. Deleting one removes its data and every
          run, model and analysis made from it.</p>
      </header>
      {err && <div className="error">{err}</div>}
      {!list && !err && <div className="dash-list">{[0, 1, 2].map((i) => <span key={i} className="skeleton" style={{ height: 64 }} />)}</div>}
      {list && !list.length && (
        <div className="empty empty-lg">
          <Icon name="folder" size={32} />
          <p><b>No projects yet.</b> Upload a dataset or a publication export to start.</p>
        </div>
      )}
      {list && list.length > 0 && (
        <ul className="project-list">
          {list.map((p) => {
            const st = stage(p);
            return (
              <li key={p.id} className="project-row">
                <span className="dash-row-icon"><Icon name={p.kind === "bibliometric" ? "book" : "table"} size={18} /></span>
                <div className="project-main">
                  {editing?.id === p.id ? (
                    <form className="rename-form" onSubmit={(e) => { e.preventDefault(); rename(); }}>
                      <label className="sr-only" htmlFor={`rename-${p.id}`}>Project name</label>
                      <input id={`rename-${p.id}`} value={editing.title} autoFocus
                             onChange={(e) => setEditing({ ...editing, title: e.target.value })}
                             onKeyDown={(e) => { if (e.key === "Escape") setEditing(null); }} />
                      <button type="submit" className="btn btn-primary btn-sm">Save</button>
                      <button type="button" className="btn btn-quiet btn-sm" onClick={() => setEditing(null)}>Cancel</button>
                    </form>
                  ) : (
                    <>
                      <b className="project-title" title={p.title}>{p.title}</b>
                      <span className="project-meta">
                        {p.title !== p.filename && <>{p.filename} · </>}
                        {p.shape ? `${p.shape[0].toLocaleString()} rows × ${p.shape[1]} columns · ` : ""}
                        {p.runs} run{p.runs === 1 ? "" : "s"} · {p.saved_models} saved model{p.saved_models === 1 ? "" : "s"} · {p.size_mb} MB
                      </span>
                    </>
                  )}
                </div>
                <span className={`pill pill-${st.tone}`}>{st.label}</span>
                <span className="project-when" title={when(p.updated)}>{ago(p.updated)}</span>
                <div className="project-actions">
                  <button type="button" className="btn btn-primary btn-sm" onClick={() => onOpen(p)}>Open</button>
                  <button type="button" className="icon-btn" title="Rename" aria-label={`Rename ${p.title}`}
                          onClick={() => setEditing({ id: p.id, title: p.title })}>
                    <Icon name="pencil" size={16} />
                  </button>
                  <button type="button" className="icon-btn icon-btn-danger" title="Delete" aria-label={`Delete ${p.title}`}
                          onClick={() => remove(p)}>
                    <Icon name="trash" size={16} />
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
