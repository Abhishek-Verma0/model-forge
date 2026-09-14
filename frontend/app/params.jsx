"use client";

// Editable parameters for one model. Specs, defaults and help text come from the
// backend model registry; the backend validates again before a run.
export default function ParamsForm({ model, values, onChange, disabled }) {
  if (!model.params.length) return <p className="note">No editable parameters — this model is trained with its defaults.</p>;
  const set = (name, v) => onChange({ ...values, [name]: v });

  function input(p) {
    const v = values[p.name] ?? p.default;
    if (p.kind === "bool")
      return <input type="checkbox" checked={!!v} disabled={disabled} onChange={(e) => set(p.name, e.target.checked)} />;
    if (p.kind === "choice")
      return (
        <select value={v} disabled={disabled} onChange={(e) => set(p.name, e.target.value)}>
          {p.choices.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      );
    const text = Array.isArray(v) ? v.join(",") : v === null ? "none" : String(v);
    return (
      <input value={text} disabled={disabled} style={{ width: 120 }}
             onChange={(e) => {
               const t = e.target.value.trim();
               if (p.kind === "layers") return set(p.name, t);
               if ((p.none && t.toLowerCase() === "none") || (p.auto && t.toLowerCase() === "auto")) return set(p.name, t.toLowerCase());
               set(p.name, t === "" ? t : Number(t));
             }} />
    );
  }

  const range = (p) => (p.kind === "choice" || p.kind === "bool" || p.kind === "layers" ? ""
    : `${p.min}–${p.max}${p.none ? " or none" : ""}${p.auto ? " or auto" : ""}`);
  const shown = (d) => (Array.isArray(d) ? d.join(",") : d === null ? "none" : String(d));

  return (
    <div>
      {model.params.map((p) => (
        <div key={p.name} style={{ display: "grid", gridTemplateColumns: "minmax(160px, 220px) 140px 1fr", gap: 10,
                                   alignItems: "center", padding: "6px 0", borderTop: "1px solid rgba(0,0,0,0.06)" }}>
          <span>{p.label}{range(p) && <span className="note"> ({range(p)})</span>}</span>
          <span>{input(p)}</span>
          <span className="note">{p.help} <span style={{ whiteSpace: "nowrap" }}>Default: {shown(p.default)}.</span></span>
        </div>
      ))}
      <button className="ghost" type="button" disabled={disabled} onClick={() => onChange({})} style={{ marginTop: 6 }}>
        Reset to defaults
      </button>
    </div>
  );
}
