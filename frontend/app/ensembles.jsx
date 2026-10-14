"use client";

// Ensembles to add to a run (plan §9.5): voting / stacking / weighted averaging of
// models from the table. The backend checks every combination (soft voting needs
// probabilities, stacking can't keep groups together) and says why it refuses.
export default function EnsemblePicker({ kinds, models, value, onChange, disabled }) {
  const usable = models.filter((m) => !m.baseline && m.compatible);
  const get = (kind) => value.find((e) => e.kind === kind);
  const set = (kind, patch) => {
    const cur = get(kind);
    const next = cur ? { ...cur, ...patch } : { kind, members: [], weights: null, ...patch };
    onChange([...value.filter((e) => e.kind !== kind), next]);
  };
  const toggle = (kind) => (get(kind) ? onChange(value.filter((e) => e.kind !== kind)) : set(kind, {}));
  const weighted = (kind) => kind !== "stacking";

  return (
    <section>
      <p className="sectitle">Ensembles (optional)</p>
      <p className="note">Combine several models into one more row of the results. Members use the parameters set on their cards.</p>
      {Object.entries(kinds).map(([kind, text]) => {
        const e = get(kind);
        return (
          <div key={kind} className="panel" style={{ padding: 12, marginBottom: 8, borderColor: e ? "var(--accent)" : undefined }}>
            <label style={{ display: "flex", gap: 8, alignItems: "baseline", cursor: disabled ? "default" : "pointer" }}>
              <input type="checkbox" checked={!!e} disabled={disabled} onChange={() => toggle(kind)} />
              <b>{text}</b>
            </label>
            {e && (
              <div style={{ marginTop: 8 }}>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 14px" }}>
                  {usable.map((m) => {
                    const i = e.members.indexOf(m.key);
                    return (
                      <label key={m.key} style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 13 }}>
                        <input type="checkbox" checked={i >= 0} disabled={disabled}
                               onChange={() => set(kind, i >= 0
                                 ? { members: e.members.filter((k) => k !== m.key), weights: e.weights && e.weights.filter((_, j) => j !== i) }
                                 : { members: [...e.members, m.key], weights: e.weights && [...e.weights, 1] })} />
                        {m.label}
                        {i >= 0 && weighted(kind) && (
                          <input type="number" min="0" step="0.5" style={{ width: 54 }} disabled={disabled} title="weight"
                                 value={e.weights ? e.weights[i] : 1}
                                 onChange={(ev) => {
                                   const w = e.weights ? [...e.weights] : e.members.map(() => 1);
                                   w[i] = Number(ev.target.value);
                                   set(kind, { weights: w });
                                 }} />
                        )}
                      </label>
                    );
                  })}
                </div>
                <p className="note" style={{ margin: "6px 0 0" }}>
                  {e.members.length < 2 ? "Choose at least 2 models." :
                    weighted(kind) ? "Weights: equal (1) = plain voting / averaging; larger = that model counts more." :
                      "Stacking learns how much to trust each member (5-fold internal CV, so it costs the members x 6)."}
                </p>
              </div>
            )}
          </div>
        );
      })}
    </section>
  );
}
