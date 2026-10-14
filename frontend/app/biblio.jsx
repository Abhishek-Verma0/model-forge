"use client";

// Bibliometrics (plan §14): upload a Scopus / Web of Science / PubMed / generic CSV
// export, then publication trends, productivity, citations and impact indices,
// collaboration / keyword / co-citation / coupling networks, thematic map and
// evolution, three-field plot, word cloud and trending topics; PDF + Excel report.
import { useEffect, useState } from "react";
import { call, download, fmt } from "./apiclient";
import { BarPlot, COLORS, Figure, LinePlot } from "./chartkit";

const TABS = [["overview", "Overview"], ["productivity", "Productivity"], ["citations", "Citations & impact"],
              ["network", "Networks"], ["themes", "Themes"], ["words", "Words & trends"]];
const NETWORKS = [["keyword", "Keyword co-occurrence"], ["coauthor", "Co-authorship"], ["country", "Country collaboration"],
                  ["cocitation", "Co-citation (references cited together)"], ["coupling", "Bibliographic coupling (documents sharing references)"]];
const FIELDS = [["keywords", "author keywords"], ["index_keywords", "index keywords / MeSH"], ["title", "title words"]];

function NetworkPlot({ g, title }) {
  const W = 680, H = 520, pad = 30;
  const X = (v) => pad + v * (W - 2 * pad), Y = (v) => pad + v * (H - 2 * pad);
  const wmax = Math.max(1, ...g.edges.map((e) => e.weight));
  const smax = Math.max(1, ...g.nodes.map((n) => n.size));
  const labelled = new Set([...g.nodes].sort((a, b) => b.size - a.size).slice(0, 30).map((n) => n.id));
  return (
    <Figure title={title} width={W} height={H}
            note={`${g.nodes.length} nodes · ${g.edges.length} links · ${g.communities} communities (colours, Newman leading eigenvector) · modularity ${g.modularity}. Node size = ${g.size_means}; line width = link strength. Layout: Fruchterman-Reingold.`}>
      {g.edges.map((e, i) => {
        const a = g.nodes[e.source], b = g.nodes[e.target];
        return <line key={i} x1={X(a.x)} y1={Y(a.y)} x2={X(b.x)} y2={Y(b.y)} stroke="#9ca3af" strokeWidth={0.4 + 2.5 * (e.weight / wmax)} opacity="0.45" />;
      })}
      {g.nodes.map((n) => (
        <circle key={n.id} cx={X(n.x)} cy={Y(n.y)} r={3 + 11 * Math.sqrt(n.size / smax)} fill={COLORS[n.community % COLORS.length]} opacity="0.85">
          <title>{`${n.label} (${n.size})`}</title>
        </circle>
      ))}
      {g.nodes.filter((n) => labelled.has(n.id)).map((n) => (
        <text key={`t${n.id}`} x={X(n.x)} y={Y(n.y) - 5 - 11 * Math.sqrt(n.size / smax)} fontSize="9" textAnchor="middle" fill="currentColor">
          {n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label}
        </text>
      ))}
    </Figure>
  );
}

function ThematicMap({ tm }) {
  const W = 640, H = 460, pad = 50;
  const cs = tm.themes.map((t) => t.centrality), ds = tm.themes.map((t) => t.density);
  const span = (v) => [Math.min(...v), Math.max(...v)];
  const [c0, c1] = span(cs), [d0, d1] = span(ds);
  const X = (v) => pad + ((v - c0) / ((c1 - c0) || 1)) * (W - 2 * pad), Y = (v) => H - pad - ((v - d0) / ((d1 - d0) || 1)) * (H - 2 * pad);
  const fmax = Math.max(1, ...tm.themes.map((t) => t.frequency));
  return (
    <Figure title={`Thematic map (${tm.field.replace("_", " ")}, terms in ${tm.min_freq}+ documents)`} width={W} height={H} note={tm.note}>
      <line x1={X(tm.median_centrality)} x2={X(tm.median_centrality)} y1={pad / 2} y2={H - pad} stroke="currentColor" strokeDasharray="4 4" opacity="0.4" />
      <line x1={pad} x2={W - pad / 2} y1={Y(tm.median_density)} y2={Y(tm.median_density)} stroke="currentColor" strokeDasharray="4 4" opacity="0.4" />
      <g fontSize="10" fill="currentColor" opacity="0.6">
        <text x={W - pad} y={pad / 2 + 10} textAnchor="end">motor themes</text>
        <text x={pad} y={pad / 2 + 10}>niche themes</text>
        <text x={pad} y={H - pad - 6}>emerging / declining</text>
        <text x={W - pad} y={H - pad - 6} textAnchor="end">basic themes</text>
        <text x={W / 2} y={H - 12} textAnchor="middle">centrality (links to other themes) →</text>
        <text x={14} y={H / 2} textAnchor="middle" transform={`rotate(-90 14 ${H / 2})`}>density (internal links) →</text>
      </g>
      {tm.themes.map((t, k) => (
        <g key={t.id}>
          <circle cx={X(t.centrality)} cy={Y(t.density)} r={6 + 22 * Math.sqrt(t.frequency / fmax)} fill={COLORS[k % COLORS.length]} opacity="0.55">
            <title>{`${t.words.slice(0, 12).join(", ")}`}</title>
          </circle>
          <text x={X(t.centrality)} y={Y(t.density) + 3} fontSize="9.5" textAnchor="middle" fill="currentColor">{t.label}</text>
        </g>
      ))}
    </Figure>
  );
}

function Evolution({ ev }) {
  const cols = ev.periods.length, W = Math.max(640, cols * 230), rowH = 34, top = 34;
  const maxThemes = Math.max(1, ...ev.periods.map((p) => p.themes.length));
  const H = top + maxThemes * rowH + 30;
  const colX = (i) => 20 + i * ((W - 200) / Math.max(cols - 1, 1));
  const pos = {};
  ev.periods.forEach((p, i) => p.themes.forEach((t, j) => { pos[`${i}:${t.id}`] = [colX(i), top + j * rowH]; }));
  return (
    <Figure title="Thematic evolution" width={W} height={H}
            note="Each column is a period's themes; a line joins themes that share terms (width = inclusion index, shared terms / the smaller theme's terms).">
      {ev.links.map((l, i) => {
        const a = pos[`${l.from_period}:${l.from}`], b = pos[`${l.to_period}:${l.to}`];
        if (!a || !b) return null;
        return <path key={i} d={`M${a[0] + 170},${a[1] + 12} C${(a[0] + 170 + b[0]) / 2},${a[1] + 12} ${(a[0] + 170 + b[0]) / 2},${b[1] + 12} ${b[0]},${b[1] + 12}`}
                     fill="none" stroke={COLORS[l.from % COLORS.length]} strokeWidth={1 + 7 * l.inclusion} opacity="0.4">
          <title>{`shared: ${l.shared.join(", ")}`}</title></path>;
      })}
      {ev.periods.map((p, i) => (
        <g key={i}>
          <text x={colX(i)} y={16} fontSize="11" fontWeight="600" fill="currentColor">{p.from}–{p.to} ({p.documents} docs)</text>
          {p.themes.length === 0 && <text x={colX(i)} y={top + 14} fontSize="10" fill="currentColor" opacity="0.6">too few terms</text>}
          {p.themes.map((t, j) => (
            <g key={t.id}>
              <rect x={colX(i)} y={top + j * rowH} width="170" height="24" rx="4" fill={COLORS[t.id % COLORS.length]} opacity="0.18" />
              <text x={colX(i) + 6} y={top + j * rowH + 16} fontSize="10" fill="currentColor">{t.label.length > 30 ? t.label.slice(0, 29) + "…" : t.label}</text>
            </g>
          ))}
        </g>
      ))}
    </Figure>
  );
}

function ThreeField({ tf }) {
  const W = 680, H = 460, colW = 12, gap = 6, top = 26, bottom = 10;
  const names = { authors: "Authors", keywords: "Keywords", index_keywords: "Index keywords", source: "Sources", countries: "Countries", title: "Title words" };
  const totals = tf.columns.map((col, c) => col.map((item) => tf.links.filter((l) => (l.from_field === c && l.from === item) || (l.to_field === c && l.to === item))
    .reduce((s, l) => s + l.weight, 0)));
  const scale = Math.min(...totals.map((t) => (H - top - bottom - gap * t.length) / Math.max(1, t.reduce((a, b) => a + b, 0) / (t.length ? 1 : 1))));
  const xs = [140, W / 2 - colW / 2, W - 150];
  const ypos = totals.map((t) => { let y = top; return t.map((v) => { const r = [y, Math.max(v * scale, 3)]; y += r[1] + gap; return r; }); });
  const out = {}, inn = {};
  return (
    <Figure title="Three-field plot" width={W} height={H} note="Bar height = how often the item co-occurs with the next column in the same documents; band width = documents in common.">
      {tf.links.map((l, i) => {
        const a = tf.columns[l.from_field].indexOf(l.from), b = tf.columns[l.to_field].indexOf(l.to);
        const ka = `${l.from_field}:${a}`, kb = `${l.to_field}:${b}`;
        const w = l.weight * scale;
        const ya = ypos[l.from_field][a][0] + (out[ka] || 0) + w / 2, yb = ypos[l.to_field][b][0] + (inn[kb] || 0) + w / 2;
        out[ka] = (out[ka] || 0) + w; inn[kb] = (inn[kb] || 0) + w;
        const x1 = xs[l.from_field] + colW, x2 = xs[l.to_field];
        return <path key={i} d={`M${x1},${ya} C${(x1 + x2) / 2},${ya} ${(x1 + x2) / 2},${yb} ${x2},${yb}`} fill="none"
                     stroke={COLORS[a % COLORS.length]} strokeWidth={Math.max(w, 1)} opacity="0.3"><title>{`${l.from} – ${l.to}: ${l.weight}`}</title></path>;
      })}
      {tf.columns.map((col, c) => (
        <g key={c}>
          <text x={xs[c] + colW / 2} y={14} fontSize="11" fontWeight="600" textAnchor="middle" fill="currentColor">{names[tf.fields[c]] || tf.fields[c]}</text>
          {col.map((item, i) => (
            <g key={item}>
              <rect x={xs[c]} y={ypos[c][i][0]} width={colW} height={ypos[c][i][1]} fill={COLORS[i % COLORS.length]} />
              <text x={c === 2 ? xs[c] + colW + 4 : xs[c] - 4} y={ypos[c][i][0] + ypos[c][i][1] / 2 + 3} fontSize="9"
                    textAnchor={c === 2 ? "start" : "end"} fill="currentColor">{item.length > 22 ? item.slice(0, 21) + "…" : item}</text>
            </g>
          ))}
        </g>
      ))}
    </Figure>
  );
}

function Trends({ tr }) {
  if (!tr.terms.length) return <p className="note">No term appears in {tr.min_freq}+ documents — lower the minimum.</p>;
  const ys = tr.terms.flatMap((t) => [t.q1, t.q3]);
  const lo = Math.floor(Math.min(...ys)), hi = Math.ceil(Math.max(...ys)) || lo + 1;
  const W = 640, left = 170, rowH = 18, H = 40 + tr.terms.length * rowH;
  const X = (v) => left + ((v - lo) / ((hi - lo) || 1)) * (W - left - 20);
  const fmax = Math.max(...tr.terms.map((t) => t.frequency));
  return (
    <Figure title="Trending topics" width={W} height={H}
            note={`Line = middle half of the publication years (first to third quartile), dot = median year, dot size = documents. Terms in ${tr.min_freq}+ documents.`}>
      {tr.terms.map((t, i) => (
        <g key={t.term}>
          <text x={left - 6} y={14 + i * rowH + 4} fontSize="10" textAnchor="end" fill="currentColor">{t.term}</text>
          <line x1={X(t.q1)} x2={X(t.q3)} y1={14 + i * rowH} y2={14 + i * rowH} stroke={COLORS[0]} strokeWidth="2" opacity="0.5" />
          <circle cx={X(t.median)} cy={14 + i * rowH} r={3 + 5 * Math.sqrt(t.frequency / fmax)} fill={COLORS[0]}><title>{`${t.term}: ${t.frequency} documents`}</title></circle>
        </g>
      ))}
      {Array.from({ length: Math.min(hi - lo, 12) + 1 }, (_, k) => lo + Math.round(k * (hi - lo) / Math.min(hi - lo, 12))).map((y) => (
        <text key={y} x={X(y)} y={H - 6} fontSize="9" textAnchor="middle" fill="currentColor">{y}</text>
      ))}
    </Figure>
  );
}

function Upload({ onOpened }) {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  async function send() {
    setBusy(true); setErr("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      onOpened(await call("/api/biblio/upload", fd));
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }
  return (
    <div className="panel" style={{ padding: 18 }}>
      <p style={{ marginTop: 0 }}>Upload a publication export you are licensed to use. The format is recognised from the file itself:</p>
      <ul style={{ marginTop: 0 }}>
        <li><b>Scopus</b>: Export → CSV (include citation information, abstracts &amp; keywords, references)</li>
        <li><b>Web of Science</b>: Export → Plain text file or Tab delimited file (Full record and cited references)</li>
        <li><b>PubMed</b>: Save → Format: PubMed (no citation counts)</li>
        <li><b>Dimensions / OpenAlex / Lens</b> CSV exports, matched by column name (title, year, source, authors, citations …)</li>
      </ul>
      <input type="file" accept=".csv,.txt,.tsv,.nbib" onChange={(e) => setFile(e.target.files?.[0] || null)} />{" "}
      <button className="primary" disabled={!file || busy} onClick={send}>{busy ? "Reading…" : "Analyse"}</button>
      {err && <div className="error" style={{ marginTop: 8 }}>{err}</div>}
    </div>
  );
}

export default function Bibliometrics({ id, onOpened }) {
  const [summary, setSummary] = useState(null);
  const [tab, setTab] = useState("overview");
  const [res, setRes] = useState({});        // analysis key -> latest result
  const [opts, setOpts] = useState({ kind: "keyword", top_n: 50, min_weight: 1, field: "keywords", min_freq: 5, cuts: "", n: 10 });
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");

  useEffect(() => {
    setSummary(null); setRes({}); setErr("");
    if (!id) return;
    call(`/api/biblio/summary?id=${id}`).then(setSummary).catch((e) => setErr(e.message));
  }, [id]);

  async function run(analysis, params, key = analysis) {
    setBusy(key); setErr("");
    try { const r = await call("/api/biblio/run", { id, analysis, params }); setRes((x) => ({ ...x, [key]: r })); }
    catch (e) { setErr(e.message); } finally { setBusy(""); }
  }
  useEffect(() => {
    if (!id || !summary) return;
    if (tab === "productivity" && !res.productivity) run("productivity", {});
    if (tab === "citations" && !res.citations) run("citations", {});
    if (tab === "words" && !res.wordcloud) { run("wordcloud", { field: opts.field }); run("trends", { field: opts.field, min_freq: opts.min_freq }); }
  }, [tab, summary]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!id) {
    return (
      <div className="page">
        <header className="page-head">
          <h1>Bibliometric analysis</h1>
          <p>Publication trends, citations, collaboration and keyword networks from a database export.</p>
        </header>
        <Upload onOpened={onOpened} />
      </div>
    );
  }
  const set = (k, v) => setOpts((o) => ({ ...o, [k]: v }));
  const ov = summary?.overview;
  return (
    <div className="page">
      <header className="page-head page-head-row">
        <h1>{summary?.title || "Publications"}</h1>
        <span className="page-actions">
          <button className="btn btn-quiet btn-sm" onClick={() => onOpened(null)}>Analyse another file</button>
          <button className="btn btn-secondary btn-sm" onClick={() => download(`/api/report/excel?id=${id}`, "bibliometrics.xlsx").catch((e) => setErr(e.message))}>Excel tables</button>
          <button className="btn btn-primary btn-sm" onClick={() => download(`/api/report/pdf?id=${id}`, "bibliometric_report.pdf").catch((e) => setErr(e.message))}>PDF report</button>
        </span>
      </header>
      {summary && <p className="note">Read as <b>{summary.format}</b>{summary.notes.length ? ` · ${summary.notes.join(" · ")}` : ""}.
        Results depend on the completeness of the exported metadata.</p>}
      <div className="tabs">{TABS.map(([k, l]) => <button key={k} className={tab === k ? "tab on" : "tab"} onClick={() => setTab(k)}>{l}</button>)}</div>
      {err && <div className="error">{err}</div>}
      {!summary && !err && <p className="note">Loading…</p>}

      {summary && tab === "overview" && (
        <>
          <div className="cards">
            {[["Documents", ov.documents], ["Timespan", `${ov.timespan[0]}–${ov.timespan[1]}`], ["Sources", ov.sources],
              ["Authors", ov.authors], ["Annual growth", ov.annual_growth_rate == null ? "n/a" : `${ov.annual_growth_rate}%`],
              ["Avg citations / doc", fmt(ov.average_citations_per_doc, 2)], ["Co-authors / doc", fmt(ov.co_authors_per_doc, 2)],
              ["International co-authorship", ov.international_coauthorship_percent == null ? "n/a" : `${ov.international_coauthorship_percent}%`]]
              .map(([k, v]) => <div className="card" key={k}><div className="k">{k}</div><div className="v">{v}</div></div>)}
          </div>
          <div className="panel tablescroll" style={{ padding: 12, marginTop: 10 }}>
            <table><tbody>
              <tr><td>Single-authored documents</td><td>{ov.single_authored_docs}</td><td>Document average age</td><td>{fmt(ov.document_average_age, 1)} years</td></tr>
              <tr><td>References</td><td>{ov.references.toLocaleString()}</td><td>Author keywords</td><td>{ov.author_keywords.toLocaleString()}</td></tr>
              <tr><td>Index keywords</td><td>{ov.index_keywords.toLocaleString()}</td><td>Total citations</td><td>{ov.total_citations?.toLocaleString() ?? "n/a"}</td></tr>
            </tbody></table>
            <p className="note" style={{ marginBottom: 0 }}>Field coverage: {Object.entries(summary.completeness).map(([k, v]) => `${k.replace("_", " ")} ${v}%`).join(" · ")}</p>
            <p className="note" style={{ margin: 0 }}>Annual growth = compound rate between the first and the last year.</p>
          </div>
          {summary.annual.years.length > 0 && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 12 }}>
              <LinePlot title="Annual scientific production" xLabel="year" yLabel="documents"
                        series={[{ label: "documents", x: summary.annual.years, y: summary.annual.documents }]} />
              <LinePlot title="Average citations per year" xLabel="publication year" yLabel="citations per document per year"
                        series={[{ label: "citations / doc / year", x: summary.annual.years, y: summary.annual.mean_citations_per_year, color: COLORS[2] }]}
                        note="Average citations of that year's documents, divided by the years they have had to be cited." />
            </div>
          )}
        </>
      )}

      {summary && tab === "productivity" && res.productivity && (() => {
        const pr = res.productivity;
        return (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 12 }}>
              <BarPlot title="Most productive authors" xLabel="documents" items={pr.authors.map((a) => ({ label: a.name, value: a.documents }))}
                       note={`Fractional counting (each document shared among its authors): ${pr.authors.slice(0, 5).map((a) => `${a.name} ${a.fractional}`).join(", ")}…`} />
              <BarPlot title="Most relevant sources" xLabel="documents" color={COLORS[1]} items={pr.sources.map((s) => ({ label: s.name, value: s.documents }))} />
              {pr.countries.length > 0 && <BarPlot title="Countries (documents with an author there)" xLabel="documents" color={COLORS[2]}
                                                   items={pr.countries.map((c) => ({ label: c.name, value: c.documents }))} />}
              {pr.institutions.length > 0 && <BarPlot title="Institutions" xLabel="documents" color={COLORS[3]}
                                                      items={pr.institutions.map((c) => ({ label: c.name, value: c.documents }))} />}
            </div>
            {pr.countries.length > 0 && (
              <div className="panel tablescroll" style={{ padding: 12, marginTop: 8 }}>
                <table><thead><tr><th>Country</th><th>Documents</th><th>Single-country</th><th>Multi-country</th></tr></thead>
                  <tbody>{pr.countries.map((c) => <tr key={c.name}><td>{c.name}</td><td>{c.documents}</td><td>{c.single_country}</td><td>{c.multi_country}</td></tr>)}</tbody></table>
              </div>
            )}
            <p className="note">{pr.notes.join(" ")}</p>
          </>
        );
      })()}

      {summary && tab === "citations" && res.citations && (() => {
        const ci = res.citations;
        if (!ci.has_citations) return <p className="note">This export carries no citation counts.</p>;
        return (
          <>
            <div className="panel tablescroll" style={{ padding: 12 }}>
              <b>Most-cited documents</b>
              <table><thead><tr><th>Document</th><th>Year</th><th>Source</th><th>Citations</th><th>Per year</th><th>Local</th></tr></thead>
                <tbody>{ci.most_cited_documents.map((d, i) => <tr key={i}><td>{d.first_author ? `${d.first_author}: ` : ""}{d.title}</td><td>{d.year}</td>
                  <td>{d.source}</td><td>{d.citations}</td><td>{d.citations_per_year}</td><td>{d.local_citations}</td></tr>)}</tbody></table>
            </div>
            <div className="panel tablescroll" style={{ padding: 12, marginTop: 8 }}>
              <b>Author impact (within this collection)</b>
              <table><thead><tr><th>Author</th><th>Documents</th><th>Citations</th><th>h-index</th><th>g-index</th><th>i10-index</th><th>m-index</th><th>First year</th></tr></thead>
                <tbody>{ci.author_impact.map((a) => <tr key={a.author}><td>{a.author}</td><td>{a.documents}</td><td>{a.total_citations}</td>
                  <td>{a.h_index}</td><td>{a.g_index}</td><td>{a.i10_index}</td><td>{a.m_index}</td><td>{a.first_year}</td></tr>)}</tbody></table>
            </div>
            <p className="note">{ci.notes.join(" · ")} · local citations matched: {ci.local_citations_matched}.</p>
          </>
        );
      })()}

      {summary && tab === "network" && (
        <>
          <div className="panel" style={{ padding: 12 }}>
            <div className="builder-row" style={{ alignItems: "flex-end" }}>
              <label className="pp-field" style={{ minWidth: 260 }}><span>Network</span>
                <select value={opts.kind} onChange={(e) => set("kind", e.target.value)}>{NETWORKS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label>
              {opts.kind === "keyword" && <label className="pp-field"><span>Terms</span>
                <select value={opts.field} onChange={(e) => set("field", e.target.value)}>{FIELDS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label>}
              <label className="pp-field" style={{ maxWidth: 110 }}><span>Top nodes</span><input type="number" min="2" max="300" value={opts.top_n} onChange={(e) => set("top_n", e.target.value)} /></label>
              <label className="pp-field" style={{ maxWidth: 140 }}><span>Min link strength</span><input type="number" min="1" value={opts.min_weight} onChange={(e) => set("min_weight", e.target.value)} /></label>
              <button className="primary" disabled={busy === "network"} onClick={() => run("network", { kind: opts.kind, top_n: opts.top_n, min_weight: opts.min_weight, field: opts.field })}>
                {busy === "network" ? "Building…" : "Draw network"}</button>
            </div>
          </div>
          {res.network && <NetworkPlot g={res.network} title={NETWORKS.find(([k]) => k === res.network.kind)?.[1] || res.network.kind} />}
        </>
      )}

      {summary && tab === "themes" && (
        <>
          <div className="panel" style={{ padding: 12 }}>
            <div className="builder-row" style={{ alignItems: "flex-end" }}>
              <label className="pp-field"><span>Terms</span>
                <select value={opts.field} onChange={(e) => set("field", e.target.value)}>{FIELDS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label>
              <label className="pp-field" style={{ maxWidth: 150 }}><span>Min documents per term</span><input type="number" min="1" value={opts.min_freq} onChange={(e) => set("min_freq", e.target.value)} /></label>
              <button className="primary" disabled={!!busy} onClick={() => run("thematic", { field: opts.field, min_freq: opts.min_freq })}>Thematic map</button>
              <label className="pp-field" style={{ maxWidth: 200 }}><span>Cut years (e.g. 2015, 2020)</span><input value={opts.cuts} placeholder="middle year" onChange={(e) => set("cuts", e.target.value)} /></label>
              <button className="ghost" disabled={!!busy} onClick={() => run("evolution", { field: opts.field, min_freq: Math.max(1, Math.round(opts.min_freq / 2)),
                cuts: opts.cuts ? opts.cuts.split(",").map((x) => Number(x.trim())).filter((x) => !Number.isNaN(x)) : null })}>Thematic evolution</button>
              <label className="pp-field" style={{ maxWidth: 110 }}><span>Items / column</span><input type="number" min="2" max="50" value={opts.n} onChange={(e) => set("n", e.target.value)} /></label>
              <button className="ghost" disabled={!!busy} onClick={() => run("three_field", { n: opts.n, fields: ["authors", opts.field === "title" ? "keywords" : opts.field, "source"] })}>Three-field plot</button>
            </div>
          </div>
          {res.thematic && <ThematicMap tm={res.thematic} />}
          {res.thematic && (
            <div className="panel tablescroll" style={{ padding: 12, marginTop: 8 }}>
              <table><thead><tr><th>Theme</th><th>Quadrant</th><th>Documents (sum)</th><th>Centrality</th><th>Density</th><th>Terms</th></tr></thead>
                <tbody>{res.thematic.themes.map((t) => <tr key={t.id}><td>{t.label}</td><td>{t.quadrant}</td><td>{t.frequency}</td>
                  <td>{fmt(t.centrality, 2)}</td><td>{fmt(t.density, 2)}</td><td className="note">{t.words.slice(0, 12).join(", ")}</td></tr>)}</tbody></table>
            </div>
          )}
          {res.evolution && <Evolution ev={res.evolution} />}
          {res.three_field && <ThreeField tf={res.three_field} />}
        </>
      )}

      {summary && tab === "words" && (
        <>
          <div className="panel" style={{ padding: 12 }}>
            <div className="builder-row" style={{ alignItems: "flex-end" }}>
              <label className="pp-field"><span>Terms</span>
                <select value={opts.field} onChange={(e) => set("field", e.target.value)}>{FIELDS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label>
              <label className="pp-field" style={{ maxWidth: 170 }}><span>Trend: min documents</span><input type="number" min="1" value={opts.min_freq} onChange={(e) => set("min_freq", e.target.value)} /></label>
              <button className="primary" disabled={!!busy} onClick={() => { run("wordcloud", { field: opts.field }); run("trends", { field: opts.field, min_freq: opts.min_freq }); }}>Update</button>
            </div>
          </div>
          {res.wordcloud && (
            <div className="panel" style={{ padding: 16, marginTop: 8, display: "flex", flexWrap: "wrap", gap: "6px 14px", alignItems: "baseline", justifyContent: "center" }}>
              {(() => {
                const max = Math.max(1, ...res.wordcloud.words.map((w) => w.count));
                return res.wordcloud.words.map((w, i) => (
                  <span key={w.word} title={`${w.count} documents`} style={{ fontSize: 12 + 26 * Math.sqrt(w.count / max), color: COLORS[i % COLORS.length], lineHeight: 1.1 }}>{w.word}</span>
                ));
              })()}
            </div>
          )}
          {res.trends && <Trends tr={res.trends} />}
        </>
      )}
    </div>
  );
}
