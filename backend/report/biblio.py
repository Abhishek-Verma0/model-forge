"""The bibliometric summary report (plan §14 "summary report", §15): PDF and Excel
of a publication collection -- computed fresh from the records, plus the
network / thematic results the user ran (kept in the collection's history)."""

import datetime
import io

import pandas as pd

from bibliometrics import analysis, networks, parse
from core import store
from report import charts as C
from report.layout import Document


def _f(v, d=2):
    return "n/a" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.{d}f}")


def pdf(ds_id):
    m = store.meta(ds_id) or {}
    df = store.load_frame(ds_id)
    if df is None or m.get("kind") != "bibliometric":
        raise LookupError("Publication collection not found.")
    ov, an = analysis.overview(df), analysis.annual(df)
    comp = parse.completeness(df)
    doc = Document(m.get("title") or "Publications", "Bibliometric report - ResearchAI Studio")
    doc.cover([f"Source file: {m.get('filename')} ({m.get('format')})", f"Documents: {ov['documents']:,}",
               f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}", "",
               "Results depend on the completeness and consistency of the exported metadata (field coverage below)."])
    doc.heading("1. Main information")
    doc.table(["Measure", "Value"], [
        ["Timespan", f"{ov['timespan'][0]} - {ov['timespan'][1]}"], ["Sources (journals, books)", ov["sources"]],
        ["Documents", ov["documents"]], ["Annual growth rate %", _f(ov["annual_growth_rate"])],
        ["Document average age (years)", _f(ov["document_average_age"])],
        ["Average citations per document", _f(ov["average_citations_per_doc"])],
        ["References", ov["references"]], ["Author keywords", ov["author_keywords"]],
        ["Index keywords", ov["index_keywords"]], ["Authors", ov["authors"]],
        ["Single-authored documents", ov["single_authored_docs"]], ["Co-authors per document", _f(ov["co_authors_per_doc"])],
        ["International co-authorship %", _f(ov["international_coauthorship_percent"])]], [0.6, 0.4])
    doc.para("Field coverage (% of documents with the field): " +
             ", ".join(f"{k} {v:g}%" for k, v in comp.items()), 8)
    doc.heading("2. Annual scientific production")
    if an["years"]:
        doc.chart(C.lines(an["years"], [("documents", an["documents"], None)], "Documents per year", "year", "documents"), 2.4)
        doc.chart(C.lines(an["years"], [("mean citations per document per citable year", an["mean_citations_per_year"], None)],
                          "Average citations per year", "year", "citations"), 2.2)
    pr = analysis.productivity(df)
    doc.heading("3. Most productive authors, sources, countries and institutions")
    doc.charts([C.bars([a["name"] for a in pr["authors"]], [a["documents"] for a in pr["authors"]], "Authors", "documents"),
                C.bars([s["name"] for s in pr["sources"]], [s["documents"] for s in pr["sources"]], "Sources", "documents", color=1)],
               3.0, left=1.5)
    if pr["countries"]:
        doc.table(["Country", "Documents", "Single-country", "Multi-country"],
                  [[c["name"], c["documents"], c["single_country"], c["multi_country"]] for c in pr["countries"]],
                  [0.4, 0.2, 0.2, 0.2])
    if pr["institutions"]:
        doc.table(["Institution", "Documents"], [[i["name"], i["documents"]] for i in pr["institutions"]], [0.8, 0.2])
    doc.para(" ".join(pr["notes"]), 7.5, color="#555555")
    ci = analysis.citations(df)
    doc.heading("4. Citations and impact")
    if ci["has_citations"]:
        doc.table(["Most-cited documents", "Year", "Citations", "Per year", "Local"],
                  [[f"{d['first_author'] or '?'}: {d['title']}", d["year"], d["citations"], d["citations_per_year"],
                    d["local_citations"]] for d in ci["most_cited_documents"]], [0.6, 0.08, 0.11, 0.11, 0.1], size=7)
        doc.table(["Author", "Documents", "Citations", "h", "g", "i10", "m"],
                  [[a["author"], a["documents"], a["total_citations"], a["h_index"], a["g_index"], a["i10_index"],
                    a["m_index"]] for a in ci["author_impact"]], [0.36, 0.11, 0.13, 0.08, 0.08, 0.08, 0.16])
        doc.para(" ".join(ci["notes"]), 7.5, color="#555555")
    else:
        doc.para("This export carries no citation counts.")
    doc.heading("5. Keywords and themes")
    wc = analysis.word_cloud(df, "keywords", 30)
    if wc["words"]:
        doc.chart(C.bars([w["word"] for w in wc["words"]], [w["count"] for w in wc["words"]],
                         "Most frequent author keywords", "documents", color=4, top=20), 3.0, left=1.9)
    saved = store.analyses(ds_id, "biblio")
    tm = next((r for r in saved if r.get("analysis") == "thematic"), None)
    if tm is None:
        try:
            tm = networks.thematic_map(df)
        except ValueError:
            tm = None
    if tm:
        doc.chart(C.thematic(tm), 3.6)
        doc.table(["Theme", "Quadrant", "Documents (sum)", "Centrality", "Density"],
                  [[t["label"], t["quadrant"], t["frequency"], _f(t["centrality"]), _f(t["density"])] for t in tm["themes"]],
                  [0.36, 0.26, 0.14, 0.12, 0.12], size=7)
    for r in [r for r in saved if r.get("analysis") == "network"][:4]:
        doc.heading(f"Network: {r['kind']}", 2)
        doc.chart(C.network(r, f"{r['kind']} network - {len(r['nodes'])} nodes, {r['communities']} communities, "
                               f"modularity {r['modularity']}"), 4.2)
    trends = next((r for r in saved if r.get("analysis") == "trends"), None)
    if trends and trends["terms"]:
        doc.heading("Trending topics", 2)
        doc.table(["Term", "Documents", "Q1 year", "Median year", "Q3 year"],
                  [[t["term"], t["frequency"], _f(t["q1"], 0), _f(t["median"], 0), _f(t["q3"], 0)] for t in trends["terms"]],
                  [0.4, 0.15, 0.15, 0.15, 0.15])
    doc.heading("6. Limitations")
    doc.bullets(["Findings depend on the coverage of the exporting database and the completeness of its metadata.",
                 "Counts and indices are within this collection only (not an author's whole career).",
                 "Countries and institutions are read from affiliation text and may miss unusual formats.",
                 "Direct subscription-database access needs separate authorisation and licences."])
    return doc.bytes()


def excel(ds_id):
    df = store.load_frame(ds_id)
    m = store.meta(ds_id) or {}
    if df is None or m.get("kind") != "bibliometric":
        raise LookupError("Publication collection not found.")
    pr, ci = analysis.productivity(df), analysis.citations(df)
    an = analysis.annual(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([analysis.overview(df)]).drop(columns=["document_types"]).T.reset_index().to_excel(
            xw, sheet_name="Main information", index=False, header=["measure", "value"])
        pd.DataFrame({"year": an["years"], "documents": an["documents"], "mean citations": an["mean_citations"],
                      "mean citations per year": an["mean_citations_per_year"]}).to_excel(xw, sheet_name="Annual", index=False)
        for key in ("authors", "sources", "countries", "institutions"):
            pd.DataFrame(pr[key]).to_excel(xw, sheet_name=key.title(), index=False)
        pd.DataFrame(ci["most_cited_documents"]).to_excel(xw, sheet_name="Most cited", index=False)
        pd.DataFrame(ci["author_impact"]).to_excel(xw, sheet_name="Author impact", index=False)
        out = df.copy()
        for f in parse.LIST_FIELDS:
            out[f] = out[f].map("; ".join)
        out.to_excel(xw, sheet_name="Records", index=False)
    return buf.getvalue()
