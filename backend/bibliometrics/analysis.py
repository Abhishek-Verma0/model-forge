"""Publication and citation analysis (plan §14) on the canonical table from
bibliometrics/parse.py. Every figure is computed from the uploaded records only
-- an author's h-index here is within this collection, not their career.

Definitions used:
  annual growth rate   compound: (docs in last year / docs in first year)^(1/(years spanned)) - 1
  h-index              largest h with h documents cited at least h times (Hirsch 2005)
  g-index              largest g with the top g documents cited at least g² times together
                       (Egghe 2006), at most the number of documents
  i10-index            documents cited at least 10 times
  m-index              h / years since the author's first document in the collection (+1)
  citations per year   citations / (current year - publication year + 1)
  fractional count     each document counts 1/number of its authors for each author
  local citations      citations from documents IN the collection, matched by DOI (references
                       without a DOI can't be matched and are not counted)
Institution names are taken from the affiliation part that names one (University,
Institute, College, Hospital, ...) -- a heuristic, shown as such.
"""

import datetime
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOP_N = 20               # our starting point: rows in a ranking
TREND_MIN_FREQ = 5       # our starting point: a keyword needs this many documents to be a trend
TREND_PER_YEAR = 3       # our starting point: trending keywords shown per year
CLOUD_WORDS = 60         # our starting point: words in the word cloud
INSTITUTION_WORDS = ("univ", "college", "institut", "school", "hospital", "academy", "centre", "center",
                     "laborator", "polytechn", "ecole", "faculty", "clinic", "council", "foundation")


def _year_now():
    return datetime.date.today().year


def _explode(df, field):
    s = df[field].explode().dropna()
    return s[s.astype(str).str.len() > 0]


def h_index(cites):
    c = sorted((int(x) for x in cites if x is not None and not pd.isna(x)), reverse=True)
    return sum(1 for i, v in enumerate(c) if v >= i + 1)


def g_index(cites):
    c = np.cumsum(sorted((int(x) for x in cites if x is not None and not pd.isna(x)), reverse=True))
    return int(max((g for g in range(1, len(c) + 1) if c[g - 1] >= g * g), default=0))


def overview(df):
    years = df["year"].dropna().astype(int)
    first, last = (int(years.min()), int(years.max())) if len(years) else (None, None)
    counts = years.value_counts()
    growth = None
    if first is not None and last > first and counts.get(first, 0):
        growth = ((counts.get(last, 0) / counts[first]) ** (1 / (last - first)) - 1) * 100
    n_auth = df["authors"].map(len)
    multi_country = df["countries"].map(len)
    with_country = multi_country > 0
    cites = df["citations"].dropna()
    return {
        "documents": int(len(df)), "timespan": [first, last], "sources": int(df["source"].nunique()),
        "annual_growth_rate": None if growth is None else round(float(growth), 2),
        "document_average_age": round(float((_year_now() - years).mean()), 2) if len(years) else None,
        "average_citations_per_doc": round(float(cites.mean()), 2) if len(cites) else None,
        "total_citations": int(cites.sum()) if len(cites) else None,
        "references": int(_explode(df, "references").nunique()),
        "author_keywords": int(_explode(df, "keywords").nunique()),
        "index_keywords": int(_explode(df, "index_keywords").nunique()),
        "authors": int(_explode(df, "authors").nunique()),
        "single_authored_docs": int((n_auth == 1).sum()),
        "co_authors_per_doc": round(float(n_auth[n_auth > 0].mean()), 2) if (n_auth > 0).any() else None,
        "international_coauthorship_percent": round(float((multi_country[with_country] > 1).mean() * 100), 2)
        if with_country.any() else None,
        "document_types": {str(k): int(v) for k, v in df["doc_type"].value_counts().items()},
    }


def annual(df):
    d = df.dropna(subset=["year"])
    if d.empty:
        return {"years": [], "documents": [], "mean_citations": [], "mean_citations_per_year": []}
    first, last = int(d["year"].min()), int(d["year"].max())
    years = list(range(first, last + 1))
    by = d.groupby(d["year"].astype(int))
    docs = by.size().reindex(years, fill_value=0)
    mean_c = by["citations"].mean().reindex(years)
    per_year = [None if pd.isna(m) else round(float(m) / (_year_now() - y + 1), 3) for y, m in zip(years, mean_c)]
    return {"years": years, "documents": [int(v) for v in docs],
            "mean_citations": [None if pd.isna(m) else round(float(m), 3) for m in mean_c],
            "mean_citations_per_year": per_year}


def institutions(affiliations):
    out = []
    for a in affiliations:
        parts = [p.strip() for p in str(a).split(",")]
        hit = next((p for p in parts if any(w in p.lower() for w in INSTITUTION_WORDS)), None)
        if hit:
            out.append(hit)
    return sorted(set(out))


def productivity(df, top=TOP_N):
    """Most productive authors (documents + fractionalised), sources, countries
    (single- vs multiple-country publications) and institutions."""
    frac = Counter()
    for authors in df["authors"]:
        for a in set(authors):
            frac[a] += 1 / len(set(authors))
    authors = _explode(df, "authors").value_counts().head(top)
    countries = Counter()
    scp, mcp = Counter(), Counter()
    for cs in df["countries"]:
        for c in cs:
            countries[c] += 1
            (mcp if len(cs) > 1 else scp)[c] += 1
    inst = Counter(i for affs in df["affiliations"] for i in institutions(affs))
    return {
        "authors": [{"name": a, "documents": int(n), "fractional": round(frac[a], 3)} for a, n in authors.items()],
        "sources": [{"name": s, "documents": int(n)} for s, n in df["source"].value_counts().head(top).items()],
        "countries": [{"name": c, "documents": n, "single_country": scp[c], "multi_country": mcp[c]}
                      for c, n in countries.most_common(top)],
        "institutions": [{"name": i, "documents": n} for i, n in inst.most_common(top)],
        "notes": ["countries and institutions come from the affiliation text (last part = country; "
                  "the part naming a University / Institute / ... = institution)"],
    }


def _local_citations(df):
    dois = {d: i for i, d in enumerate(df["doi"]) if d}
    local = np.zeros(len(df), dtype=int)
    pat = re.compile(r"10\.\d{4,9}/[^\s;,]+", re.I)
    for refs in df["references"]:
        seen = set()
        for r in refs:
            for d in pat.findall(r):
                d = d.lower().rstrip(".")
                if d in dois and d not in seen:
                    local[dois[d]] += 1
                    seen.add(d)
    return local


def citations(df, top=TOP_N):
    """Most-cited documents (global = the database's count; local = by documents in
    this collection), most-cited authors, and author impact indices."""
    d = df.assign(local=_local_citations(df))
    now = _year_now()
    has_c = d["citations"].notna().any()
    docs = d.sort_values("citations", ascending=False, na_position="last").head(top) if has_c else d.head(0)
    most_cited = [{"title": r.title, "first_author": r.authors[0] if r.authors else None, "year": None if pd.isna(r.year) else int(r.year),
                   "source": r.source, "citations": None if pd.isna(r.citations) else int(r.citations),
                   "citations_per_year": None if pd.isna(r.citations) or pd.isna(r.year) else
                   round(int(r.citations) / (now - int(r.year) + 1), 2), "local_citations": int(r.local), "doi": r.doi}
                  for r in docs.itertuples()]
    per_author = {}
    for r in d.itertuples():
        for a in set(r.authors):
            e = per_author.setdefault(a, {"cites": [], "years": []})
            e["cites"].append(0 if pd.isna(r.citations) else int(r.citations))
            if not pd.isna(r.year):
                e["years"].append(int(r.year))
    impact = []
    for a, e in per_author.items():
        h = h_index(e["cites"])
        first = min(e["years"]) if e["years"] else None
        impact.append({"author": a, "documents": len(e["cites"]), "total_citations": int(sum(e["cites"])),
                       "h_index": h, "g_index": g_index(e["cites"]), "i10_index": int(sum(c >= 10 for c in e["cites"])),
                       "m_index": round(h / (now - first + 1), 3) if first else None, "first_year": first})
    impact.sort(key=lambda x: (-x["h_index"], -x["total_citations"], x["author"]))
    return {"has_citations": bool(has_c), "most_cited_documents": most_cited,
            "most_cited_authors": sorted(impact, key=lambda x: -x["total_citations"])[:top],
            "author_impact": impact[:top],
            "local_citations_matched": int(d["local"].sum()),
            "notes": ["indices are within this collection, not an author's whole career",
                      "local citations are matched by DOI; references without a DOI are not counted"]}


def keyword_list(df, field):
    """Per-document term lists for a field: keywords, index_keywords, or title words
    (lower-cased, English stop words removed -- scikit-learn's list)."""
    if field in ("keywords", "index_keywords"):
        return df[field]
    if field == "title":
        return df["title"].fillna("").map(lambda t: sorted({w for w in re.findall(r"[a-z][a-z-]{2,}", t.lower())
                                                            if w not in ENGLISH_STOP_WORDS}))
    raise ValueError("field must be keywords, index_keywords or title")


def word_cloud(df, field="keywords", n=CLOUD_WORDS):
    counts = Counter(w for ws in keyword_list(df, field) for w in set(ws))
    return {"field": field, "words": [{"word": w, "count": c} for w, c in counts.most_common(n)]}


def trends(df, field="keywords", min_freq=TREND_MIN_FREQ, per_year=TREND_PER_YEAR):
    """Trending topics: each frequent term's publication years (first quartile, median,
    third quartile); per median year the most frequent terms are shown."""
    years = {}
    for ws, y in zip(keyword_list(df, field), df["year"]):
        if pd.isna(y):
            continue
        for w in set(ws):
            years.setdefault(w, []).append(int(y))
    rows = []
    for w, ys in years.items():
        if len(ys) >= min_freq:
            q1, med, q3 = np.percentile(ys, [25, 50, 75])
            rows.append({"term": w, "frequency": len(ys), "q1": float(q1), "median": float(med), "q3": float(q3)})
    by_year = {}
    for r in sorted(rows, key=lambda r: -r["frequency"]):
        by_year.setdefault(round(r["median"]), []).append(r)
    shown = [r for y in sorted(by_year) for r in by_year[y][:per_year]]
    return {"field": field, "min_freq": min_freq, "terms": shown}


if __name__ == "__main__":
    assert h_index([10, 8, 5, 4, 3]) == 4 and h_index([25, 8, 5, 3, 3]) == 3 and h_index([]) == 0
    assert g_index([10, 8, 5, 4, 3]) == 5 and g_index([1, 1, 1]) == 1     # 30 >= 25; 1 >= 1 but 2 < 4
    now = _year_now()
    df = pd.DataFrame({
        "title": ["Deep mapping of science", "Science maps", "Citation networks", "Mapping citation science"],
        "year": pd.array([2015, 2017, 2019, 2019], dtype="Int64"),
        "source": ["J A", "J A", "J B", "J A"],
        "authors": [["Ann", "Bob"], ["Ann"], ["Cid", "Ann", "Bob"], ["Dee"]],
        "affiliations": [["Dept X, University of Pisa, Pisa, Italy", "Lab, MIT, Cambridge, USA"],
                         ["University of Pisa, Pisa, Italy"], ["Inst Y, Tokyo, Japan"], []],
        "countries": [["Italy", "USA"], ["Italy"], ["Japan"], []],
        "keywords": [["mapping", "science"], ["mapping"], ["citation"], ["mapping", "citation"]],
        "index_keywords": [[], [], [], []],
        "citations": pd.array([12, 30, 4, None], dtype="Int64"),
        "references": [["Ref 1, DOI 10.1016/j.b"], [], ["Ref 2, DOI 10.1016/J.A"], []],
        "doi": ["10.1016/j.a", "10.1016/j.b", None, None], "doc_type": ["Article"] * 4})
    ov = overview(df)
    assert ov["documents"] == 4 and ov["timespan"] == [2015, 2019] and ov["single_authored_docs"] == 2
    assert ov["annual_growth_rate"] == round(((2 / 1) ** (1 / 4) - 1) * 100, 2)        # 1 doc -> 2 docs over 4 years
    assert ov["international_coauthorship_percent"] == round(100 / 3, 2)               # 1 of 3 docs with countries
    an = annual(df)
    assert an["years"] == [2015, 2016, 2017, 2018, 2019] and an["documents"] == [1, 0, 1, 0, 2]
    assert an["mean_citations_per_year"][0] == round(12 / (now - 2015 + 1), 3)
    pr = productivity(df)
    ann = next(a for a in pr["authors"] if a["name"] == "Ann")
    assert ann["documents"] == 3 and abs(ann["fractional"] - (1 / 2 + 1 + 1 / 3)) < 1e-3
    assert pr["countries"][0] == {"name": "Italy", "documents": 2, "single_country": 1, "multi_country": 1}
    assert {"name": "University of Pisa", "documents": 2} in pr["institutions"]
    ci = citations(df)
    assert ci["most_cited_documents"][0]["citations"] == 30
    imp = {a["author"]: a for a in ci["author_impact"]}
    assert imp["Ann"]["h_index"] == 3 and imp["Ann"]["total_citations"] == 46 and imp["Ann"]["i10_index"] == 2
    local = {d["doi"]: d["local_citations"] for d in ci["most_cited_documents"]}
    assert local["10.1016/j.b"] == 1 and local["10.1016/j.a"] == 1                                # DOI matched case-insensitively
    wc = word_cloud(df)
    assert wc["words"][0] == {"word": "mapping", "count": 3}
    assert word_cloud(df, "title")["words"][0]["word"] in ("science", "mapping", "citation")
    tr = trends(df, min_freq=2)
    assert {t["term"] for t in tr["terms"]} == {"mapping", "citation"} and tr["terms"][0]["median"] <= tr["terms"][-1]["median"]
    print("bibliometric analysis self-check passed")
