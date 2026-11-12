"""Read exported publication records (plan §14) into one table, one row per document.

Formats, recognised from the file's own content (never from its name):
  scopus        Scopus "CSV" export: Authors, Title, Year, Source title, Cited by, DOI,
                Affiliations, Author Keywords, Index Keywords, References ... ("; " lists)
  wos_tab       Web of Science "Tab-delimited" export: a header of field tags (PT AU TI SO
                PY TC DE ID C1 CR DI ...)
  wos_plain     Web of Science "Plain text": "PT J" ... tag lines, 3-space continuation
                lines, "ER" ends a record
  pubmed        PubMed "PubMed format" (MEDLINE tags: PMID- TI  - AU  - DP  - JT  - OT  - ...)
  csv           any other CSV whose columns can be matched by name (Dimensions, OpenAlex,
                Lens and similar exports): title / year / source / authors / citations ...
                A one-cell line above the header (Dimensions' "About the data" note) is skipped.
  crossref      Crossref REST API JSON (/works list or one work: title, author, container-title,
                issued, is-referenced-by-count, reference, DOI, type, subject)
  openalex      OpenAlex API JSON (/works list or one work: title, publication_year, authorships,
                primary_location, cited_by_count, referenced_works, keywords, topics, doi, type)

Canonical columns: title, year, source, authors (list), affiliations (list),
countries (list), keywords (author keywords), index_keywords (database keywords /
MeSH), citations (times cited in the exporting database), references (list), doi,
doc_type. Results depend on the completeness of these fields (plan §14), so the
reader reports the share of documents that have each one.

Countries come from the last part of each affiliation ("..., City, Country") --
the same rule bibliometrix uses; a short list of spellings is unified (USA, UK,
China). Affiliations without a country give none.
"""

import csv
import io
import json
import re

import numpy as np
import pandas as pd

LIST_FIELDS = ("authors", "affiliations", "countries", "keywords", "index_keywords", "references")
FIELDS = ("title", "year", "source", "authors", "affiliations", "countries", "keywords", "index_keywords",
          "citations", "references", "doi", "doc_type")
# spellings of the same country seen in exports -> one name
COUNTRY_ALIASES = {"united states": "USA", "united states of america": "USA", "u.s.a.": "USA", "us": "USA",
                   "usa": "USA", "united kingdom": "UK", "england": "UK", "scotland": "UK", "wales": "UK",
                   "northern ireland": "UK", "uk": "UK", "peoples r china": "China",
                   "people's republic of china": "China", "china": "China", "south korea": "South Korea",
                   "korea": "South Korea", "republic of korea": "South Korea", "russian federation": "Russia",
                   "russia": "Russia", "iran": "Iran", "islamic republic of iran": "Iran",
                   "viet nam": "Vietnam", "vietnam": "Vietnam", "turkiye": "Turkey", "turkey": "Turkey"}
# generic CSV: canonical field -> accepted column names (case-insensitive)
ALIASES = {
    "title": ["title", "article title", "document title", "display_name"],
    "year": ["year", "pubyear", "publication year", "publication_year", "py"],
    "source": ["source title", "source", "journal", "source_title", "publication venue", "venue",
               "source title/anthology title", "primary_location.source.display_name", "so"],
    "authors": ["authors", "author", "author names", "authorships.author.display_name", "au"],
    "affiliations": ["affiliations", "author affiliations", "authors affiliations", "institutions",
                     "authorships.institutions.display_name", "c1", "research organizations - standardized",
                     "authors affiliations - name of research organization"],
    # a country column, when the export has one (Dimensions); otherwise read from affiliations
    "countries": ["country of research organization", "authors affiliations - country of research organization",
                  "countries", "authorships.countries"],
    "keywords": ["author keywords", "keywords", "de", "mesh terms"],
    "index_keywords": ["index keywords", "keywords plus", "concepts.display_name", "id"],
    "citations": ["cited by", "times cited", "citations", "cited_by_count", "citation count", "tc", "times_cited"],
    "references": ["references", "cited references", "cr", "referenced_works"],
    "doi": ["doi"],
    "doc_type": ["document type", "type", "publication type", "dt", "pt"],
}
WOS_TAGS = {"TI": "title", "PY": "year", "SO": "source", "AU": "authors", "C1": "affiliations", "DE": "keywords",
            "ID": "index_keywords", "TC": "citations", "CR": "references", "DI": "doi", "DT": "doc_type"}
PUBMED_TAGS = {"TI": "title", "DP": "year", "JT": "source", "AU": "authors", "AD": "affiliations",
               "OT": "keywords", "MH": "index_keywords", "LID": "doi", "PT": "doc_type"}


def _split(value, sep=";"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value).split(sep) if p.strip()]


def country(affiliation):
    """Last comma-separated part of an affiliation, unified; None when it is not a name."""
    part = re.sub(r"\.$", "", str(affiliation).split(",")[-1].strip())
    # postcode and anything before it: WoS "MA 02138 USA" -> "USA", "10115 Germany" -> "Germany"
    part = re.sub(r"^.*\d\s*", "", part).strip()
    if not part or len(part) > 40:
        return None
    return COUNTRY_ALIASES.get(part.lower(), part.title() if part.isupper() and len(part) > 3 else part)


def _decode(raw):
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def detect(text):
    if text.lstrip()[:1] in ("{", "["):
        return "json"
    head = text[:4000]
    first = head.splitlines()[0] if head else ""
    if re.match(r"^(FN |VR |PT )", head) and "\nER" in text:
        return "wos_plain"
    if re.match(r"^PMID- ", head.lstrip()):
        return "pubmed"
    tags = first.split("\t")
    if len(tags) > 5 and {"PT", "AU", "TI", "SO"} <= {t.strip() for t in tags}:
        return "wos_tab"
    low = first.lower()
    if "source title" in low and "cited by" in low and "authors" in low:
        return "scopus"
    return "csv"


def _wos_records(text):
    """Field-tagged records -> list of {tag: [lines]}."""
    records, rec, tag = [], {}, None
    for line in text.splitlines():
        if line.startswith("ER"):
            if rec:
                records.append(rec)
            rec, tag = {}, None
        elif line[:2].strip() and len(line) > 2 and line[2] == " ":
            tag = line[:2]
            rec.setdefault(tag, []).append(line[3:].strip())
        elif line.startswith("   ") and tag:
            rec[tag].append(line[3:].strip())
    return records


def _from_wos(records):
    rows = []
    for r in records:
        joined = {t: " ".join(v) for t, v in r.items()}
        row = {f: joined.get(t) for t, f in WOS_TAGS.items()}
        row["authors"] = r.get("AU", [])
        row["references"] = r.get("CR", [])
        # C1 lines: "[Author; Author] Institution, City, Country." -- one affiliation per line
        row["affiliations"] = [re.sub(r"^\[[^\]]*\]\s*", "", a) for a in r.get("C1", [])]
        row["keywords"] = _split(joined.get("DE"))
        row["index_keywords"] = _split(joined.get("ID"))
        rows.append(row)
    return pd.DataFrame(rows)


def _from_pubmed(text):
    records, rec, tag = [], {}, None
    for line in text.splitlines():
        if not line.strip():
            if rec:
                records.append(rec)
            rec, tag = {}, None
            continue
        m = re.match(r"^([A-Z]{2,4})\s*- (.*)$", line)
        if m:
            tag = m.group(1)
            rec.setdefault(tag, []).append(m.group(2).strip())
        elif line.startswith("      ") and tag:
            rec[tag][-1] += " " + line.strip()
    if rec:
        records.append(rec)
    rows = []
    for r in records:
        dp = " ".join(r.get("DP", []))
        year = re.search(r"\d{4}", dp)
        doi = next((x.split(" ")[0] for x in r.get("LID", []) + r.get("AID", []) if "[doi]" in x), None)
        rows.append({"title": " ".join(r.get("TI", [])), "year": int(year.group()) if year else None,
                     "source": " ".join(r.get("JT", [])) or " ".join(r.get("TA", [])), "authors": r.get("AU", []),
                     "affiliations": r.get("AD", []), "keywords": r.get("OT", []),
                     "index_keywords": [m.split("/")[0].lstrip("*") for m in r.get("MH", [])],
                     "citations": None, "references": [], "doi": doi,
                     "doc_type": "; ".join(r.get("PT", []))})
    return pd.DataFrame(rows)


def _match(columns):
    low = {str(c).strip().lower(): c for c in columns}
    return {f: next((low[a] for a in names if a in low), None) for f, names in ALIASES.items()}


def _from_table(df):
    m = _match(df.columns)
    out = pd.DataFrame({f: (df[c] if c is not None else pd.Series([None] * len(df), index=df.index, dtype=object))
                        for f, c in m.items()})
    # list separators: Scopus / WoS use "; ", OpenAlex CSV uses "|"
    for f in ("authors", "affiliations", "countries", "keywords", "index_keywords", "references"):
        out[f] = [_split(v, "|" if isinstance(v, str) and "|" in v and ";" not in v else ";") for v in out[f]]
    return out, [f for f, c in m.items() if c is None]


def _year(item):
    for key in ("issued", "published-print", "published-online", "published", "created"):
        parts = ((item.get(key) or {}).get("date-parts") or [[None]])[0]
        if parts and parts[0]:
            return parts[0]
    return None


def _crossref_ref(r):
    """One Crossref reference -> one text the co-citation network can key on."""
    if r.get("DOI"):
        return f"DOI {r['DOI'].lower()}"
    if r.get("unstructured"):
        return r["unstructured"]
    return ", ".join(str(r[k]) for k in ("author", "year", "journal-title", "volume", "first-page") if r.get(k)) or None


def _from_crossref(items):
    rows = []
    for it in items:
        authors = [a.get("name") or ", ".join(p for p in (a.get("family"), a.get("given")) if p)
                   for a in it.get("author") or []]
        rows.append({"title": " ".join(it.get("title") or []) or None, "year": _year(it),
                     "source": (it.get("container-title") or [None])[0],
                     "authors": [a for a in authors if a],
                     "affiliations": [f["name"] for a in it.get("author") or [] for f in a.get("affiliation") or []
                                      if f.get("name")],
                     "keywords": [], "index_keywords": list(it.get("subject") or []),
                     "citations": it.get("is-referenced-by-count"),
                     "references": [x for x in (_crossref_ref(r) for r in it.get("reference") or []) if x],
                     "doi": it.get("DOI"), "doc_type": it.get("type")})
    return pd.DataFrame(rows, columns=list(FIELDS))


def _from_openalex(works):
    rows = []
    for w in works:
        ships = w.get("authorships") or []
        raw = [a for s in ships for a in s.get("raw_affiliation_strings") or []]
        inst = [i.get("display_name") for s in ships for i in s.get("institutions") or [] if i.get("display_name")]
        source = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        rows.append({"title": w.get("title") or w.get("display_name"), "year": w.get("publication_year"),
                     "source": source,
                     "authors": [(s.get("author") or {}).get("display_name") for s in ships
                                 if (s.get("author") or {}).get("display_name")],
                     "affiliations": raw or inst,
                     # OpenAlex's own standardised country codes (ISO 3166, e.g. "IT") for every record, so
                     # one file never mixes names read from affiliation text with codes
                     "countries": sorted({c for s in ships for c in s.get("countries") or []}),
                     "keywords": [k.get("display_name") for k in w.get("keywords") or [] if k.get("display_name")],
                     "index_keywords": [t.get("display_name") for t in (w.get("topics") or w.get("concepts") or [])
                                        if t.get("display_name")],
                     "citations": w.get("cited_by_count"),
                     "references": list(w.get("referenced_works") or []),
                     "doi": w.get("doi"), "doc_type": w.get("type")})
    return pd.DataFrame(rows, columns=list(FIELDS))


def _from_json(text):
    """Crossref or OpenAlex API JSON: a list response, one work, or a bare list of works."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"This JSON file can't be read ({exc.msg}, line {exc.lineno}).") from None
    if isinstance(data, dict) and isinstance(data.get("message"), dict):        # Crossref envelope
        msg = data["message"]
        return "crossref", msg.get("items") if "items" in msg else [msg]
    if isinstance(data, dict) and isinstance(data.get("results"), list):        # OpenAlex list response
        return "openalex", data["results"]
    works = data if isinstance(data, list) else [data]
    works = [w for w in works if isinstance(w, dict)]
    if works and all("authorships" in w or "publication_year" in w for w in works):
        return "openalex", works
    if works and all("DOI" in w or "container-title" in w for w in works):
        return "crossref", works
    raise ValueError("This JSON is not a Crossref or OpenAlex works export.")


def _skip_preamble(text):
    """Drop a note line above the header (Dimensions: 'About the data: Exported on ...'):
    the line says so, or has under half the cells of the header line after it."""
    lines = text.split("\n", 2)
    if len(lines) < 3:
        return text
    sep = _sep(lines[1])
    first, second = (len(next(csv.reader([ln], delimiter=sep), [])) for ln in lines[:2])
    note = lines[0].lstrip('﻿"').lower().startswith("about the data")
    return text.split("\n", 1)[1] if note or (second >= 3 and first * 2 < second) else text


def _sep(text):
    first = text.split("\n", 1)[0]
    return "\t" if first.count("\t") > first.count(",") else ","


def read(raw):
    """File bytes -> (DataFrame of canonical columns, format, notes)."""
    text = _decode(raw)
    fmt = detect(text)
    notes = []
    if fmt == "json":
        fmt, works = _from_json(text)
        df = _from_crossref(works) if fmt == "crossref" else _from_openalex(works)
        if fmt == "crossref":
            notes.append("Crossref records carry no author keywords: keyword analyses use the journal subject "
                         "categories (index keywords) when present; references are listed only where the "
                         "publisher deposited them")
        else:
            notes.append("countries are OpenAlex's ISO 3166 codes (e.g. KR = South Korea, GB = United Kingdom)")
    elif fmt in ("wos_plain",):
        df = _from_wos(_wos_records(text))
    elif fmt == "pubmed":
        df = _from_pubmed(text)
        notes.append("PubMed records carry no citation counts or reference lists - citation and co-citation "
                     "results are not available for this file")
    elif fmt == "wos_tab":
        tab = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, index_col=False, quoting=3)
        tab.columns = [c.strip() for c in tab.columns]
        recs = []
        for _, r in tab.iterrows():
            rec = {t: [v] for t, v in r.items() if isinstance(v, str) and v.strip()}
            for t in ("AU", "CR", "C1"):
                if t in rec:
                    # C1: "; " separates affiliations, but not the "; " between names inside [..]
                    rec[t] = _split(rec[t][0]) if t != "C1" else re.split(r";\s+(?![^\[]*\])", rec[t][0])
            recs.append(rec)
        df = _from_wos(recs)
    else:
        text = _skip_preamble(text)
        table = pd.read_csv(io.StringIO(text), sep=_sep(text), dtype=str, keep_default_na=False, na_values=[""])
        df, unmatched = _from_table(table)
        if fmt == "csv":
            if df["title"].isna().all() or df["year"].isna().all():
                raise ValueError("This doesn't look like a publication export: no title / year columns found. "
                                 "Supported: Scopus CSV, Web of Science (plain text or tab-delimited), PubMed, "
                                 "or a CSV with title, year, source, authors, citations columns.")
            notes.append("generic CSV - matched columns by name; not found: " + (", ".join(unmatched) or "none"))
    df = _normalise(df)
    if df.empty:
        raise ValueError("No publication records found in this file.")
    return df, fmt, notes


def _normalise(df):
    for f in FIELDS:
        if f not in df.columns:
            df[f] = None
    for f in LIST_FIELDS:
        df[f] = [_split(v) if not isinstance(v, list) else [str(x).strip() for x in v if str(x).strip()]
                 for v in df[f]]
    df["authors"] = [[a.replace(".", "").strip() for a in v] for v in df["authors"]]
    df["keywords"] = [[k.lower() for k in v] for v in df["keywords"]]
    df["index_keywords"] = [[k.lower() for k in v] for v in df["index_keywords"]]
    df["year"] = pd.to_numeric(df["year"].astype("string").str.extract(r"(\d{4})")[0], errors="coerce").astype("Int64")
    df["citations"] = pd.to_numeric(df["citations"], errors="coerce").astype("Int64")
    # countries the export lists itself are kept; otherwise read from the affiliations
    df["countries"] = [sorted(set(given)) if given else sorted({c for c in (country(a) for a in affs) if c})
                       for given, affs in zip(df["countries"], df["affiliations"])]
    df["doi"] = [str(d).strip().lower().removeprefix("https://doi.org/") if isinstance(d, str) and d.strip() else None
                 for d in df["doi"]]
    for f in ("title", "source", "doc_type"):
        df[f] = df[f].astype("string").str.strip()
    df = df[[f for f in FIELDS]]
    return df[df["title"].notna() | df["year"].notna()].reset_index(drop=True)


def completeness(df):
    """Share of documents that have each field (plan §14: results depend on it)."""
    out = {}
    for f in FIELDS:
        col = df[f]
        has = col.map(lambda v: bool(v)) if f in LIST_FIELDS else col.notna()
        out[f] = round(float(has.mean()) * 100, 1) if len(df) else 0.0
    return out


if __name__ == "__main__":
    scopus = (
        'Authors,Author full names,Title,Year,Source title,Cited by,DOI,Affiliations,Author Keywords,'
        'Index Keywords,References,Document Type\n'
        '"Aria M.; Cuccurullo C.","Aria, Massimo; Cuccurullo, Corrado","bibliometrix: An R-tool",2017,'
        'Journal of Informetrics,1500,10.1016/j.joi.2017.08.007,'
        '"Department of Economics, University of Naples Federico II, Naples, Italy; University of Campania, Caserta, Italy",'
        '"science mapping; bibliometrics","Bibliometrics; Software","Cobo M.J., Science mapping software tools, (2011); '
        'Garfield E., Citation indexes, (1955)",Article\n'
        '"Cobo M.J.; Herrera F.","x","Science mapping software tools",2011,Journal of the American Society for '
        'Information Science and Technology,900,10.1002/asi.21525,"University of Granada, Granada, Spain",'
        '"science mapping; software","",,Article\n')
    df, fmt, notes = read(scopus.encode("utf-8"))
    assert fmt == "scopus" and len(df) == 2 and notes == [], (fmt, notes)
    assert df["authors"][0] == ["Aria M", "Cuccurullo C"] and df["countries"][0] == ["Italy"]
    assert df["keywords"][0] == ["science mapping", "bibliometrics"] and df["citations"].tolist() == [1500, 900]
    assert len(df["references"][0]) == 2 and df["year"].tolist() == [2017, 2011]

    wos = ("FN Clarivate Analytics Web of Science\nVR 1.0\n"
           "PT J\nAU Smith, J\n   Lee, K\nTI A study of\n   bibliometric maps\nSO SCIENTOMETRICS\n"
           "DE citation analysis; co-word\nID NETWORKS\n"
           "C1 [Smith, J] Harvard Univ, Dept Stat, Cambridge, MA 02138 USA.\n   [Lee, K] Seoul Natl Univ, Seoul, South Korea.\n"
           "CR Garfield E, 1955, SCIENCE, V122, P108, DOI 10.1126/science.122.3159.108\n   Small H, 1973, J AM SOC INFORM SCI, V24, P265\n"
           "TC 42\nPY 2019\nDI 10.1007/s11192-019-03000-1\nER\n\n"
           "PT J\nAU Lee, K\nTI Second paper\nSO SCIENTOMETRICS\nTC 3\nPY 2021\nC1 [Lee, K] Seoul Natl Univ, Seoul, South Korea.\nER\nEF\n")
    w, fmt, _ = read(wos.encode("utf-8"))
    assert fmt == "wos_plain" and len(w) == 2, (fmt, w)
    assert w["title"][0] == "A study of bibliometric maps" and w["authors"][0] == ["Smith, J", "Lee, K"]
    assert w["countries"][0] == ["South Korea", "USA"], w["countries"][0]
    assert w["citations"].tolist() == [42, 3] and len(w["references"][0]) == 2 and w["keywords"][0][0] == "citation analysis"

    tab = ("PT\tAU\tTI\tSO\tDE\tID\tC1\tCR\tTC\tPY\tDI\n"
           "J\tSmith, J; Lee, K\tTab paper\tSCIENTOMETRICS\tco-word; mapping\tNETWORKS\t"
           "[Smith, J] Harvard Univ, Cambridge, MA 02138 USA; [Lee, K] Seoul Natl Univ, Seoul, South Korea\t"
           "Garfield E, 1955, SCIENCE, V122, P108; Small H, 1973, J AM SOC INFORM SCI, V24, P265\t7\t2020\t10.1/x\n")
    t, fmt, _ = read(tab.encode("utf-16"))
    assert fmt == "wos_tab" and t["authors"][0] == ["Smith, J", "Lee, K"] and t["countries"][0] == ["South Korea", "USA"]
    assert t["citations"][0] == 7 and len(t["references"][0]) == 2, t.iloc[0].to_dict()

    pubmed = ("PMID- 123\nTI  - A trial of something\n      continued title\nDP  - 2018 Mar\n"
              "AU  - Doe J\nAU  - Roe R\nAD  - Dept of Medicine, Oxford University, Oxford, UK.\n"
              "JT  - The Lancet\nMH  - *Diabetes Mellitus/therapy\nOT  - insulin\nLID - 10.1016/x [doi]\n\n"
              "PMID- 124\nTI  - Another\nDP  - 2020\nAU  - Doe J\nJT  - BMJ\n")
    p, fmt, notes = read(pubmed.encode())
    assert fmt == "pubmed" and p["title"][0] == "A trial of something continued title" and p["year"][0] == 2018
    assert p["index_keywords"][0] == ["diabetes mellitus"] and p["countries"][0] == ["UK"] and notes

    oa = ("title,publication_year,cited_by_count,authorships.author.display_name,authorships.countries,doi\n"
          "X,2022,5,A B|C D,KR|CN,https://doi.org/10.1/abc\n")
    o, fmt, notes = read(oa.encode())
    assert fmt == "csv" and o["authors"][0] == ["A B", "C D"] and o["doi"][0] == "10.1/abc" and "not found" in notes[0]
    assert o["countries"][0] == ["CN", "KR"], o["countries"][0]           # "|" lists split like the others
    try:
        read(b"a,b\n1,2\n")
        raise AssertionError("a non-bibliographic file must be refused")
    except ValueError as e:
        assert "publication export" in str(e)
    assert completeness(df)["countries"] == 100.0 and completeness(df)["references"] == 50.0
    assert country("Dept X, Univ Y, Berlin 10115, Germany") == "Germany" and country("Lab, 12345") is None

    # Crossref REST API (/works): list envelope and a single work
    cr_work = {"DOI": "10.1016/J.JOI.2017.08.007", "type": "journal-article", "title": ["bibliometrix: An R-tool"],
               "container-title": ["Journal of Informetrics"], "issued": {"date-parts": [[2017, 11]]},
               "is-referenced-by-count": 1500, "subject": ["Library and Information Sciences"],
               "author": [{"given": "Massimo", "family": "Aria",
                           "affiliation": [{"name": "University of Naples Federico II, Naples, Italy"}]},
                          {"given": "Corrado", "family": "Cuccurullo", "affiliation": []}],
               "reference": [{"key": "r1", "DOI": "10.1002/ASI.21525"},
                             {"key": "r2", "unstructured": "Garfield E. Citation indexes. Science 1955"},
                             {"key": "r3", "author": "Small", "year": "1973", "journal-title": "JASIS"}]}
    env = {"status": "ok", "message-type": "work-list", "message": {"items": [cr_work, {**cr_work, "DOI": "10.1/b",
                                                                                     "reference": []}]}}
    c, fmt, notes = read(json.dumps(env).encode())
    assert fmt == "crossref" and len(c) == 2 and notes, (fmt, notes)
    assert c["authors"][0] == ["Aria, Massimo", "Cuccurullo, Corrado"] and c["year"][0] == 2017
    assert c["citations"][0] == 1500 and c["countries"][0] == ["Italy"] and c["source"][0] == "Journal of Informetrics"
    assert c["references"][0] == ["DOI 10.1002/asi.21525", "Garfield E. Citation indexes. Science 1955",
                                  "Small, 1973, JASIS"] and c["doi"][0] == "10.1016/j.joi.2017.08.007"
    assert read(json.dumps({"message-type": "work", "message": cr_work}).encode())[1] == "crossref"
    # OpenAlex API (/works): list response; countries from affiliation text, else OpenAlex's codes
    oa_work = {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/x", "title": "Mapping science",
               "publication_year": 2021, "type": "article", "cited_by_count": 12,
               "primary_location": {"source": {"display_name": "Scientometrics"}},
               "authorships": [{"author": {"display_name": "Ann Lee"}, "countries": ["KR"],
                                "raw_affiliation_strings": ["Seoul National University, Seoul, South Korea"]}],
               "keywords": [{"display_name": "Co-word analysis"}], "topics": [{"display_name": "Citation analysis"}],
               "referenced_works": ["https://openalex.org/W9", "https://openalex.org/W8"]}
    bare = {**oa_work, "authorships": [{"author": {"display_name": "Bo Li"}, "countries": ["CN"],
                                        "institutions": [{"display_name": "Tsinghua University"}]}]}
    o, fmt, _ = read(json.dumps({"meta": {"count": 2}, "results": [oa_work, bare]}).encode())
    assert fmt == "openalex" and o["authors"].tolist() == [["Ann Lee"], ["Bo Li"]] and o["source"][0] == "Scientometrics"
    assert o["countries"].tolist() == [["KR"], ["CN"]] and o["keywords"][0] == ["co-word analysis"]   # one kind
    assert o["references"][0] == ["https://openalex.org/W9", "https://openalex.org/W8"] and o["doi"][0] == "10.1/x"
    assert read(json.dumps([oa_work]).encode())[1] == "openalex"
    for bad in (b'{"a": 1}', b"[1, 2", b"[]"):
        try:
            read(bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    # Dimensions CSV: a note line above the header, a country column of its own
    dim = ('"About the data: Exported on Oct 31, 2026. Criteria: keyword, in title and abstract."\n'
           "Rank,Publication ID,DOI,Title,Source title/Anthology title,PubYear,Authors,"
           "Research Organizations - standardized,Country of Research organization,Times cited,Publication Type\n"
           '1,pub.1,10.1/d,Dimensions paper,Scientometrics,2020,"Lee, Ann; Li, Bo",'
           '"Seoul National University; Tsinghua University","South Korea; China",4,article\n')
    d, fmt, notes = read(dim.encode())
    assert fmt == "csv" and len(d) == 1 and d["title"][0] == "Dimensions paper" and d["year"][0] == 2020, d
    assert d["countries"][0] == ["China", "South Korea"] and d["citations"][0] == 4 and d["source"][0] == "Scientometrics"
    assert _skip_preamble("a,b,c\n1,2,3\n4,5,6\n") == "a,b,c\n1,2,3\n4,5,6\n"      # a normal header is kept
    print("bibliographic parser self-check passed")
