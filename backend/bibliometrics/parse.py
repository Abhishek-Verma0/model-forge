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

Canonical columns: title, year, source, authors (list), affiliations (list),
countries (list), keywords (author keywords), index_keywords (database keywords /
MeSH), citations (times cited in the exporting database), references (list), doi,
doc_type. Results depend on the completeness of these fields (plan §14), so the
reader reports the share of documents that have each one.

Countries come from the last part of each affiliation ("..., City, Country") --
the same rule bibliometrix uses; a short list of spellings is unified (USA, UK,
China). Affiliations without a country give none.
"""

import io
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
               "primary_location.source.display_name", "so"],
    "authors": ["authors", "author", "author names", "authorships.author.display_name", "au"],
    "affiliations": ["affiliations", "author affiliations", "authors affiliations", "institutions",
                     "authorships.institutions.display_name", "c1"],
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
    for f in ("authors", "affiliations", "keywords", "index_keywords", "references"):
        out[f] = [_split(v, "|" if isinstance(v, str) and "|" in v and ";" not in v else ";") for v in out[f]]
    return out, [f for f, c in m.items() if c is None]


def read(raw):
    """File bytes -> (DataFrame of canonical columns, format, notes)."""
    text = _decode(raw)
    fmt = detect(text)
    notes = []
    if fmt in ("wos_plain",):
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
        sep = "\t" if text.split("\n", 1)[0].count("\t") > text.split("\n", 1)[0].count(",") else ","
        table = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False, na_values=[""])
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
    df["countries"] = [sorted({c for c in (country(a) for a in affs) if c}) for affs in df["affiliations"]]
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

    oa = "title,publication_year,cited_by_count,authorships.author.display_name,doi\nX,2022,5,A B|C D,https://doi.org/10.1/abc\n"
    o, fmt, notes = read(oa.encode())
    assert fmt == "csv" and o["authors"][0] == ["A B", "C D"] and o["doi"][0] == "10.1/abc" and "not found" in notes[0]
    try:
        read(b"a,b\n1,2\n")
        raise AssertionError("a non-bibliographic file must be refused")
    except ValueError as e:
        assert "publication export" in str(e)
    assert completeness(df)["countries"] == 100.0 and completeness(df)["references"] == 50.0
    assert country("Dept X, Univ Y, Berlin 10115, Germany") == "Germany" and country("Lab, 12345") is None
    print("bibliographic parser self-check passed")
