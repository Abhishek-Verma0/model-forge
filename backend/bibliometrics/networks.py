"""Bibliometric networks and thematic analysis (plan §14) -- numpy only (NetworkX is
not installed), each algorithm written from its published definition:

  layout       Fruchterman & Reingold (1991) force-directed placement, with the force
               formula and defaults of NetworkX's spring_layout (k = sqrt(1/n),
               50 iterations, temperature 0.1 of the frame, linear cooling); seeded.
  communities  Newman (2006) leading-eigenvector method: split a group by the sign of
               the leading eigenvector of its modularity matrix while modularity rises.
  modularity   Q = 1/(2m) * sum_ij (A_ij - k_i k_j / 2m) * [same community]

Networks: co-authorship (authors), country collaboration, keyword co-occurrence,
co-citation (references cited together), bibliographic coupling (documents sharing
references). Thematic map (Cobo et al. 2011, after Callon 1991): communities of the
keyword network on the equivalence index e_ij = c_ij² / (c_i c_j); centrality =
10 x external links, density = 100 x internal links / words; quadrants split at the
medians. Thematic evolution: one map per period, consecutive themes joined by the
inclusion index |A ∩ B| / min(|A|, |B|). Three-field plot: top items of three fields
and how often they occur in the same documents.
"""

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from bibliometrics.analysis import keyword_list

TOP_N = 50              # our starting point: nodes in a network
LAYOUT_ITERATIONS = 50  # published default: NetworkX spring_layout iterations
THEME_WORDS = 250       # our starting point (bibliometrix thematicMap uses 250 words)
THEME_MIN_FREQ = 5      # our starting point: a word needs this many documents to be mapped
THEME_LABEL = 3         # words naming a theme
THREE_FIELD_N = 10      # our starting point: items per column


def modularity(W, labels):
    k = W.sum(axis=1)
    m2 = k.sum()
    if m2 == 0:
        return 0.0
    same = labels[:, None] == labels[None, :]
    return float(((W - np.outer(k, k) / m2) * same).sum() / m2)


def communities(W):
    """Newman (2006) leading-eigenvector division; returns a label per node (0 = largest)."""
    n = len(W)
    k = W.sum(axis=1)
    m2 = k.sum()
    if n == 0 or m2 == 0:
        return np.zeros(n, dtype=int)
    B = W - np.outer(k, k) / m2
    todo, done = [np.arange(n)], []
    while todo:
        g = todo.pop()
        if len(g) < 2:
            done.append(g)
            continue
        Bg = B[np.ix_(g, g)].copy()
        Bg[np.diag_indices(len(g))] -= Bg.sum(axis=1)       # generalised modularity matrix of the subgroup
        vals, vecs = np.linalg.eigh(Bg)
        s = np.where(vecs[:, -1] >= 0, 1.0, -1.0)
        gain = s @ Bg @ s / (2 * m2)                        # delta Q = s'B(g)s / 4m, and 2*m2 = 4m
        if vals[-1] <= 1e-10 or gain <= 1e-10 or s.min() == s.max():
            done.append(g)
        else:
            todo += [g[s > 0], g[s < 0]]
    labels = np.zeros(n, dtype=int)
    for i, g in enumerate(sorted(done, key=lambda g: (-len(g), g.min()))):
        labels[g] = i
    return labels


def layout(W, seed=0):
    """Fruchterman-Reingold positions in [0, 1]² (NetworkX spring_layout formula)."""
    n = len(W)
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.array([[0.5, 0.5]])
    A = W / W.max() if W.max() > 0 else W
    pos = np.random.default_rng(seed).random((n, 2))
    k = np.sqrt(1.0 / n)
    t = 0.1 * max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]))
    dt = t / (LAYOUT_ITERATIONS + 1)
    for _ in range(LAYOUT_ITERATIONS):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 0.01)
        disp = np.einsum("ijk,ij->ik", delta, k * k / dist ** 2 - A * dist / k)
        length = np.maximum(np.linalg.norm(disp, axis=-1), 0.01)
        pos += disp * (t / length)[:, None]
        t -= dt
    pos -= pos.min(axis=0)
    span = pos.max(axis=0)
    return pos / np.where(span > 0, span, 1)


def cooccurrence(lists, top_n, min_freq=1):
    """Top items by document frequency and their co-occurrence counts."""
    freq = Counter(x for items in lists for x in set(items))
    items = [w for w, c in freq.most_common() if c >= min_freq][:top_n]
    idx = {w: i for i, w in enumerate(items)}
    C = np.zeros((len(items), len(items)))
    for doc in lists:
        present = sorted({idx[x] for x in doc if x in idx})
        for a, b in combinations(present, 2):
            C[a, b] += 1
            C[b, a] += 1
    return items, freq, C


def _graph(labels, sizes, W, seed, kind, extra=None):
    comm = communities(W)
    pos = layout(W, seed)
    nodes = [{"id": i, "label": str(lab), "size": int(sizes[i]), "community": int(comm[i]),
              "x": round(float(pos[i, 0]), 4), "y": round(float(pos[i, 1]), 4)} for i, lab in enumerate(labels)]
    edges = [{"source": int(a), "target": int(b), "weight": float(W[a, b])}
             for a, b in zip(*np.triu_indices(len(labels), 1)) if W[a, b] > 0]
    return {"kind": kind, "nodes": nodes, "edges": edges, "communities": int(comm.max() + 1) if len(comm) else 0,
            "modularity": round(modularity(W, comm), 4), **(extra or {})}


def _ref_key(ref):
    return " ".join(str(ref).lower().split()).rstrip(".;, ")


KINDS = ("coauthor", "country", "keyword", "cocitation", "coupling")


def network(df, kind, top_n=TOP_N, min_weight=1, field="keywords", seed=0):
    if kind not in KINDS:
        raise ValueError(f"Network kind must be one of {', '.join(KINDS)}.")
    if kind == "coupling":   # documents linked by the references they share
        refs = [{_ref_key(r) for r in rs} for rs in df["references"]]
        order = df["citations"].fillna(0).astype(float).to_numpy().argsort()[::-1]
        keep = [i for i in order if refs[i]][:top_n]
        if not keep:
            raise ValueError("No reference lists in this file - bibliographic coupling needs them.")
        W = np.array([[len(refs[a] & refs[b]) if a != b else 0 for b in keep] for a in keep], float)
        W[W < min_weight] = 0
        labels = [f"{(df['authors'][i][0] if df['authors'][i] else '?')} {df['year'][i] if not pd.isna(df['year'][i]) else ''}"
                  f": {str(df['title'][i])[:60]}" for i in keep]
        sizes = [0 if pd.isna(df["citations"][i]) else int(df["citations"][i]) for i in keep]
        return _graph(labels, sizes, W, seed, kind, {"size_means": "citations"})
    lists = {"coauthor": df["authors"], "country": df["countries"],
             "keyword": keyword_list(df, field),
             "cocitation": df["references"].map(lambda rs: [_ref_key(r) for r in rs])}[kind]
    items, freq, C = cooccurrence(list(lists), top_n)
    if not items:
        raise ValueError(f"No {kind} data in this file.")
    C[C < min_weight] = 0
    labels = [w[:80] for w in items] if kind == "cocitation" else items
    return _graph(labels, [freq[w] for w in items], C, seed, kind,
                  {"size_means": "times cited" if kind == "cocitation" else "documents"})


def thematic_map(df, field="keywords", words=THEME_WORDS, min_freq=THEME_MIN_FREQ, label_words=THEME_LABEL):
    items, freq, C = cooccurrence(list(keyword_list(df, field)), words, min_freq)
    if len(items) < 3:
        raise ValueError(f"Fewer than 3 terms appear in {min_freq}+ documents - lower the minimum frequency.")
    f = np.array([freq[w] for w in items], float)
    E = C ** 2 / np.outer(f, f)
    np.fill_diagonal(E, 0)
    comm = communities(E)
    themes = []
    for c in sorted(set(comm)):
        members = np.where(comm == c)[0]
        inside = np.zeros(len(items), bool)
        inside[members] = True
        terms = sorted(members, key=lambda i: -f[i])
        themes.append({"id": int(c), "label": " / ".join(items[i] for i in terms[:label_words]),
                       "words": [items[i] for i in terms], "frequency": int(f[members].sum()),
                       "centrality": float(10 * E[np.ix_(inside, ~inside)].sum()),
                       "density": float(100 * E[np.ix_(inside, inside)].sum() / 2 / len(members))})
    cm, dm = np.median([t["centrality"] for t in themes]), np.median([t["density"] for t in themes])
    for t in themes:
        hi_c, hi_d = t["centrality"] >= cm, t["density"] >= dm
        t["quadrant"] = ("motor themes" if hi_c and hi_d else "niche themes" if hi_d else
                         "basic themes" if hi_c else "emerging or declining themes")
    return {"field": field, "min_freq": min_freq, "themes": themes, "median_centrality": float(cm),
            "median_density": float(dm),
            "note": "Centrality = how linked a theme is to the others; density = how tightly its own terms link."}


def evolution(df, field="keywords", cuts=None, min_freq=2, words=THEME_WORDS):
    years = df["year"].dropna().astype(int)
    if years.empty:
        raise ValueError("No publication years in this file.")
    lo, hi = int(years.min()), int(years.max())
    cuts = sorted({int(c) for c in (cuts or [(lo + hi) // 2]) if lo <= int(c) < hi})
    if not cuts:
        raise ValueError(f"Cut years must lie between {lo} and {hi - 1}.")
    bounds = [lo] + [c + 1 for c in cuts] + [hi + 1]
    periods = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        part = df[(df["year"] >= a) & (df["year"] < b)].reset_index(drop=True)
        try:
            themes = thematic_map(part, field, words, min_freq)["themes"] if len(part) else []
        except ValueError:
            themes = []
        periods.append({"from": a, "to": b - 1, "documents": int(len(part)), "themes": themes})
    links = []
    for p, (pa, pb) in enumerate(zip(periods[:-1], periods[1:])):
        for ta in pa["themes"]:
            for tb in pb["themes"]:
                shared = set(ta["words"]) & set(tb["words"])
                if shared:
                    links.append({"from_period": p, "from": ta["id"], "to_period": p + 1, "to": tb["id"],
                                  "inclusion": round(len(shared) / min(len(ta["words"]), len(tb["words"])), 4),
                                  "shared": sorted(shared)[:10]})
    return {"field": field, "cuts": cuts, "periods": periods, "links": links}


def three_field(df, fields=("authors", "keywords", "source"), n=THREE_FIELD_N):
    def items(field):
        if field == "source":
            return df["source"].map(lambda s: [s] if isinstance(s, str) and s else [])
        return keyword_list(df, field) if field in ("keywords", "index_keywords", "title") else df[field]
    cols = [list(items(f)) for f in fields]
    tops = [[w for w, _ in Counter(x for doc in c for x in set(doc)).most_common(n)] for c in cols]
    links = []
    for side in (0, 1):
        left, right = set(tops[side]), set(tops[side + 1])
        pairs = Counter()
        for a_doc, b_doc in zip(cols[side], cols[side + 1]):
            for a in set(a_doc) & left:
                for b in set(b_doc) & right:
                    pairs[(a, b)] += 1
        links += [{"from_field": side, "from": a, "to_field": side + 1, "to": b, "weight": w}
                  for (a, b), w in pairs.items()]
    return {"fields": list(fields), "columns": tops, "links": links}


if __name__ == "__main__":
    # two 4-cliques joined by one edge: two communities, modularity matches the definition
    W = np.zeros((8, 8))
    for block in (range(4), range(4, 8)):
        for a, b in combinations(block, 2):
            W[a, b] = W[b, a] = 1
    W[3, 4] = W[4, 3] = 1
    lab = communities(W)
    assert len(set(lab[:4])) == 1 and len(set(lab[4:])) == 1 and lab[0] != lab[7], lab
    m = W.sum() / 2                                          # 13 edges
    by_def = sum((W[i, j] - W[i].sum() * W[j].sum() / (2 * m)) for i in range(8) for j in range(8)
                 if lab[i] == lab[j]) / (2 * m)
    assert abs(modularity(W, lab) - by_def) < 1e-12 and modularity(W, lab) > 0.4
    assert len(set(communities(np.ones((5, 5)) - np.eye(5)))) == 1      # a clique doesn't split
    pos = layout(W)
    assert pos.shape == (8, 2) and pos.min() >= 0 and pos.max() <= 1
    # connected nodes end up closer than the two far ends of the bridge
    near = np.linalg.norm(pos[0] - pos[1])
    far = np.linalg.norm(pos[0] - pos[7])
    assert near < far, (near, far)
    assert np.allclose(layout(W, 3), layout(W, 3))                      # seeded: same picture every time

    df = pd.DataFrame({
        "title": [f"t{i}" for i in range(8)], "year": pd.array([2010, 2011, 2012, 2013, 2018, 2019, 2020, 2021], "Int64"),
        "source": ["A", "A", "B", "B", "A", "C", "C", "C"],
        "authors": [["x", "y"], ["x", "y"], ["x"], ["z"], ["z", "w"], ["w"], ["z", "w"], ["y"]],
        "countries": [["Italy", "USA"], ["Italy"], [], ["Spain"], ["Spain", "Italy"], [], [], []],
        "keywords": [["ml", "nets"], ["ml", "nets"], ["ml", "data"], ["data", "stats"], ["llm", "agents"],
                     ["llm", "agents"], ["llm", "agents", "ml"], ["agents"]],
        "citations": pd.array([50, 20, 5, 1, 9, 3, 2, 0], "Int64"),
        "references": [["R1", "R2"], ["R1", "R2", "R3"], ["R2"], [], ["R4"], ["R4", "R5"], ["R4"], []]})
    co = network(df, "coauthor")
    edge = {(co["nodes"][e["source"]]["label"], co["nodes"][e["target"]]["label"]): e["weight"] for e in co["edges"]}
    assert edge.get(("x", "y"), edge.get(("y", "x"))) == 2 and co["nodes"][0]["size"] >= 3
    cc = network(df, "cocitation")
    assert {"r1", "r2"} <= {n["label"] for n in cc["nodes"]}
    cp = network(df, "coupling")
    assert cp["nodes"][0]["size"] == 50 and any(e["weight"] == 2 for e in cp["edges"])   # docs 0 and 1 share R1, R2
    assert network(df, "country")["nodes"][0]["label"] == "Italy"
    tm = thematic_map(df, min_freq=2)
    labels = {t["label"] for t in tm["themes"]}
    assert any("llm" in l or "agents" in l for l in labels) and all(t["quadrant"] for t in tm["themes"]), tm["themes"]
    ev = evolution(df, cuts=[2015], min_freq=2)
    assert [p["documents"] for p in ev["periods"]] == [4, 4] and ev["periods"][0]["themes"]
    assert ev["periods"][1]["themes"] == []          # only 2 terms in 2+ documents there: too few for a map
    tf = three_field(df, n=3)
    assert len(tf["columns"]) == 3 and tf["links"] and all(l["weight"] >= 1 for l in tf["links"])
    print("bibliometric networks self-check passed")
