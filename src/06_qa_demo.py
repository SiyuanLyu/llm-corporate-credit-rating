"""
Step 5b - Cited mini-QA demo over the 10-K corpus
===================================================
A prototype of the analyst-facing side of retrieval: ask a fixed set of
credit questions across all 34 filings and get back VERBATIM passages
with full citations (company, filing date, section, char offsets), so
every answer can be checked against the source in one click.

Point-in-time discipline: every question carries an AS_OF date and the
retriever only sees filings with filing_date <= AS_OF. Asking a question
"as of 2019" cannot surface a 2023 filing - the same anti-leakage rule
the rating arms follow.

This demo runs in RETRIEVAL mode: it returns the evidence passages an
LLM answer layer would be grounded on. If an API key is later provided,
an answer layer can summarise ONLY these passages (with the same
verbatim-quote verification as Step 3); until then the demo fabricates
nothing - the passages ARE the answer.

Outputs: outputs/qa_demo.md (human-readable) + outputs/qa_demo.json
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "rag", Path(__file__).resolve().parent / "05_rag_retrieval.py")
rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TEXT = ROOT / "data" / "text"
OUT = ROOT / "outputs"
TOP_K = 5
MIN_SCORE = 0.05

QUESTIONS = [
    {"id": "going_concern",
     "as_of": "2024-12-31",
     "question": "Which companies disclosed going-concern doubts, auditor concerns, or bankruptcy / restructuring proceedings?",
     "query": "going concern substantial doubt bankruptcy chapter 11 restructuring auditor expressed concern"},
    {"id": "covenant_breach",
     "as_of": "2024-12-31",
     "question": "Which companies reported covenant violations, waivers, or defaults under their credit agreements?",
     "query": "covenant violation waiver default under credit agreement not in compliance with financial covenants"},
    {"id": "maturity_wall",
     "as_of": "2024-12-31",
     "question": "Which companies faced significant debt maturities or refinancing needs within the next twelve months?",
     "query": "debt maturing within one year principal payments due in 12 months refinancing obligations current portion long-term debt"},
    {"id": "liquidity_stress_2019",
     "as_of": "2019-12-31",
     "question": "As of end-2019 only: which companies showed liquidity stress (cash burn, reliance on credit facilities)?",
     "query": "liquidity cash used in operations revolving credit facility borrowings fund operations cash requirements"},
]


def build_index(index: pd.DataFrame):
    """Chunk every filing once; each passage keeps its filing_date."""
    passages = []
    for _, row in index.iterrows():
        cik, fy = row["cik"], int(row["filing_year"])
        for section in ("mda", "risk"):
            path = TEXT / f"{cik}_{fy}_{section}.txt"
            if not path.exists():
                continue
            for start, end, text in rag.chunk(path.read_text()):
                passages.append({"cik": cik, "name": row["name"],
                                 "filing_date": row["filing_date"],
                                 "section": section, "char_start": start,
                                 "char_end": end, "text": text})
    return passages


def main():
    index = pd.read_csv(PROC / "corpus_index.csv", dtype={"cik": str})
    passages = build_index(index)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform([p["text"] for p in passages])
    results = []
    md = ["# Step 5b - Cited QA demo over the 10-K corpus (retrieval mode)", "",
          f"Corpus: {len(index)} filings, {len(passages)} passages indexed. "
          "Every hit is a verbatim passage with its citation; no LLM answer "
          "layer is run without an API key.", ""]
    for q in QUESTIONS:
        eligible = [i for i, p in enumerate(passages)
                    if p["filing_date"] <= q["as_of"]]
        qv = vec.transform([q["query"]])
        sims = cosine_similarity(qv, mat[eligible]).ravel()
        order = sims.argsort()[::-1][:TOP_K]
        hits = []
        md.append(f"## Q: {q['question']}")
        md.append(f"As of: {q['as_of']} | filings eligible: {len(set(passages[i]['cik'] for i in eligible))}")
        md.append("")
        for rank in order:
            if sims[rank] < MIN_SCORE:
                continue
            p = passages[eligible[rank]]
            quote = " ".join(p["text"].split())[:600]
            hits.append({**{k: p[k] for k in ("cik", "name", "filing_date",
                                               "section", "char_start", "char_end")},
                         "score": round(float(sims[rank]), 4), "quote": quote})
            md.append(f"- **{p['name']}** (10-K filed {p['filing_date']}, "
                      f"{p['section'].upper()}, chars {p['char_start']}-{p['char_end']}, "
                      f"score {sims[rank]:.3f})")
            md.append(f"  > {quote}")
        if not hits:
            md.append("- No passages above the relevance threshold.")
        md.append("")
        results.append({"question_id": q["id"], "question": q["question"],
                        "as_of": q["as_of"], "hits": hits})
    (OUT / "qa_demo.md").write_text("\n".join(md))
    (OUT / "qa_demo.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"done: {len(QUESTIONS)} questions over {len(passages)} passages")


if __name__ == "__main__":
    main()
