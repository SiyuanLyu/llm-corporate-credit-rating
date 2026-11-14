"""
Step 5a - RAG comparison arm: retrieval-selected excerpts for LLM rating
=========================================================================
Step 3 feeds the LLM a fixed-position excerpt of each 10-K (head + middle
of MD&A and Risk Factors). That wastes most of the token budget on
boilerplate. This arm instead RETRIEVES the passages most relevant to
credit risk and builds the rating prompt from those.

Design points (the interview-relevant part):
- Chunking: each section is split into overlapping passages
  (~1200 chars, 200 overlap), each carrying provenance metadata
  (cik, filing_date, section, char offsets) so every selected passage is
  citable back to the source text.
- Retrieval: TF-IDF (word 1-2 grams) cosine similarity against a fixed,
  versioned set of credit-risk queries (liquidity, debt maturity,
  covenants, going concern, cash-flow deterioration, restructuring).
  TF-IDF is deliberate: transparent, reproducible, no extra model -
  an embedding variant can be swapped in behind the same interface.
- Point-in-time discipline: retrieval for a firm runs ONLY over that
  firm's own filing, and the corpus index records filing_date for every
  passage, so nothing dated after the prediction point can be selected.
- Budget parity: the total selected characters are capped at the same
  budget Step 3 spends (MD&A 10k + Risk 6k), so the comparison isolates
  the effect of SELECTION, not of giving one arm more text.
- NO API key is needed for this step and none is used: it writes the
  retrieval-arm prompts (outputs/llm_prompts_rag/) in the same format
  and prompt version family as Step 3, ready to be scored by the same
  runner once a key is provided. It produces NO ratings by itself.

Outputs:
  outputs/rag_passages/<cik>_<fy>.json   selected passages + provenance
  outputs/llm_prompts_rag/<cik>_<fy>.txt rating prompts (retrieval arm)
  outputs/rag_summary.md                 selection statistics vs Step 3
"""
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TEXT = ROOT / "data" / "text"
OUT = ROOT / "outputs"
PROMPT_VERSION = "v1.0-rag"

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
TOP_K_PER_QUERY = 2
MDA_BUDGET = 10_000   # Step 3 spends head 6000 + mid 4000 on MD&A
RISK_BUDGET = 6_000   # Step 3 spends head 4000 + mid 2000 on Risk Factors

RISK_QUERIES = {
    "liquidity": "liquidity cash and cash equivalents working capital access to capital markets fund operations",
    "debt_maturity": "debt maturing within twelve months principal payments due refinancing long-term debt obligations",
    "covenants": "covenant compliance violation waiver default under credit agreement indebtedness restrictions",
    "going_concern": "going concern substantial doubt ability to continue bankruptcy chapter 11 restructuring",
    "cashflow_deterioration": "net loss declining revenues decreased operating cash flow negative cash flows from operations",
    "asset_impairment": "impairment write-down goodwill charges losses on disposal of assets",
}

PROMPT = """You are a corporate credit analyst. Below are excerpts from the
Management's Discussion & Analysis and Risk Factors sections of a U.S.
company's annual report (Form 10-K). Assess the company's credit
worthiness over the NEXT 12 months from the filing date.

Rules:
- Use ONLY the information in the excerpts. Do not use outside knowledge
  about this company or what happened to it later.
- Quote evidence verbatim from the excerpts.
- Return ONLY a JSON object with this schema:
{{"rating": "<one of AAA, AA, A, BBB, BB, B, CCC, CC, C, D>",
  "pd_1y": <your estimated 1-year probability of default, 0 to 1>,
  "outlook": "<positive|stable|negative>",
  "evidence": ["<verbatim quote>", "<verbatim quote>"],
  "key_risks": ["<short phrase>", "<short phrase>"]}}

=== MD&A EXCERPTS (retrieval-selected) ===
{mda}

=== RISK FACTORS EXCERPTS (retrieval-selected) ===
{risk}
"""


def chunk(text: str):
    """Split text into overlapping passages with char offsets."""
    out, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        out.append((start, end, text[start:end]))
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return out


def select_passages(text: str, budget: int):
    """TF-IDF retrieval of the most credit-risk-relevant passages."""
    chunks = chunk(text)
    if not chunks:
        return []
    docs = [c[2] for c in chunks]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform(docs)
    scored = {}  # chunk idx -> (best score, best query)
    for qname, qtext in RISK_QUERIES.items():
        qv = vec.transform([qtext])
        sims = cosine_similarity(qv, mat).ravel()
        for idx in sims.argsort()[::-1][:TOP_K_PER_QUERY]:
            if sims[idx] <= 0:
                continue
            if idx not in scored or sims[idx] > scored[idx][0]:
                scored[idx] = (float(sims[idx]), qname)
    ranked = sorted(scored.items(), key=lambda kv: -kv[1][0])
    picked, used = [], 0
    for idx, (score, qname) in ranked:
        start, end, passage = chunks[idx]
        if used + len(passage) > budget and picked:
            continue
        picked.append({"char_start": start, "char_end": end,
                       "score": round(score, 4), "query": qname,
                       "text": passage})
        used += len(passage)
        if used >= budget:
            break
    picked.sort(key=lambda p: p["char_start"])  # restore document order
    return picked


def baseline_regions(text: str, head: int, mid: int):
    """Char ranges Step 3's fixed excerpt would cover (for overlap stats)."""
    if len(text) <= head + mid:
        return [(0, len(text))]
    m0 = len(text) // 2
    return [(0, head), (m0, m0 + mid)]


def overlap_frac(picked, regions, text_len):
    if not picked or text_len == 0:
        return 0.0
    covered = 0
    for p in picked:
        for r0, r1 in regions:
            covered += max(0, min(p["char_end"], r1) - max(p["char_start"], r0))
    return covered / sum(p["char_end"] - p["char_start"] for p in picked)


def main():
    index = pd.read_csv(PROC / "corpus_index.csv", dtype={"cik": str})
    (OUT / "rag_passages").mkdir(parents=True, exist_ok=True)
    (OUT / "llm_prompts_rag").mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in index.iterrows():
        cik, fy = row["cik"], int(row["filing_year"])
        record = {"cik": cik, "name": row["name"], "filing_date": row["filing_date"],
                  "label": int(row["label"]), "sections": {}}
        prompt_parts = {}
        for section, budget, h, m in (("mda", MDA_BUDGET, 6000, 4000),
                                      ("risk", RISK_BUDGET, 4000, 2000)):
            path = TEXT / f"{cik}_{fy}_{section}.txt"
            text = path.read_text() if path.exists() else ""
            picked = select_passages(text, budget) if text else []
            record["sections"][section] = {
                "source_chars": len(text), "n_passages": len(picked),
                "selected_chars": sum(len(p["text"]) for p in picked),
                "overlap_with_step3_excerpt": round(
                    overlap_frac(picked, baseline_regions(text, h, m), len(text)), 3),
                "passages": picked,
            }
            prompt_parts[section] = "\n[...]\n".join(p["text"] for p in picked)
        (OUT / "rag_passages" / f"{cik}_{fy}.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False))
        prompt = PROMPT.format(mda=prompt_parts["mda"] or "(no passages retrieved)",
                               risk=prompt_parts["risk"] or "(no passages retrieved)")
        header = (f"# prompt {PROMPT_VERSION} | arm: retrieval-selected excerpts "
                  f"(Step 5a) | cik {cik} | filing {row['filing_date']}\n\n")
        (OUT / "llm_prompts_rag" / f"{cik}_{fy}.txt").write_text(header + prompt)
        rows.append({"cik": cik, "name": row["name"][:40], "label": int(row["label"]),
                     "mda_passages": record["sections"]["mda"]["n_passages"],
                     "risk_passages": record["sections"]["risk"]["n_passages"],
                     "mda_overlap": record["sections"]["mda"]["overlap_with_step3_excerpt"],
                     "risk_overlap": record["sections"]["risk"]["overlap_with_step3_excerpt"]})
    stats = pd.DataFrame(rows)
    lines = ["# Step 5a - Retrieval-selected excerpts (RAG comparison arm)", "",
             f"Firms processed: {len(stats)}  |  Prompt version: {PROMPT_VERSION}", "",
             "Overlap = share of selected characters that Step 3's fixed head+middle "
             "excerpt would also have shown the LLM. Low overlap means retrieval is "
             "surfacing material the baseline arm never sees.", "",
             f"- Mean MD&A passages per firm: {stats.mda_passages.mean():.1f}",
             f"- Mean Risk passages per firm: {stats.risk_passages.mean():.1f}",
             f"- Mean MD&A overlap with Step 3 excerpt: {stats.mda_overlap.mean():.2f}",
             f"- Mean Risk overlap with Step 3 excerpt: {stats.risk_overlap.mean():.2f}",
             "", "Ratings for this arm are produced by the Step 3 runner once an API "
             "key is provided (same schema, same verification). This step generates "
             "NO ratings on its own.", "", stats.to_string(index=False)]
    (OUT / "rag_summary.md").write_text("\n".join(lines))
    print(f"done: {len(stats)} firms | mean MD&A overlap "
          f"{stats.mda_overlap.mean():.2f} | mean Risk overlap {stats.risk_overlap.mean():.2f}")


if __name__ == "__main__":
    main()
