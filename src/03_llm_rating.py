"""
Step 3 - The GenAI leg: LLM reads the 10-K and issues a credit rating
====================================================================
For each corpus firm, the LLM receives excerpts of the MD&A and Risk
Factors from the LAST 10-K before the sample date and must return a
structured JSON rating:

  {"rating": "AAA..D letter grade", "pd_1y": 0-1 probability,
   "outlook": "positive|stable|negative",
   "evidence": ["verbatim quote supporting the assessment", ...],
   "key_risks": ["...", "..."]}

Governance built in (this is the part banks actually interview on):
- the prompt is versioned (PROMPT_VERSION) and saved per firm under
  outputs/llm_prompts/ before any call is made;
- evidence quotes are verified programmatically against the source text
  (hallucinated quotes are flagged, not silently accepted);
- token usage and estimated cost are logged per call;
- NO API key -> the module runs in DRY-RUN mode: it writes the prompts
  and a status file, and produces NO ratings. It never fabricates
  model output.

Providers: OpenAI (OPENAI_API_KEY) or Anthropic (ANTHROPIC_API_KEY),
small/cheap models by default - a few hundred filings cost tens of
dollars at most. Swap MODEL per provider as desired.

Outputs: outputs/llm_ratings.json (+ outputs/llm_prompts/*.txt)
"""
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TEXT = ROOT / "data" / "text"
OUT = ROOT / "outputs"
PROMPT_VERSION = "v1.0"
MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5"}
PRICE_PER_MTOK = {"openai": (0.15, 0.60), "anthropic": (1.00, 5.00)}  # in, out (USD, approx)

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

=== MD&A EXCERPT ===
{mda}

=== RISK FACTORS EXCERPT ===
{risk}
"""


def excerpt(text: str, head=6000, mid=4000) -> str:
    if len(text) <= head + mid:
        return text
    return text[:head] + "\n[...]\n" + text[len(text) // 2: len(text) // 2 + mid]


def call_llm(provider: str, prompt: str) -> tuple[dict, dict]:
    """Returns (parsed_json, usage_dict). Requires the provider's SDK."""
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=MODELS[provider],
            messages=[{"role": "user", "content": prompt}],
            temperature=0, response_format={"type": "json_object"})
        usage = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}
        return json.loads(resp.choices[0].message.content), usage
    from anthropic import Anthropic
    client = Anthropic()
    resp = client.messages.create(
        model=MODELS[provider], max_tokens=800, temperature=0,
        messages=[{"role": "user", "content": prompt}])
    usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    txt = resp.content[0].text
    return json.loads(re.search(r"\{.*\}", txt, re.S).group()), usage


def main() -> None:
    idx = pd.read_csv(PROC / "corpus_index.csv")
    provider = ("openai" if os.getenv("OPENAI_API_KEY")
                else "anthropic" if os.getenv("ANTHROPIC_API_KEY") else None)
    (OUT / "llm_prompts").mkdir(exist_ok=True)
    results = []
    for _, r in idx.iterrows():
        cik, fy = int(r["cik"]), int(r["filing_year"])
        mda = (TEXT / f"{cik}_{fy}_mda.txt").read_text()
        risk_path = TEXT / f"{cik}_{fy}_risk.txt"
        risk = risk_path.read_text() if risk_path.exists() else ""
        prompt = PROMPT.format(mda=excerpt(mda), risk=excerpt(risk, 4000, 2000))
        (OUT / "llm_prompts" / f"{cik}_{fy}.txt").write_text(
            f"# prompt {PROMPT_VERSION} | provider {provider} | model "
            f"{MODELS.get(provider, 'n/a')}\n\n" + prompt)
        entry = {"cik": cik, "name": r["name"], "label": int(r["label"]),
                 "prompt_version": PROMPT_VERSION}
        if provider is None:
            entry["status"] = "not_run_no_api_key"
        else:
            try:
                rating, usage = call_llm(provider, prompt)
                src = (mda + risk).lower()
                rating["evidence_verified"] = [
                    q.lower()[:60] in src for q in rating.get("evidence", [])]
                entry.update({"status": "ok", "model": MODELS[provider],
                              "rating": rating, "usage": usage,
                              "est_cost_usd": round(
                                  usage["in"] / 1e6 * PRICE_PER_MTOK[provider][0]
                                  + usage["out"] / 1e6 * PRICE_PER_MTOK[provider][1], 4)})
            except Exception as e:
                entry.update({"status": "error", "error": str(e)[:200]})
        results.append(entry)
        print(f"  {r['name']}: {entry['status']}")
    (OUT / "llm_ratings.json").write_text(json.dumps(results, indent=2))
    ran = sum(1 for e in results if e["status"] == "ok")
    print(f"LLM ratings completed: {ran}/{len(results)}"
          + ("" if provider else " (dry run - set OPENAI_API_KEY or ANTHROPIC_API_KEY)"))


if __name__ == "__main__":
    main()
