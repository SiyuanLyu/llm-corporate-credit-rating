# Baseline results — 2026-11-13

Corpus: 34 firm-year samples from SEC EDGAR — 13 verified Chapter 11
cases (Item 1.03 8-K on file; sample document = last 10-K filed before
the event) and 21 controls matched on SIC industry and filing year,
deduplicated on (CIK, filing year).

| Model | LOOCV AUC | KS |
|---|---|---|
| Ratio logistic (Altman-style features, XBRL facts) | 0.681 | 0.476 |
| Negative-word lexicon (Loughran-McDonald style) | 0.535 | 0.176 |
| MD&A embeddings (MiniLM) + logistic | 0.659 | 0.330 |
| Fusion: ratios + embeddings | 0.582 | 0.476 |
| LLM zero-shot rating | not run yet — needs an API key; 34 versioned prompts are saved in outputs/llm_prompts/ | — |

## Honest reading of these numbers
- At n=34, NOTHING here is conclusive. Ratios and embeddings are in the
  same band (0.66–0.68); the lexicon is barely above chance; fusion is
  WORSE than ratios alone — textbook small-sample overfitting, and a
  useful reminder that "add more features" is not free.
- The sample is deliberately hard: controls come from the same
  industries and years, and a pre-bankruptcy 10-K is a late-stage
  document. Published studies with thousands of firms find text adds
  signal; whether it does here is exactly what the scale-up tests.
- Case quality is high (every event is the firm's own Item 1.03
  disclosure); control quality is mixed (a few matches are tiny or
  unrelated firms that share an SIC code — see corpus_index.csv).

## Known limitations / practice tasks
1. Add an API key and run Step 3; then compare LLM ratings against all
   legs, and audit its evidence quotes (verification is built in).
2. Scale the corpus: relax to 3 controls per case, add candidate CIKs,
   and re-run — watch whether the fusion penalty disappears with n.
3. Add the RAG arm (Step 5): retrieval-selected MD&A excerpts for the
   LLM vs the current head+middle excerpting; and a small cited-QA demo
   over the corpus. Keep the index point-in-time by filing date.
4. Winsorise interest coverage (extreme outliers, e.g. -9137x, come from
   near-zero interest expense denominators); add Altman Z'' itself as a
   single-feature benchmark.
5. Improve control matching: size (assets) proximity in addition to SIC.
