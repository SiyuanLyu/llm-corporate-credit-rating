# Head-to-head comparison (leave-one-out CV, same firms)

Firms: 34 (13 bankruptcy cases, 21 matched controls). LLM leg: not run (no API key yet).

| Model | AUC | KS |
|---|---|---|
| Ratio logistic (financials only) | 0.681 | 0.476 |
| Negative-word lexicon (text, simple) | 0.535 | 0.176 |
| MD&A embeddings (text, ML) | 0.659 | 0.330 |
| Fusion: ratios + embeddings | 0.582 | 0.476 |

Research question: does text beat / add to ratios on this sample?
Read the fusion row against the ratio row - and remember the
sample is small, so treat any gap < ~0.05 AUC as suggestive,
not conclusive. Expanding the corpus is Step 0's --scale TODO.
