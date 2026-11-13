# LLM Corporate Credit Risk — Can an LLM Read a 10-K and Rate Credit Risk?

A case-control study on SEC EDGAR data: for U.S. firms that filed for
Chapter 11, take the **last 10-K filed before the bankruptcy** and ask
four increasingly modern models to assess 12-month default risk — then
compare them honestly on the same firms, and test whether text adds
signal beyond financial ratios.

## Why this project
Traditional corporate credit models use structured ratios. Most of what
a credit analyst actually reads is unstructured text (MD&A, Risk
Factors). This project measures — on verified bankruptcy events, with
strict point-in-time discipline — how far text-based methods (lexicon,
embeddings, and an LLM acting as an analyst) get against the ratio
benchmark, and what governance an LLM rater needs before a bank could
use one.

## Design
- **Cases**: firms with a verified Item 1.03 (Bankruptcy) 8-K on EDGAR;
  sample document = last 10-K filed *before* the event (≤18 months).
- **Controls**: listed firms with no bankruptcy 8-K, matched on SIC
  industry and 10-K year (2 per case).
- **Legs**: (1) Altman-style ratio logistic; (2) Loughran-McDonald-style
  negative-word lexicon; (3) MD&A sentence-embeddings + logistic;
  (4) LLM zero-shot rating with structured JSON output and
  verbatim-evidence requirements; (5) fusion of ratios + text.
- **Evaluation**: leave-one-out CV (small-n honest), AUC / KS.
- **LLM governance**: versioned prompts saved per firm, evidence quotes
  programmatically verified against the source text, per-call token and
  cost logging, no-key dry-run mode that fabricates nothing.

## Pipeline
| Script | What it does |
|---|---|
| `src/00_build_corpus.py` | Verify events & download/extract 10-K text from EDGAR |
| `src/01_ratio_model.py` | XBRL company-facts ratios → logistic benchmark |
| `src/02_text_model.py` | MD&A embeddings + lexicon text models |
| `src/03_llm_rating.py` | LLM credit rating (needs OPENAI_API_KEY or ANTHROPIC_API_KEY; dry-run otherwise) |
| `src/04_evaluate.py` | Head-to-head comparison + fusion test → `outputs/comparison.md` |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/00_build_corpus.py    # EDGAR is free; be patient, it's rate-limited
python src/01_ratio_model.py
python src/02_text_model.py
python src/03_llm_rating.py      # dry-run without a key; real ratings with one
python src/04_evaluate.py
```

## Data & ethics
All data are public SEC filings. Bankruptcy labels come from the firms'
own 8-K disclosures. Point-in-time rule enforced throughout: nothing
dated after a sample's 10-K filing is used as an input.

## Companion project
Retail-credit sibling: [credit-risk-model-development-validation](https://github.com/SiyuanLyu/credit-risk-model-development-validation)
(PD/LGD/EAD, SR 11-7 validation, CECL/CCAR stress, ML challengers).
