"""
Step 2 - Benchmark #2: text-embedding model (the "modern ML" leg)
=================================================================
Embeds each firm's MD&A text with a sentence-transformer
(all-MiniLM-L6-v2 by default; swap in ProsusAI/finbert-based encoders
the same way), mean-pools chunk embeddings into one firm vector, and
fits a logistic regression. Also computes a simple lexicon distress
score (share of negative-financial-dictionary words, Loughran-McDonald
style word list subset) as an interpretable text feature.

Evaluation mirrors Step 1: LOOCV AUC/KS.

Outputs: outputs/text_metrics.json, outputs/text_predictions.csv
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TEXT = ROOT / "data" / "text"
OUT = ROOT / "outputs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

NEG_WORDS = {
    "loss", "losses", "decline", "declined", "default", "bankruptcy",
    "covenant", "liquidity", "impairment", "restructuring", "doubt",
    "delist", "delisted", "violation", "waiver", "downgrade", "negative",
    "decrease", "decreased", "adverse", "litigation", "going concern",
    "substantial doubt", "forbearance", "missed", "unable",
}


def ks_statistic(y, p):
    order = np.argsort(p)[::-1]
    ys = np.asarray(y)[order]
    return float(np.max(np.abs(np.cumsum(ys) / ys.sum()
                               - np.cumsum(1 - ys) / (1 - ys).sum())))


def chunk(text: str, size=1200, max_chunks=40):
    return [text[i:i + size] for i in range(0, min(len(text), size * max_chunks), size)]


def main() -> None:
    from sentence_transformers import SentenceTransformer
    idx = pd.read_csv(PROC / "corpus_index.csv")
    model = SentenceTransformer(MODEL_NAME)
    vecs, lex, meta = [], [], []
    for _, r in idx.iterrows():
        text = (TEXT / f"{int(r['cik'])}_{int(r['filing_year'])}_mda.txt").read_text()
        chunks = chunk(text.lower())
        emb = model.encode(chunks, show_progress_bar=False)
        vecs.append(emb.mean(axis=0))
        words = re.findall(r"[a-z ]+", text.lower())
        joined = " ".join(words)
        total = max(len(joined.split()), 1)
        neg = sum(joined.count(w) for w in NEG_WORDS)
        lex.append(neg / total)
        meta.append({"cik": r["cik"], "filing_year": r["filing_year"],
                     "sample_id": f"{int(r['cik'])}_{int(r['filing_year'])}",
                     "label": r["label"],
                     "neg_word_share": neg / total, "n_chunks": len(chunks)})
        print(f"  embedded {r['name']}: {len(chunks)} chunks, neg {neg / total:.4f}")
    X = np.vstack(vecs)
    m = pd.DataFrame(meta)
    y = m["label"].to_numpy()

    def loocv_auc(Xin):
        preds = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(Xin):
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=1000, C=0.5))
            pipe.fit(Xin[tr], y[tr])
            preds[te] = pipe.predict_proba(Xin[te])[:, 1]
        return preds

    p_emb = loocv_auc(X)
    p_lex = loocv_auc(m[["neg_word_share"]].to_numpy(dtype=float))
    result = {
        "model": f"MD&A embeddings ({MODEL_NAME}) + logistic",
        "n": int(len(y)), "n_cases": int(y.sum()),
        "embedding_loocv": {"auc": float(roc_auc_score(y, p_emb)),
                            "ks": ks_statistic(y, p_emb)},
        "lexicon_loocv": {"auc": float(roc_auc_score(y, p_lex)),
                          "ks": ks_statistic(y, p_lex)},
    }
    m.assign(pred_text=p_emb, pred_lexicon=p_lex).to_csv(
        OUT / "text_predictions.csv", index=False)
    np.save(OUT / "embeddings.npy", X)
    (OUT / "text_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
