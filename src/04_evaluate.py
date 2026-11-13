"""
Step 4 - Head-to-head evaluation and the fusion question
========================================================
Merges the legs' out-of-sample (LOOCV) predictions and answers the
project's single research question: does TEXT add signal BEYOND the
financial ratios?

- comparison table (ratios / lexicon / embeddings / LLM if run / fusion)
- fusion: logistic regression on [ratio score, text score] - if the text
  coefficient is positive and fusion AUC > ratio AUC, text has
  incremental value on this sample (small-n caveats apply).

Outputs: outputs/comparison.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def ks_statistic(y, p):
    order = np.argsort(p)[::-1]
    ys = np.asarray(y)[order]
    return float(np.max(np.abs(np.cumsum(ys) / ys.sum()
                               - np.cumsum(1 - ys) / (1 - ys).sum())))


def main() -> None:
    ratio = pd.read_csv(OUT / "ratio_predictions.csv")[["sample_id", "cik", "label", "pred_ratio"]]
    text = pd.read_csv(OUT / "text_predictions.csv")[["sample_id", "pred_text", "pred_lexicon"]]
    df = ratio.merge(text, on="sample_id", how="inner")
    assert len(df) == len(ratio) == len(text), "sample_id merge mismatch"
    y = df["label"].to_numpy()

    rows = []
    for name, col in [("Ratio logistic (financials only)", "pred_ratio"),
                      ("Negative-word lexicon (text, simple)", "pred_lexicon"),
                      ("MD&A embeddings (text, ML)", "pred_text")]:
        rows.append((name, roc_auc_score(y, df[col]), ks_statistic(y, df[col])))

    # LLM leg, if it has actually run
    llm_path = OUT / "llm_ratings.json"
    llm_note = "not run (no API key yet)"
    if llm_path.exists():
        llm = {e["cik"]: e for e in json.loads(llm_path.read_text())}
        ok = {c: e for c, e in llm.items() if e.get("status") == "ok"}
        if ok:
            sub = df[df["cik"].isin(ok)]
            p = np.array([ok[c]["rating"]["pd_1y"] for c in sub["cik"]])
            rows.append(("LLM zero-shot rating (10-K reader)",
                         roc_auc_score(sub["label"], p),
                         ks_statistic(sub["label"].to_numpy(), p)))
            llm_note = f"ran on {len(ok)} firms"
            ev = [v for e in ok.values()
                  for v in e["rating"].get("evidence_verified", [])]
            if ev:
                llm_note += (f"; evidence quotes verified against source: "
                             f"{sum(ev)}/{len(ev)}")

    # Fusion: ratios + text
    X = df[["pred_ratio", "pred_text"]].to_numpy()
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict_proba(X[te])[:, 1]
    rows.append(("Fusion: ratios + embeddings", roc_auc_score(y, preds),
                 ks_statistic(y, preds)))

    lines = ["# Head-to-head comparison (leave-one-out CV, same firms)",
             "",
             f"Firms: {len(y)} ({int(y.sum())} bankruptcy cases, "
             f"{int((1 - y).sum())} matched controls). LLM leg: {llm_note}.",
             "",
             "| Model | AUC | KS |", "|---|---|---|"]
    lines += [f"| {n} | {a:.3f} | {k:.3f} |" for n, a, k in rows]
    lines += ["",
              "Research question: does text beat / add to ratios on this sample?",
              "Read the fusion row against the ratio row - and remember the",
              "sample is small, so treat any gap < ~0.05 AUC as suggestive,",
              "not conclusive. Expanding the corpus is Step 0's --scale TODO.",
              ""]
    (OUT / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
