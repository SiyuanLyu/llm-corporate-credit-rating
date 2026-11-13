"""
Step 1 - Benchmark #1: financial-ratio model (the "traditional" leg)
====================================================================
Pulls structured financials for each corpus firm's 10-K fiscal year from
EDGAR's XBRL company-facts API, builds Altman-style ratios and fits a
logistic regression. With a tiny case-control sample we report
leave-one-out cross-validated (LOOCV) AUC/KS - honest small-sample
evaluation, no random split luck.

Ratios: interest coverage, leverage (liabilities/assets), ROA,
current ratio, equity ratio, revenue growth, plus an Altman Z''-style
composite where data permit.

Outputs: outputs/ratio_metrics.json, outputs/ratio_features.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
S = requests.Session()
S.headers.update({"User-Agent": "Siyuan-Lyu-Research sylyu2026@gmail.com"})

TAGS = {
    "assets": "Assets", "assets_cur": "AssetsCurrent",
    "liab": "Liabilities", "liab_cur": "LiabilitiesCurrent",
    "equity": "StockholdersEquity",
    "revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "revenue_alt": "Revenues",
    "opinc": "OperatingIncomeLoss", "netinc": "NetIncomeLoss",
    "interest": "InterestExpense", "interest_alt": "InterestExpenseNonoperating",
    "wc_proxy": "AssetsCurrent",
}


def fact(facts: dict, tag: str, fy_target: int):
    """Most recent annual (10-K, FP=FY) value for tag with FY near target."""
    units = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {})
    rows = []
    for unit, vals in units.items():
        if unit not in ("USD",):
            continue
        for v in vals:
            if v.get("form") == "10-K" and v.get("fp") == "FY" and v.get("fy"):
                rows.append((abs(v["fy"] - fy_target), v["fy"], v["val"]))
    if not rows:
        return np.nan
    rows.sort()
    return rows[0][2]


def ks_statistic(y, p):
    order = np.argsort(p)[::-1]
    ys = np.asarray(y)[order]
    return float(np.max(np.abs(np.cumsum(ys) / ys.sum()
                               - np.cumsum(1 - ys) / (1 - ys).sum())))


def main() -> None:
    idx = pd.read_csv(PROC / "corpus_index.csv")
    rows = []
    for _, r in idx.iterrows():
        time.sleep(0.15)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(r['cik']):010d}.json"
        try:
            facts = S.get(url, timeout=30).json()
        except Exception:
            facts = {}
        fy = int(r["filing_year"]) - 1          # 10-K covers prior fiscal year
        g = {k: fact(facts, t, fy) for k, t in TAGS.items()}
        rev = g["revenue"] if not np.isnan(g["revenue"]) else g["revenue_alt"]
        intr = g["interest"] if not np.isnan(g["interest"]) else g["interest_alt"]
        eps = 1e-9
        rows.append({
            "cik": r["cik"], "filing_year": r["filing_year"],
            "sample_id": f"{int(r['cik'])}_{int(r['filing_year'])}",
            "label": r["label"],
            "interest_coverage": g["opinc"] / (intr + eps) if not np.isnan(g["opinc"]) and not np.isnan(intr) else np.nan,
            "leverage": g["liab"] / (g["assets"] + eps) if not np.isnan(g["liab"]) and not np.isnan(g["assets"]) else np.nan,
            "roa": g["netinc"] / (g["assets"] + eps) if not np.isnan(g["netinc"]) and not np.isnan(g["assets"]) else np.nan,
            "current_ratio": g["assets_cur"] / (g["liab_cur"] + eps) if not np.isnan(g["assets_cur"]) and not np.isnan(g["liab_cur"]) else np.nan,
            "equity_ratio": g["equity"] / (g["assets"] + eps) if not np.isnan(g["equity"]) and not np.isnan(g["assets"]) else np.nan,
            "log_assets": np.log(g["assets"]) if not np.isnan(g["assets"]) and g["assets"] > 0 else np.nan,
        })
        print(f"  ratios {r['name']}: cov={rows[-1]['interest_coverage']}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ratio_features.csv", index=False)

    feats = [c for c in df.columns
             if c not in ("cik", "filing_year", "sample_id", "label")]
    X, y = df[feats], df["label"].to_numpy()
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=float)
    for tr, te in loo.split(X):
        pipe = make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(),
                             LogisticRegression(max_iter=1000, C=1.0))
        pipe.fit(X.iloc[tr], y[tr])
        preds[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, preds)) if len(set(y)) > 1 else float("nan")
    result = {"model": "Ratio logistic (Altman-style features)",
              "n": int(len(y)), "n_cases": int(y.sum()),
              "loocv": {"auc": auc, "ks": ks_statistic(y, preds)},
              "features": feats}
    df.assign(pred_ratio=preds).to_csv(OUT / "ratio_predictions.csv", index=False)
    (OUT / "ratio_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
