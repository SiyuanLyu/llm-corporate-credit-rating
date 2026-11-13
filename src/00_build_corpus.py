"""
Step 0 - Build the case-control corpus from SEC EDGAR (all public, no key)
==========================================================================
CASES: firms with a VERIFIED Chapter 11 event - an 8-K whose items include
1.03 (Bankruptcy or Receivership) in their EDGAR filing history. The
candidate CIK list below is only a starting point; every candidate is
verified against EDGAR and dropped if the event or a usable pre-event
10-K cannot be found. For each case we take the LAST 10-K filed BEFORE
the first bankruptcy 8-K (point-in-time: no information from after the
filing date is used anywhere in this project).

CONTROLS: currently listed firms with no Item 1.03 8-K, matched to cases
by SIC industry and 10-K year (up to 2 per case), drawn from EDGAR's
company_tickers.json.

Text extraction: MD&A (Item 7) and Risk Factors (Item 1A) sections are
sliced out of the 10-K HTML and saved as plain text.

Outputs
-------
data/processed/corpus_index.csv   one row per firm-year sample
data/text/{cik}_{fy}_mda.txt      MD&A text (truncated at 80k chars)
data/text/{cik}_{fy}_risk.txt     Risk Factors text (truncated)
"""
import html as html_lib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TEXT = ROOT / "data" / "text"
PROC = ROOT / "data" / "processed"
for d in (RAW, TEXT, PROC):
    d.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Siyuan-Lyu-Research sylyu2026@gmail.com"}
S = requests.Session()
S.headers.update(HEADERS)


def get_json(url: str):
    last = None
    for attempt in range(3):
        time.sleep(0.15 * (attempt + 1))
        try:
            r = S.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:      # transient proxy/network errors: retry
            last = e
    raise last


def all_filings(cik: int) -> pd.DataFrame:
    """Every filing row EDGAR lists for a CIK (recent + history files)."""
    base = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    frames = []
    recent = base["filings"]["recent"]
    frames.append(pd.DataFrame(recent))
    for extra in base["filings"].get("files", []):
        try:
            frames.append(pd.DataFrame(
                get_json(f"https://data.sec.gov/submissions/{extra['name']}")))
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True)
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    return df.sort_values("filingDate"), base


def strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html_lib.unescape(raw)
    # Filings mix straight and typographic quotes; normalise so section
    # headings like "Management's Discussion and Analysis" always match.
    text = (text.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))
    return re.sub(r"\s+", " ", text).strip()


def slice_section(text: str, start_pats, end_pats, cap=80_000) -> str:
    low = text.lower()
    start = -1
    for pat in start_pats:
        idx = low.rfind(pat)          # last occurrence = body, not the TOC
        if idx > start:
            start = idx
    if start < 0:
        return ""
    end = len(text)
    for pat in end_pats:
        idx = low.find(pat, start + 500)
        if idx > 0:
            end = min(end, idx)
    return text[start:end][:cap]


def download_10k_text(cik: int, row) -> tuple[str, str]:
    acc = row["accessionNumber"].replace("-", "")
    doc = row["primaryDocument"]
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
    time.sleep(0.15)
    r = S.get(url, timeout=60)
    r.raise_for_status()
    (RAW / f"{cik}_{row['accessionNumber']}.html").write_bytes(r.content)
    text = strip_html(r.text)
    mda = slice_section(
        text,
        ["management's discussion and analysis"],
        ["quantitative and qualitative disclosures about market risk",
         "financial statements and supplementary data"])
    risk = slice_section(
        text,
        ["risk factors"],
        ["unresolved staff comments", "properties", "legal proceedings"])
    return mda, risk


# Candidate bankrupt firms (CIK). VERIFIED in code below; bad ones drop out.
CANDIDATES = {
    886158: "Bed Bath & Beyond", 47129: "Hertz",
    1255474: "Whiting Petroleum", 949158: "Diamond Offshore",
    1310067: "Sears Holdings",
    1004980: "PG&E", 1282266: "Windstream", 84129: "Rite Aid",
    719739: "SVB Financial", 1813756: "WeWork", 1567892: "Mallinckrodt",
    1024401: "Enron", 1326428: "Linn Energy", 1349436: "SandRidge Energy",
    1025315: "Ultra Petroleum", 1055722: "Halcon Resources",
    1403939: "iHeartMedia", 887921: "Revlon",
    1518235: "Party City", 1625135: "Neiman Marcus",
}


def find_case(cik: int, label: str):
    try:
        filings, base = all_filings(cik)
    except Exception as e:
        print(f"  [drop] {label}: submissions fetch failed ({e})")
        return None
    bk = filings[(filings["form"] == "8-K")
                 & (filings["items"].str.contains("1.03", na=False))]
    if bk.empty:
        print(f"  [drop] {label}: no Item 1.03 bankruptcy 8-K found")
        return None
    event_date = bk["filingDate"].min()
    k = filings[(filings["form"] == "10-K")
                & (filings["filingDate"] < event_date)
                & (filings["filingDate"] > event_date - pd.Timedelta(days=550))]
    if k.empty:
        print(f"  [drop] {label}: no 10-K within 18m before {event_date.date()}")
        return None
    row = k.iloc[-1]
    try:
        mda, risk = download_10k_text(cik, row)
    except Exception as e:
        print(f"  [drop] {label}: 10-K download failed ({e})")
        return None
    if len(mda) < 2000:
        print(f"  [drop] {label}: MD&A extraction too short ({len(mda)} chars)")
        return None
    fy = int(row["filingDate"].year)
    (TEXT / f"{cik}_{fy}_mda.txt").write_text(mda)
    (TEXT / f"{cik}_{fy}_risk.txt").write_text(risk)
    print(f"  [case] {label}: event {event_date.date()}, 10-K {row['filingDate'].date()}, "
          f"MD&A {len(mda)} chars, SIC {base.get('sic')}")
    return {"cik": cik, "name": base.get("name", label),
            "sic": str(base.get("sic") or ""), "label": 1,
            "event_date": event_date.date().isoformat(),
            "filing_date": row["filingDate"].date().isoformat(),
            "filing_year": fy, "accession": row["accessionNumber"]}


def find_controls(cases: list[dict], per_case=2) -> list[dict]:
    tickers = get_json("https://www.sec.gov/files/company_tickers.json")
    pool = pd.DataFrame(tickers.values())
    pool = pool.sample(n=min(1600, len(pool)), random_state=42)
    controls, used = [], set()
    by_sic: dict[str, list] = {}
    for _, t in pool.iterrows():
        if len(controls) >= per_case * len(cases):
            break
        cik = int(t["cik_str"])
        if cik in CANDIDATES or cik in used:
            continue
        try:
            filings, base = all_filings(cik)
        except Exception:
            continue
        if filings[(filings["form"] == "8-K")
                   & (filings["items"].str.contains("1.03", na=False))].shape[0]:
            continue                      # any bankruptcy history -> not a control
        sic = str(base.get("sic") or "")
        ks = filings[filings["form"] == "10-K"]
        if ks.empty or not sic:
            continue
        by_sic.setdefault(sic, []).append((cik, base, ks))
        used.add(cik)
    for case in cases:
        cands = by_sic.get(case["sic"], [])
        if not cands:                     # fall back to 2-digit SIC
            cands = [v for s, lst in by_sic.items()
                     if s[:2] == case["sic"][:2] for v in lst]
        picked = 0
        for cik, base, ks in cands:
            if picked >= per_case:
                break
            same_year = ks[ks["filingDate"].dt.year == case["filing_year"]]
            if same_year.empty:   # accept a 10-K one year either side
                near = ks[(ks["filingDate"].dt.year - case["filing_year"]).abs() <= 1]
                if near.empty:
                    continue
                same_year = near
            row = same_year.iloc[-1]
            try:
                mda, risk = download_10k_text(cik, row)
            except Exception:
                continue
            if len(mda) < 2000:
                continue
            fy = int(row["filingDate"].date().year)
            (TEXT / f"{cik}_{fy}_mda.txt").write_text(mda)
            (TEXT / f"{cik}_{fy}_risk.txt").write_text(risk)
            controls.append({"cik": cik, "name": base.get("name", ""),
                             "sic": case["sic"], "label": 0, "event_date": "",
                             "filing_date": row["filingDate"].date().isoformat(),
                             "filing_year": fy,
                             "accession": row["accessionNumber"]})
            print(f"  [control] {base.get('name')} (SIC {case['sic']}, {fy}) "
                  f"for case {case['name']}")
            picked += 1
    return controls


def main() -> None:
    print("Verifying bankruptcy cases on EDGAR...")
    cases = [c for cik, lab in CANDIDATES.items()
             if (c := find_case(cik, lab)) is not None]
    print(f"Cases verified: {len(cases)}")
    print("Finding matched controls...")
    controls = find_controls(cases)
    print(f"Controls found: {len(controls)}")
    idx = pd.DataFrame(cases + controls)
    # The same firm-year can be matched to two cases; keep it once -
    # duplicate samples would leak across LOOCV folds later.
    idx = idx.drop_duplicates(subset=["cik", "filing_year"]).reset_index(drop=True)
    idx.to_csv(PROC / "corpus_index.csv", index=False)
    print(idx.groupby("label").size().rename({0: "controls", 1: "cases"}))


if __name__ == "__main__":
    main()
