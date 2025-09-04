from __future__ import annotations
import argparse, io, re, time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text

UA = "emily-research-screen/1.0 (mailto:xmeng05@uw.edu)"
AGE_MIN = 2
AGE_MAX = 17

def polite_get(url: str, timeout=25) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code == 200:
            return r
    except Exception:
        return None
    return None

def looks_like_pdf_url(url: str) -> bool:
    u = (url or "").lower()
    return u.endswith(".pdf") or "pdf" in u.split("?")[0].split("#")[0]

def fetch_fulltext_from_url(url: str) -> Tuple[str, str]:
    if not url or not url.lower().startswith("http"):
        return "", ""
    r = polite_get(url)
    if not r:
        return "", ""
    ct = (r.headers.get("Content-Type") or "").lower()
    if "pdf" in ct or looks_like_pdf_url(url):
        try:
            text = pdf_extract_text(io.BytesIO(r.content))
            return (text or "", url)
        except Exception:
            return "", ""
    try:
        html = r.text
        soup = BeautifulSoup(html, "lxml")
        cand = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            label = (a.get_text() or "").lower()
            if "pdf" in href.lower() or "pdf" in label:
                cand.append(href)
        from urllib.parse import urljoin
        pdf_urls = [urljoin(url, h) for h in cand]
        for pu in pdf_urls[:3]:
            pr = polite_get(pu)
            if pr and ("pdf" in (pr.headers.get("Content-Type", "").lower()) or looks_like_pdf_url(pu)):
                try:
                    text = pdf_extract_text(io.BytesIO(pr.content))
                    if text:
                        return text, pu
                except Exception:
                    continue
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        return text, url
    except Exception:
        return "", ""

@dataclass
class AgeEvidence:
    kind: str
    value: Optional[float]
    low: Optional[float]
    high: Optional[float]
    context: str

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def grade_to_age_bounds(g: int) -> Tuple[int, int]:
    low = 6 + (g - 1)
    high = low + 1
    return low, high

def extract_age_evidence(text: str) -> List[AgeEvidence]:
    if not text:
        return []
    t = text.replace("–", "-").replace("—", "-")

    def ctx(i0: int, i1: int) -> str:
        lo = max(0, i0 - 80)
        hi = min(len(t), i1 + 80)
        return _clean(t[lo:hi])

    evid: List[AgeEvidence] = []
    seen = set()  # dedupe by (kind, value, low, high, short_ctx)

    # mean / average age (allow "of"/"was", "=", ":", and optional "years")
    for m in re.finditer(
        r"(?:mean|average)\s+age(?:\s+at\s+baseline)?\s*(?:of|was|=|:)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:years?|y\.o\.)?",
        t, flags=re.I):
        c = ctx(*m.span())
        key = ("mean", float(m.group(1)), None, None, c[:120])
        if key not in seen:
            seen.add(key)
            evid.append(AgeEvidence("mean", float(m.group(1)), None, None, c))

    # median age (same flexibility)
    for m in re.finditer(
        r"median\s+age(?:\s+at\s+baseline)?\s*(?:of|was|=|:)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:years?|y\.o\.)?",
        t, flags=re.I):
        c = ctx(*m.span())
        key = ("median", float(m.group(1)), None, None, c[:120])
        if key not in seen:
            seen.add(key)
            evid.append(AgeEvidence("median", float(m.group(1)), None, None, c))

    # ranges: "aged 12-16"
    for m in re.finditer(r"aged\s+(\d{1,2})\s*-\s*(\d{1,2})", t, flags=re.I):
        lo, hi = int(m.group(1)), int(m.group(2))
        c = ctx(*m.span())
        key = ("range", None, float(lo), float(hi), c[:120])
        if key not in seen:
            seen.add(key)
            evid.append(AgeEvidence("range", None, float(lo), float(hi), c))

    # other range phrasing: "between 3 and 10 years", "3 to 10 years"
    for m in re.finditer(r"(?:between\s+)?(\d{1,2})\s*(?:to|and|-)\s*(\d{1,2})\s*(?:years|y\.o\.)", t, flags=re.I):
        lo, hi = int(m.group(1)), int(m.group(2))
        c = ctx(*m.span())
        key = ("range", None, float(lo), float(hi), c[:120])
        if key not in seen:
            seen.add(key)
            evid.append(AgeEvidence("range", None, float(lo), float(hi), c))

    # single age: "15 years old" / "15 y.o."
    for m in re.finditer(r"(\d{1,2})\s*(?:years?\s*old|y\.o\.)", t, flags=re.I):
        val = float(m.group(1))
        c = ctx(*m.span())
        key = ("single", val, None, None, c[:120])
        if key not in seen:
            seen.add(key)
            evid.append(AgeEvidence("single", val, None, None, c))

    for m in re.finditer(r"grade\s+(\d{1,2})(?:\s*-\s*(\d{1,2}))?", t, flags=re.I):
        g1 = int(m.group(1))
        g2 = int(m.group(2)) if m.group(2) else g1
        lo1, hi1 = grade_to_age_bounds(g1)
        lo2, hi2 = grade_to_age_bounds(g2)
        lo = min(lo1, lo2); hi = max(hi1, hi2)
        c = ctx(*m.span())
        key = ("grade", None, float(lo), float(hi), c[:120])
        if key not in seen:
            seen.add(key)
            evid.append(AgeEvidence("grade", None, float(lo), float(hi), c))

    #months 
    # 115 month / 12 
    return evid

@dataclass
class AgeDecision:
    covidence_num: str
    decision: str           # "Yes" | "No" | "Maybe" | "UnknownAge"
    reasons: List[str]
    evidence: List[Dict]
    source_url: str

def _in_window(v: float) -> bool:
    return AGE_MIN <= v <= AGE_MAX

def _overlaps_window(lo: float, hi: float) -> bool:
    return not (hi < AGE_MIN or lo > AGE_MAX)

def decide_age(evid: List[AgeEvidence], covidence_num: str, source_url: str) -> AgeDecision:
    if not evid:
        return AgeDecision(
            covidence_num=covidence_num,
            decision="UnknownAge",
            reasons=["No age evidence found; consider checking supplements."],
            evidence=[],
            source_url=source_url,
        )

    # detect baseline-only phrasing anywhere in the evidence context
    baseline_flag = any(("baseline" in e.context.lower()) for e in evid)

    # we only give a hard YES if there's an in-range MEAN
    mean_in = any(e.kind == "mean" and e.value is not None and _in_window(e.value) for e in evid)
    mean_out = any(e.kind == "mean" and e.value is not None and not _in_window(e.value) for e in evid)

    # other signals (range/grade/single/median) — these make it Maybe, not Yes
    other_in = any(
        (e.kind in {"range", "grade"} and e.low is not None and e.high is not None and _overlaps_window(e.low, e.high))
        or (e.kind in {"single", "median"} and e.value is not None and _in_window(e.value))
        for e in evid
    )
    other_out = any(
        (e.kind in {"range", "grade"} and e.low is not None and e.high is not None and not _overlaps_window(e.low, e.high))
        or (e.kind in {"single", "median"} and e.value is not None and not _in_window(e.value))
        for e in evid
    )

    # rule 1: baseline-only → Maybe 
    if baseline_flag and not mean_in:
        return AgeDecision(
            covidence_num, "Maybe",
            ["Age described at baseline only; no in-range mean reported."],
            [asdict(e) for e in evid], source_url
        )

    # rule 2: hard Yes only when mean age is in-range (2–17)
    if mean_in:
        # mixed cohorts? (there is also evidence clearly out-of-range)
        if mean_out or other_out:
            return AgeDecision(
                covidence_num, "Maybe",
                ["In-range mean detected, but mixed cohorts also out-of-range."],
                [asdict(e) for e in evid], source_url
            )
        return AgeDecision(
            covidence_num, "Yes",
            ["In-range mean age detected (2–17)."],
            [asdict(e) for e in evid], source_url
        )

    # rule 3: no mean; but other signals in range → Maybe
    if other_in:
        return AgeDecision(
            covidence_num, "Maybe",
            ["Age appears in range, but no mean reported."],
            [asdict(e) for e in evid], source_url
        )

    # rule 4: everything we saw is outside → No
    if other_out:
        return AgeDecision(
            covidence_num, "No",
            ["All detected age evidence outside 2–17 (<2 or ≥18)."],
            [asdict(e) for e in evid], source_url
        )

    # fallback
    return AgeDecision(
        covidence_num, "UnknownAge", 
        ["Age not determinable; consider checking supplements."],
        [asdict(e) for e in evid], source_url
    )


def main():
    ap = argparse.ArgumentParser(description="Screen mean/age ranges from URL full text against 2–17 window.")
    ap.add_argument("csv_path", help="CSV from find_links.py (must have 'Covidence #' and a URL column)")
    ap.add_argument("--url-col", default="found_url", help="Column containing the URL (default: found_url)")
    ap.add_argument("--n", type=int, default=10, help="Max rows to process")
    ap.add_argument("--out", default=None, help="Optional output CSV for decisions")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    # case-insensitive mapping
    colmap = {c.lower().strip(): c for c in df.columns}
    url_col_real = colmap.get(args.url_col.lower())
    if not url_col_real:
        raise ValueError(f"URL column '{args.url_col}' not found. Available: {list(df.columns)}")
    cov_col_real = colmap.get("covidence #")
    if not cov_col_real:
        raise ValueError("Input must include a 'Covidence #' column (e.g., '#293').")

    rows = []
    processed = 0
    for _, row in df.iterrows():
        if processed >= args.n:
            break

        raw_cov = str(row.get(cov_col_real, "")).strip()
        cov_num = raw_cov.replace("#", "") if raw_cov else "(unknown-id)"
        url = str(row.get(url_col_real, "")).strip()

        if not url:
            rows.append({
                "Covidence #": cov_num,
                "decision": "UnknownAge",
                "reasons": "No URL provided",
                "source_url": "",
                "evidence": "",
            })
            processed += 1
            continue

        text, src = fetch_fulltext_from_url(url)
        evid = extract_age_evidence(text) if text else []
        dec = decide_age(evid, cov_num, src)

        rows.append({
            "Covidence #": dec.covidence_num,
            "decision": dec.decision,
            "reasons": "; ".join(dec.reasons),
            "source_url": dec.source_url,
            "evidence": dec.evidence,
        })
        processed += 1
        time.sleep(0.3)

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
