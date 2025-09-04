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
    """
    Return (text, source_url). Only uses the provided URL.
    - If URL serves a PDF: parse it.
    - If HTML: try to find a PDF link on the page; else return visible HTML text.
    On failure: ("", "").
    """
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

    # HTML landing → hunt for a PDF link; fallback to page text
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

        # fallback: visible HTML text
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        return text, url
    except Exception:
        return "", ""

# ---------------- Age extraction ----------------

@dataclass
class AgeEvidence:
    kind: str                # "mean" | "median" | "range" | "single" | "grade"
    value: Optional[float]   # for mean/median/single
    low: Optional[float]     # for range/grade low
    high: Optional[float]    # for range/grade high
    context: str             # snippet for audit

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def grade_to_age_bounds(g: int) -> Tuple[int, int]:
    # coarse mapping (G1≈6–7 … G12≈17–18)
    low = 6 + (g - 1)
    high = low + 1
    return low, high

def extract_age_evidence(text: str) -> List[AgeEvidence]:
    if not text:
        return []
    t = text.replace("–", "-").replace("—", "-")
    evid: List[AgeEvidence] = []

    def ctx(i0: int, i1: int) -> str:
        lo = max(0, i0 - 80)
        hi = min(len(t), i1 + 80)
        return _clean(t[lo:hi])

    # mean / average age
    for m in re.finditer(r"(mean|average)\s+age\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)", t, flags=re.I):
        evid.append(AgeEvidence("mean", float(m.group(2)), None, None, ctx(*m.span())))

    # median age
    for m in re.finditer(r"median\s+age\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)", t, flags=re.I):
        evid.append(AgeEvidence("median", float(m.group(1)), None, None, ctx(*m.span())))

    # ranges: "aged 12-16"
    for m in re.finditer(r"aged\s+(\d{1,2})\s*-\s*(\d{1,2})", t, flags=re.I):
        lo, hi = int(m.group(1)), int(m.group(2))
        evid.append(AgeEvidence("range", None, float(lo), float(hi), ctx(*m.span())))

    # other range phrasing: "between 3 and 10 years", "3 to 10 years"
    for m in re.finditer(r"(?:between\s+)?(\d{1,2})\s*(?:to|and|-)\s*(\d{1,2})\s*(?:years|y\.o\.)", t, flags=re.I):
        lo, hi = int(m.group(1)), int(m.group(2))
        evid.append(AgeEvidence("range", None, float(lo), float(hi), ctx(*m.span())))

    # single age: "15 years old" / "15 y.o."
    for m in re.finditer(r"(\d{1,2})\s*(?:years?\s*old|y\.o\.)", t, flags=re.I):
        evid.append(AgeEvidence("single", float(m.group(1)), None, None, ctx(*m.span())))

    # grades: "grade 7-9"
    for m in re.finditer(r"grade\s+(\d{1,2})(?:\s*-\s*(\d{1,2}))?", t, flags=re.I):
        g1 = int(m.group(1))
        g2 = int(m.group(2)) if m.group(2) else g1
        lo1, hi1 = grade_to_age_bounds(g1)
        lo2, hi2 = grade_to_age_bounds(g2)
        lo = min(lo1, lo2); hi = max(hi1, hi2)
        evid.append(AgeEvidence("grade", None, float(lo), float(hi), ctx(*m.span())))

    return evid

# ---------------- Decision ----------------

@dataclass
class AgeDecision:
    covidence_id: str
    decision: str           # "Yes" | "No" | "Maybe" | "UnknownAge"
    reasons: List[str]
    evidence: List[Dict]
    source_url: str

def _in_window(v: float) -> bool:
    return AGE_MIN <= v <= AGE_MAX

def _overlaps_window(lo: float, hi: float) -> bool:
    return not (hi < AGE_MIN or lo > AGE_MAX)

def decide_age(evid: List[AgeEvidence], covidence_id: str, source_url: str) -> AgeDecision:
    if not evid:
        return AgeDecision(
            covidence_id=covidence_id,
            decision="UnknownAge",
            reasons=["No age evidence found; consider checking supplements."],
            evidence=[],
            source_url=source_url,
        )

    any_in = False
    any_out = False
    for e in evid:
        if e.kind in {"mean", "median", "single"} and e.value is not None:
            if _in_window(e.value):
                any_in = True
            else:
                any_out = True
        elif e.kind in {"range", "grade"} and e.low is not None and e.high is not None:
            if _overlaps_window(e.low, e.high):
                any_in = True
            else:
                any_out = True

    if any_in and any_out:
        return AgeDecision(
            covidence_id=covidence_id,
            decision="Maybe",
            reasons=["Mixed cohorts: in-range (2–17) and out-of-range (<2 or ≥18)."],
            evidence=[asdict(e) for e in evid],
            source_url=source_url,
        )
    if any_in:
        return AgeDecision(
            covidence_id=covidence_id,
            decision="Yes",
            reasons=["At least one cohort falls within 2–17."],
            evidence=[asdict(e) for e in evid],
            source_url=source_url,
        )
    return AgeDecision(
        covidence_id=covidence_id,
        decision="No",
        reasons=["All detected age evidence outside 2–17 (<2 or ≥18)."],
        evidence=[asdict(e) for e in evid],
        source_url=source_url,
    )

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Screen mean/age ranges from URL full text against 2–17 window.")
    ap.add_argument("csv_path", help="CSV with covidence_id and a URL column")
    ap.add_argument("--url-col", default="found_url", help="Column containing the URL (default: found_url)")
    ap.add_argument("--n", type=int, default=10, help="Max rows to process")
    ap.add_argument("--out", default=None, help="Optional output CSV for decisions")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    url_col = args.url_col.lower().strip()
    if url_col not in df.columns:
        raise ValueError(f"URL column '{args.url_col}' not found. Available: {list(df.columns)}")
    if "covidence_id" not in df.columns:
        df["covidence_id"] = ""

    rows = []
    processed = 0
    for _, row in df.iterrows():
        if processed >= args.n:
            break
        cov_id = str(row.get("covidence_id", "")).strip() or "(unknown-id)"
        url = str(row.get(url_col, "")).strip()

        if not url:
            rows.append({
                "covidence_id": cov_id,
                "decision": "UnknownAge",
                "reasons": "No URL provided",
                "source_url": "",
                "evidence": "",
            })
            processed += 1
            continue

        text, src = fetch_fulltext_from_url(url)
        evid = extract_age_evidence(text) if text else []
        dec = decide_age(evid, cov_id, src)

        rows.append({
            "covidence_id": cov_id,
            "decision": dec.decision,
            "reasons": "; ".join(dec.reasons),
            "source_url": dec.source_url,
            "evidence": dec.evidence,  # list of dicts for auditing
        })
        processed += 1
        time.sleep(0.3)  # be polite to hosts

    if args.out:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
