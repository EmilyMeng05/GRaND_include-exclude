from __future__ import annotations
import argparse
import re
import time
from typing import Optional, Tuple, Dict
import pandas as pd
import requests

# so that if we made any mistake, it will contact me
UA = "emily-research-screen/1.0 (mailto:xmeng05@uw.edu)"
# AI TOOLS for search
CROSSREF = "https://api.crossref.org/works"
# another AI tools for search
SERPAPI = "https://serpapi.com/search.json"

# return true if doi is missing from covidence csv
def is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none"}

# remove all doi prefixes, return a clean version of doi
def normalize_doi(s: Optional[str]) -> Optional[str]:
    if is_missing(s):
        return None
    s = str(s).strip()
    s = re.sub(r"^(https?://(dx\.)?doi\.org/)", "", s, flags=re.I)
    s = re.sub(r"^(doi:)\s*", "", s, flags=re.I)
    m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", s, flags=re.I)
    return m.group(1) if m else None

# convert doi to an url link
def doi_to_url(doi: Optional[str]) -> Optional[str]:
    return f"https://doi.org/{doi}" if doi else None

# return a clean version for the first author
def first_author_from(authors: str) -> Optional[str]:
    if is_missing(authors):
        return None
    s = str(authors)
    s = s.split(';')[0]
    s = s.split(' and ')[0]
    s = s.split(',')[0]
    s = s.strip()
    return s or None

# return a clean version for the title
def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip())

# good for CROSSREF, remove subtitles
def short_title(title: str) -> str:
    t = normalize_title(title)
    t = t.split(" - ")[0]
    t = t.split(":")[0]
    return t.strip()

# generic HTTP GET helper
def polite_get(url: str, params: dict, timeout=25) -> Optional[requests.Response]:
    try:
        r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code == 200:
            return r
    except Exception:
        return None
    return None

# letting crossref rest, cuz we don't wanna ask for api link every seconds
def crossref_top_item(params: Dict, sleep_sec: float = 0.3):
    r = polite_get(CROSSREF, {**params, "rows": 1})
    if not r:
        time.sleep(sleep_sec); return None
    try:
        items = r.json().get("message", {}).get("items", [])
        time.sleep(sleep_sec)
        return items[0] if items else None
    except Exception:
        time.sleep(sleep_sec); return None

# search using different ways
# 1. title + author
# 2. short title + author
def robust_crossref_find_doi(title: str, first_author: Optional[str], sleep_sec: float = 0.3
                            ) -> Tuple[str, str]:
    """
    Returns (found_doi, found_url).
    If nothing found after multiple strategies, returns ("NOT_FOUND", "").
    """
    title = normalize_title(title)
    short = short_title(title)

    # 1) title + author
    params = {"query.title": title}
    if first_author:
        params["query.author"] = first_author
    it = crossref_top_item(params, sleep_sec)
    if it and it.get("DOI"):
        doi = normalize_doi(it.get("DOI"))
        url = it.get("URL") or doi_to_url(doi)
        return (doi or "NOT_FOUND", url or "")

    # 2) short title + author
    if short and short.lower() != title.lower():
        params = {"query.title": short}
        if first_author:
            params["query.author"] = first_author
        it = crossref_top_item(params, sleep_sec)
        if it and it.get("DOI"):
            doi = normalize_doi(it.get("DOI"))
            url = it.get("URL") or doi_to_url(doi)
            return (doi or "NOT_FOUND", url or "")
        
    return ("NOT_FOUND", "")

# if crossref didn't find anything, try searching in google scholar 
# if still nothing is found, return not found
def serpapi_scholar_link(title: str, author: Optional[str], api_key: str) -> Tuple[str, str]:
    try:
        q = f"{title} {author}" if author else title
        r = requests.get(
            SERPAPI,
            params={"engine": "google_scholar", "q": q, "hl": "en", "api_key": api_key},
            timeout=25,
        )
        if r.status_code != 200:
            return ("NOT_FOUND", "")
        data = r.json()
        results = data.get("organic_results") or []
        for res in results:
            link = res.get("link") or ""
            if not link:
                continue
            m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", link, flags=re.I)
            if m:
                doi = normalize_doi(m.group(1))
                return (doi or "NOT_FOUND", link)
        if results:
            return ("NOT_FOUND", results[0].get("link") or "")
        return ("NOT_FOUND", "")
    except Exception:
        return ("NOT_FOUND", "")
    
# main function!
# parse through csv 
# load covidence csv into dataframe
# for each row, 
#   if doi is present, use it
#   if doi is missing, try robust_crossref_find_doi to search 
#   if still crossref can't find, try serpapi_scholar_link to search in google scholar
# If --out is provided, write the results CSV; otherwise run silently.
def main():
    ap = argparse.ArgumentParser(description="Resolve missing DOIs from a Covidence CSV via Crossref (and optional Scholar fallback).")
    ap.add_argument("csv_path", help="Path to covidence.csv")
    ap.add_argument("--n", type=int, default=20, help="How many rows to try")
    ap.add_argument("--serpapi-key", default=None, help="SerpAPI key to enable Google Scholar fallback")
    ap.add_argument("--out", default=None, help="Optional path to write a results CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Covidence #" not in df.columns:
        df["Covidence #"] = ""

    for col in ["title", "authors", "doi", "url"]:
        if col.lower() not in [c.lower() for c in df.columns]:
            df[col] = ""

    rows = []
    tried = 0
    for _, row in df.iterrows():
        if tried >= args.n:
            break

        cov_id = str(row.get("Covidence #", "")).strip()
        cov_id = cov_id.replace("#", "") if cov_id else ""

        title   = str(row.get("title", "")).strip()
        authors = str(row.get("authors", "")).strip()
        doi_csv = normalize_doi(row.get("doi", ""))

        first_author = first_author_from(authors)

        source = "csv"
        found_doi = doi_csv or ""
        found_url = str(row.get("url", "")).strip() or (doi_to_url(doi_csv) if doi_csv else "")

        if not doi_csv:
            found_doi, found_url = robust_crossref_find_doi(title, first_author)
            source = "crossref"
            if found_doi == "NOT_FOUND" and args.serpapi_key:
                s_doi, s_url = serpapi_scholar_link(title, first_author, args.serpapi_key)
                if s_doi != "NOT_FOUND":
                    found_doi, found_url, source = s_doi, (s_url or doi_to_url(s_doi)), "scholar"

        rows.append({
            "Covidence #": cov_id,
            "title": title,
            "authors": authors,
            "csv_doi": doi_csv or "",
            "found_doi": found_doi or "NOT_FOUND",
            "found_url": found_url or "",
            "source": source,
        })
        time.sleep(0.2)
        tried += 1

    if args.out:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
