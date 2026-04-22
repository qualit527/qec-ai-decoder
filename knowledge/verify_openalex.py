"""
Backfill venue info via OpenAlex API for entries where Semantic Scholar failed or returned no venue.

OpenAlex is free, requires no API key, and has very generous rate limits (~10 req/s).
We query by arxiv ID via the doi-like identifier: https://api.openalex.org/works/https://doi.org/10.48550/arXiv.<arxiv_id>
Fallback: query by title search.

Merges results into venue_info.json and regenerates classification.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent.resolve()
VENUE_INFO = HERE / "venue_info.json"
BIB_FILE = HERE / "bibliography.bib"
INDEX_FILE = HERE / "INDEX.md"
LOG_FILE = HERE / "venue_check.log"

OA_BASE = "https://api.openalex.org/works"
MAILTO = "autoqec@example.org"  # OpenAlex polite-pool header
USER_AGENT = f"AutoQEC-knowledge/1.0 (mailto:{MAILTO})"


def oa_query_by_arxiv(arxiv_id: str) -> dict | None:
    """Try OpenAlex direct arxiv lookup."""
    # OpenAlex indexes arxiv preprints with DOI 10.48550/arXiv.<id>
    doi = f"10.48550/arXiv.{arxiv_id}"
    url = f"{OA_BASE}/doi:{urllib.parse.quote(doi, safe='')}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        return {"__error": f"HTTP {e.code}"}
    except Exception as e:
        return {"__error": f"{type(e).__name__}: {e}"}


def oa_query_by_title(title: str, year_hint: int | None = None) -> dict | None:
    """Fallback: title search."""
    if not title:
        return None
    # Clean title for search
    clean = re.sub(r"[^\w\s]", " ", title)[:120]
    url = f"{OA_BASE}?search={urllib.parse.quote(clean)}&per_page=5"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                return None
            # Pick the closest by title + year
            best = results[0]
            if year_hint:
                for r in results:
                    if r.get("publication_year") == year_hint:
                        best = r
                        break
            return best
    except Exception as e:
        return {"__error": f"{type(e).__name__}: {e}"}


# Venue normalization → short tag
VENUE_SHORT = {
    "nature": "Nature",
    "nature physics": "Nat. Phys.",
    "nature communications": "Nat. Commun.",
    "nature machine intelligence": "Nat. Mach. Intell.",
    "npj quantum information": "npj QI",
    "npj quantum inf.": "npj QI",
    "physical review x": "PRX",
    "prx quantum": "PRX Quantum",
    "physical review letters": "PRL",
    "physical review a": "PRA",
    "physical review b": "PRB",
    "physical review research": "PRR",
    "physical review applied": "PRApplied",
    "quantum": "Quantum",
    "quantum science and technology": "QST",
    "ieee transactions on quantum engineering": "IEEE TQE",
    "advances in neural information processing systems": "NeurIPS",
    "neurips": "NeurIPS",
    "icml": "ICML",
    "iclr": "ICLR",
    "aaai": "AAAI",
    "scientific reports": "Sci. Rep.",
    "science": "Science",
    "science advances": "Sci. Adv.",
    "communications physics": "Commun. Phys.",
    "ieee international conference on quantum computing and engineering": "IEEE QCE",
    "qce": "IEEE QCE",
}


def normalize_venue(raw: str) -> str:
    if not raw:
        return ""
    low = raw.lower().strip()
    for k, v in VENUE_SHORT.items():
        if k in low:
            return v
    # Default: keep original, trimmed
    return raw.strip()


WORKSHOP_MARKERS = ["workshop", "symposium"]


def classify_oa(oa: dict | None) -> dict:
    """Extract classification from OpenAlex response."""
    if oa is None:
        return {"status": "PREPRINT", "venue_tag": "", "raw_venue": "", "year": None, "doi": "", "volume": "", "pages": "", "notes": "no-oa-match"}
    if "__error" in oa:
        return {"status": "UNKNOWN", "venue_tag": "", "raw_venue": "", "year": None, "doi": "", "volume": "", "pages": "", "notes": oa["__error"]}

    pub_year = oa.get("publication_year")
    doi = oa.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    # Primary location
    primary = oa.get("primary_location") or {}
    source = primary.get("source") or {}
    venue_name = source.get("display_name", "") or ""
    source_type = source.get("type", "") or ""

    # Volume / issue / pages from biblio
    bib = oa.get("biblio") or {}
    volume = bib.get("volume", "") or ""
    issue = bib.get("issue", "") or ""
    first_page = bib.get("first_page", "") or ""
    last_page = bib.get("last_page", "") or ""
    pages = f"{first_page}--{last_page}" if first_page and last_page else first_page

    # Classify
    status = "PREPRINT"
    low_venue = venue_name.lower()

    # arxiv preprint server
    if "arxiv" in low_venue or source_type == "repository":
        status = "PREPRINT"
        venue_name = ""  # don't count arxiv as a venue
    elif any(w in low_venue for w in WORKSHOP_MARKERS):
        status = "WORKSHOP"
    elif venue_name:
        status = "PUBLISHED"

    tag = ""
    if status == "PUBLISHED":
        short = normalize_venue(venue_name)
        tag = f"*{short}* {pub_year}" if pub_year else f"*{short}*"
    elif status == "WORKSHOP":
        short = normalize_venue(venue_name)
        tag = f"{short} {pub_year}" if pub_year else short

    return {
        "status": status,
        "venue_tag": tag,
        "raw_venue": venue_name,
        "year": pub_year,
        "doi": doi,
        "volume": volume,
        "issue": issue,
        "pages": pages,
        "notes": "",
    }


def merge_classifications(old: dict, new_oa: dict) -> dict:
    """Prefer any classification with a concrete venue over one without."""
    # old came from S2 path
    old_venue = (old or {}).get("raw_venue", "") or ""
    new_venue = new_oa.get("raw_venue", "") or ""
    # If old has a venue → keep old but enrich doi/pages/volume from new
    if old_venue and old.get("status") == "PUBLISHED":
        merged = dict(old)
        for field in ("doi", "volume", "issue", "pages"):
            if not merged.get(field) and new_oa.get(field):
                merged[field] = new_oa[field]
        return merged
    # Otherwise new wins if it has a venue
    if new_venue and new_oa.get("status") == "PUBLISHED":
        return new_oa
    # Both empty: use whichever gives more info
    return new_oa if new_oa.get("status") != "UNKNOWN" else old


def main():
    data = json.loads(VENUE_INFO.read_text(encoding="utf-8"))
    targets = []
    for aid, r in data.items():
        cls = r.get("classification") or {}
        s2 = r.get("semantic_scholar") or {}
        # Hit OpenAlex if: S2 errored, OR we currently have PREPRINT with no venue
        s2_ok = "__error" not in s2 and (s2.get("venue") or s2.get("journal"))
        if not s2_ok:
            targets.append(aid)

    print(f"Querying OpenAlex for {len(targets)}/{len(data)} entries", file=sys.stderr)

    for i, aid in enumerate(targets, 1):
        entry = data[aid]
        title = entry.get("title", "") or ""
        year_hint = None
        # Year hint from arxiv ID (YYMM)
        m = re.match(r"^(\d{4})\.", aid)
        if m:
            y = int(m.group(1)[:2])
            year_hint = 2000 + y if y < 50 else 1900 + y
            # Arxiv's 4-digit era
        elif "/" in aid:
            pass

        oa = oa_query_by_arxiv(aid)
        if oa is None or "__error" in (oa or {}):
            # Fallback to title search
            oa = oa_query_by_title(title, year_hint)

        time.sleep(0.15)  # very polite

        entry["openalex"] = oa
        new_cls = classify_oa(oa)
        merged = merge_classifications(entry.get("classification"), new_cls)
        entry["classification"] = merged

        tag = merged.get("venue_tag", "") or ""
        status = merged.get("status", "UNKNOWN")
        print(f"[{i}/{len(targets)}] {aid}: {status} / {tag}", file=sys.stderr)

        if i % 10 == 0:
            VENUE_INFO.write_text(json.dumps(data, indent=2), encoding="utf-8")

    VENUE_INFO.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Rewrite log
    log_lines = []
    for aid, r in data.items():
        cls = r.get("classification") or {}
        log_lines.append(
            f"{aid}: {cls.get('status','?')} / venue='{cls.get('raw_venue','')}' / tag='{cls.get('venue_tag','')}' / year={cls.get('year','?')} / doi={cls.get('doi','')}"
        )
    LOG_FILE.write_text("\n".join(log_lines), encoding="utf-8")

    # Stats
    from collections import Counter
    statuses = Counter(r["classification"]["status"] for r in data.values())
    venues = Counter(r["classification"]["venue_tag"].split(" ")[0].strip("*")
                     for r in data.values() if r["classification"].get("venue_tag"))
    print(file=sys.stderr)
    print("Final status:", dict(statuses), file=sys.stderr)
    print("Top venues:", venues.most_common(10), file=sys.stderr)


if __name__ == "__main__":
    main()
