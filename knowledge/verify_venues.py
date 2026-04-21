"""
Verify publication status for all 81 arxiv papers in the AutoQEC knowledge base.

Queries two sources:
  1. Semantic Scholar Graph API (primary) — venue, journal, DOI
  2. arXiv API (secondary) — arxiv:journal_ref element

Writes:
  - venue_info.json   : structured per-paper verification result
  - venue_check.log   : human-readable trace

Re-run safely; results are cached by arxiv_id in venue_info.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).parent.resolve()
BIB_PATH = HERE / "bibliography.bib"
OUT_JSON = HERE / "venue_info.json"
OUT_LOG = HERE / "venue_check.log"

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{}"
S2_FIELDS = "title,year,venue,journal,publicationVenue,publicationTypes,externalIds,authors"
ARXIV_BASE = "https://export.arxiv.org/api/query?id_list={}"

USER_AGENT = "AutoQEC-venue-verifier/1.0 (mailto:jiahanchen@proton.me)"

# Venue name normalizer — map raw venue strings to compact tags for INDEX.md.
VENUE_TAG_MAP = [
    (r"Nature Communications", "Nat. Commun."),
    (r"Nature Electronics", "Nat. Electron."),
    (r"Nature Physics", "Nat. Phys."),
    (r"\bNature\b", "Nature"),
    (r"npj Quantum Information", "npj QI"),
    (r"PRX Quantum", "PRX Quantum"),
    (r"Physical Review Letters", "PRL"),
    (r"Physical Review Research", "PRR"),
    (r"Physical Review A\b", "PRA"),
    (r"Physical Review B\b", "PRB"),
    (r"Physical Review X\b", "PRX"),
    (r"\bQuantum\b", "Quantum"),
    (r"IEEE Transactions on Quantum Engineering", "IEEE TQE"),
    (r"IEEE.*International Conference on Quantum Computing and Engineering", "IEEE QCE"),
    (r"\bNeurIPS\b|Neural Information Processing Systems", "NeurIPS"),
    (r"\bICML\b|International Conference on Machine Learning", "ICML"),
    (r"\bICLR\b", "ICLR"),
    (r"\bAAAI\b", "AAAI"),
    (r"New Journal of Physics", "New J. Phys."),
    (r"Scientific Reports", "Sci. Rep."),
    (r"Quantum Science and Technology", "Quantum Sci. Technol."),
]


def compact_venue(raw: str) -> str:
    if not raw:
        return ""
    for pat, tag in VENUE_TAG_MAP:
        if re.search(pat, raw, re.IGNORECASE):
            return tag
    # Fallback: use raw
    return raw.strip()


def extract_arxiv_ids(bib_text: str) -> list[str]:
    # All eprint = {xxxx.xxxxx}
    return re.findall(r"eprint\s*=\s*\{([\d]{4}\.[\d]{4,5})\}", bib_text)


def s2_query(arxiv_id: str) -> dict | None:
    url = S2_BASE.format(arxiv_id) + f"?fields={S2_FIELDS}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            # Back-off
            time.sleep(3)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e2:
                return {"__error": f"429 retry failed: {e2}"}
        if e.code == 404:
            return {"__error": "404 not found"}
        return {"__error": f"HTTPError {e.code}: {e.reason}"}
    except Exception as e:
        return {"__error": f"{type(e).__name__}: {e}"}


ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def arxiv_query(arxiv_id: str) -> dict | None:
    url = ARXIV_BASE.format(arxiv_id)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        return {"__error": f"{type(e).__name__}: {e}"}
    try:
        root = ET.fromstring(text)
        entry = root.find("atom:entry", ATOM_NS)
        if entry is None:
            return {"__error": "no entry"}
        jr = entry.find("arxiv:journal_ref", ATOM_NS)
        doi = entry.find("arxiv:doi", ATOM_NS)
        title = entry.find("atom:title", ATOM_NS)
        published = entry.find("atom:published", ATOM_NS)
        return {
            "journal_ref": jr.text.strip() if jr is not None and jr.text else None,
            "doi": doi.text.strip() if doi is not None and doi.text else None,
            "title": title.text.strip() if title is not None and title.text else None,
            "published": published.text.strip() if published is not None and published.text else None,
        }
    except ET.ParseError as e:
        return {"__error": f"xml parse error: {e}"}


def classify(s2: dict | None, arx: dict | None) -> dict:
    """Return dict with keys: status, venue, year, doi, raw_venue, notes."""
    notes = []
    venue = None
    year = None
    doi = None
    raw_venue = None
    status = "PREPRINT"

    if s2 and "__error" not in s2:
        raw_venue = s2.get("venue") or ""
        pv = s2.get("publicationVenue") or {}
        pv_name = pv.get("name") if isinstance(pv, dict) else None
        journal = s2.get("journal") or {}
        j_name = journal.get("name") if isinstance(journal, dict) else None
        # Prefer journal/publicationVenue specificity
        best = j_name or pv_name or raw_venue
        if best:
            raw_venue = best
        year = s2.get("year")
        ext = s2.get("externalIds") or {}
        if isinstance(ext, dict):
            doi = ext.get("DOI")
        pub_types = s2.get("publicationTypes") or []
        if pub_types is None:
            pub_types = []

        # Heuristic: arxiv-only vs. journal
        is_preprint_only = False
        if pv and isinstance(pv, dict):
            pv_type = pv.get("type", "")
            if pv_type and pv_type.lower() == "preprint":
                is_preprint_only = True
        if raw_venue and re.search(r"\barxiv\b|preprint", raw_venue, re.IGNORECASE):
            is_preprint_only = True

        if raw_venue and not is_preprint_only:
            status = "PUBLISHED"
            if re.search(r"workshop", raw_venue, re.IGNORECASE):
                status = "WORKSHOP"
        elif "JournalArticle" in pub_types and raw_venue:
            status = "PUBLISHED"
        else:
            notes.append("S2: no venue info")

    else:
        if s2 and "__error" in s2:
            notes.append(f"S2 error: {s2['__error']}")
        else:
            notes.append("S2: no response")

    # Cross-check with arXiv journal_ref
    if arx and "__error" not in arx:
        jr = arx.get("journal_ref")
        if jr:
            if status == "PREPRINT":
                status = "PUBLISHED"
                raw_venue = jr
                notes.append("arXiv journal_ref used (S2 had none)")
            else:
                # Check consistency
                if raw_venue and jr and raw_venue.lower()[:6] != jr.lower()[:6]:
                    notes.append(f"disagreement: S2='{raw_venue}' arxiv='{jr}'")
        if not doi and arx.get("doi"):
            doi = arx.get("doi")

    venue = compact_venue(raw_venue or "")

    return {
        "status": status,
        "raw_venue": raw_venue or "",
        "venue_tag": venue,
        "year": year,
        "doi": doi,
        "notes": notes,
    }


def main():
    bib_text = BIB_PATH.read_text(encoding="utf-8")
    arxiv_ids = extract_arxiv_ids(bib_text)
    # Preserve order but unique
    seen = set()
    unique_ids = []
    for a in arxiv_ids:
        if a not in seen:
            seen.add(a)
            unique_ids.append(a)
    print(f"Found {len(unique_ids)} unique arxiv IDs", file=sys.stderr)

    # Load cache if exists
    cache = {}
    if OUT_JSON.exists():
        try:
            cache = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    log_lines = []
    results = {}
    for i, aid in enumerate(unique_ids, 1):
        if aid in cache and cache[aid].get("classification", {}).get("status"):
            results[aid] = cache[aid]
            log_lines.append(f"[{i}/{len(unique_ids)}] {aid}: (cached) {cache[aid]['classification']['status']} / {cache[aid]['classification']['venue_tag']}")
            print(log_lines[-1], file=sys.stderr)
            continue

        print(f"[{i}/{len(unique_ids)}] querying {aid}...", file=sys.stderr)
        s2 = s2_query(aid)
        time.sleep(1.2)  # rate limit
        arx = arxiv_query(aid)
        time.sleep(0.5)
        cls = classify(s2, arx)
        results[aid] = {
            "arxiv_id": aid,
            "semantic_scholar": s2,
            "arxiv": arx,
            "classification": cls,
        }
        log_lines.append(
            f"[{i}/{len(unique_ids)}] {aid}: {cls['status']} / venue='{cls['raw_venue']}' / tag='{cls['venue_tag']}' / year={cls['year']} / doi={cls['doi']} / notes={cls['notes']}"
        )
        print(log_lines[-1], file=sys.stderr)

        # Save incrementally
        if i % 5 == 0:
            OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    OUT_LOG.write_text("\n".join(log_lines), encoding="utf-8")

    # Summary
    buckets = {"PUBLISHED": 0, "PREPRINT": 0, "WORKSHOP": 0, "ACCEPTED": 0}
    venues = {}
    for r in results.values():
        s = r["classification"]["status"]
        buckets[s] = buckets.get(s, 0) + 1
        tag = r["classification"]["venue_tag"]
        if s == "PUBLISHED" and tag:
            venues[tag] = venues.get(tag, 0) + 1
    print("\n=== SUMMARY ===", file=sys.stderr)
    for k, v in buckets.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print("  Top venues:", file=sys.stderr)
    for tag, n in sorted(venues.items(), key=lambda x: -x[1])[:10]:
        print(f"    {tag}: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
