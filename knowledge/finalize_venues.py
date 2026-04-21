"""
Final cleanup of venue_info.json + regeneration of INDEX.md and bibliography.bib.

Fixes the following issues introduced by S2 + OpenAlex merges:
  1. OpenAlex display_name sometimes contains the full citation — strip it.
  2. S2 occasionally returns "ArXiv" as venue — demote to PREPRINT.
  3. Compact venue tags ("*Nat. Phys.* 2024") are computed from cleaned names.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent.resolve()
VENUE_INFO = HERE / "venue_info.json"
BIB_FILE = HERE / "bibliography.bib"
INDEX_FILE = HERE / "INDEX.md"

VENUE_SHORT = [
    ("nature physics", "Nat. Phys."),
    ("nature communications", "Nat. Commun."),
    ("nature machine intelligence", "Nat. Mach. Intell."),
    ("nature computational science", "Nat. Comput. Sci."),
    ("npj quantum information", "npj QI"),
    ("npj quantum inf", "npj QI"),
    ("nature", "Nature"),  # after the more specific "nature ..." entries
    ("science advances", "Sci. Adv."),
    ("science", "Science"),
    ("physical review x", "PRX"),
    ("prx quantum", "PRX Quantum"),
    ("physical review letters", "PRL"),
    ("phys. rev. lett", "PRL"),
    ("phys rev lett", "PRL"),
    ("physical review research", "PRR"),
    ("phys. rev. research", "PRR"),
    ("phys rev research", "PRR"),
    ("physical review applied", "PRApplied"),
    ("phys. rev. applied", "PRApplied"),
    ("physical review a", "PRA"),
    ("phys. rev. a", "PRA"),
    ("physical review b", "PRB"),
    ("phys. rev. b", "PRB"),
    ("physical review", "PRR"),  # ambiguous, default to Research
    ("phys. rev", "PRR"),
    ("new journal of physics", "NJP"),
    ("new j. phys", "NJP"),
    ("quantum science and technology", "QST"),
    ("machine learning: science and technology", "MLST"),
    ("mach. learn. sci. technol", "MLST"),
    ("communications physics", "Commun. Phys."),
    ("scientific reports", "Sci. Rep."),
    ("ieee transactions on quantum engineering", "IEEE TQE"),
    ("ieee transactions on computers", "IEEE TC"),
    ("ieee international conference on quantum computing and engineering", "IEEE QCE"),
    ("advances in neural information processing systems", "NeurIPS"),
    ("neurips", "NeurIPS"),
    ("icml", "ICML"),
    ("iclr", "ICLR"),
    ("proceedings of the national academy of sciences", "PNAS"),
    ("asia and south pacific design automation conference", "ASP-DAC"),
    ("architectural support for programming languages and operating systems", "ASPLOS"),
    ("allerton conference", "Allerton"),
    ("workshop", "Workshop"),
    ("quantum", "Quantum"),
]


def clean_venue(raw: str) -> str:
    """Strip citation/page/year junk from venue names."""
    if not raw:
        return ""
    s = raw.strip()
    # Remove trailing year in parens: "Phys. Rev. Lett. 122, 200501 (2019)" → "Phys. Rev. Lett. 122, 200501"
    s = re.sub(r"\s*\(\d{4}\)\s*$", "", s).strip()
    # Remove "Published <date>" suffix
    s = re.sub(r",\s*Published [A-Za-z]+ \d+,?\s*\d{4}\s*$", "", s).strip()
    # Strip "<volume> <issue/pages>" pattern: "Phys. Rev. Lett. 122, 200501" → "Phys. Rev. Lett."
    s = re.sub(r"\s+\d+\s*,\s*[A-Z]?\d+.*$", "", s).strip()
    # Strip leading year: "2016 54th Annual Allerton ..." → "54th Annual Allerton ..."
    s = re.sub(r"^\d{4}\s+", "", s).strip()
    return s


def short_tag(clean: str) -> str:
    low = clean.lower()
    for key, short in VENUE_SHORT:
        if key in low:
            return short
    return clean


def is_arxiv_or_repo(raw: str) -> bool:
    low = raw.lower() if raw else ""
    return low in ("arxiv", "arxiv.org", "") or low.startswith("arxiv ")


def finalize():
    data = json.loads(VENUE_INFO.read_text(encoding="utf-8"))

    for aid, r in data.items():
        cls = r.get("classification") or {}
        raw = cls.get("raw_venue") or ""
        cleaned = clean_venue(raw)

        # If after cleaning it's arxiv/empty → demote to PREPRINT
        if not cleaned or is_arxiv_or_repo(cleaned):
            cls["status"] = "PREPRINT"
            cls["venue_tag"] = ""
            cls["raw_venue"] = ""
            r["classification"] = cls
            continue

        # Workshop check
        status = cls.get("status", "PUBLISHED")
        if "workshop" in cleaned.lower() or "workshop" in raw.lower():
            status = "WORKSHOP"
        else:
            status = "PUBLISHED"

        # Prefer arxiv's journal_ref publication year (most accurate), then
        # OpenAlex (explicit publication_year), then S2 (sometimes returns
        # submission year instead of publication year), then fall back to
        # arxiv submission year as a last resort.
        arxiv_meta = r.get("arxiv") or {}
        s2 = r.get("semantic_scholar") or {}
        oa = r.get("openalex") or {}
        ref_year = None
        jref = (arxiv_meta if isinstance(arxiv_meta, dict) else {}).get("journal_ref", "")
        if jref:
            m = re.search(r"\((\d{4})\)", jref)
            if m:
                ref_year = int(m.group(1))
        year = (
            ref_year
            or (oa if isinstance(oa, dict) else {}).get("publication_year")
            or cls.get("year")
            or (s2 if isinstance(s2, dict) else {}).get("year")
            or _extract_year(aid)
        )
        tag = short_tag(cleaned)
        if status == "PUBLISHED":
            venue_tag = f"*{tag}* {year}" if year else f"*{tag}*"
        else:
            venue_tag = f"{tag} {year}" if year else tag

        cls["status"] = status
        cls["raw_venue"] = cleaned
        cls["venue_tag"] = venue_tag
        cls["short_venue"] = tag
        cls["year"] = year
        r["classification"] = cls

    VENUE_INFO.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _extract_year(aid: str) -> int | None:
    # Handle YYMM.xxxxx and YYYY.xxxxx
    m = re.match(r"^(\d+)\.", aid)
    if not m:
        return None
    prefix = m.group(1)
    if len(prefix) == 4:
        yy = int(prefix[:2])
        return 2000 + yy if yy < 50 else 1900 + yy
    return None


def load_existing_bib():
    """Parse existing bib to get titles, authors, keys."""
    if not BIB_FILE.exists():
        return {}
    text = BIB_FILE.read_text(encoding="utf-8")
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, flags=re.DOTALL):
        etype, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*$", body, flags=re.MULTILINE | re.DOTALL):
            fname, fval = fm.group(1).lower(), fm.group(2).strip()
            # Clean nested braces
            fval = fval.replace("{", "").replace("}", "").strip()
            fields[fname] = fval
        fields["_type"] = etype
        fields["_key"] = key
        entries[key] = fields
    return entries


def generate_bib(data, existing):
    """Rewrite bibliography.bib with proper @article / @misc / @inproceedings entries."""
    lines = []
    # Build arxiv_id → citation_key map from existing bib
    aid_to_key = {}
    for key, fields in existing.items():
        eid = (fields.get("eprint") or "").strip()
        if eid:
            aid_to_key[eid] = key

    for aid, r in data.items():
        cls = r["classification"]
        status = cls.get("status", "PREPRINT")
        key = aid_to_key.get(aid)
        if not key:
            # Fallback: use arxiv id
            key = f"arxiv{aid.replace('.','_').replace('/','_')}"

        base = existing.get(key, {})
        title = base.get("title", r.get("title", "") or "Untitled")
        author = base.get("author", "Unknown")
        year = cls.get("year") or base.get("year") or _extract_year(aid)

        if status == "PUBLISHED":
            venue = cls.get("raw_venue", "")
            volume = cls.get("volume", "")
            pages = cls.get("pages", "")
            doi = cls.get("doi", "")
            fields = [
                f"  title = {{{title}}}",
                f"  author = {{{author}}}",
                f"  journal = {{{venue}}}",
            ]
            if volume:
                fields.append(f"  volume = {{{volume}}}")
            if pages:
                fields.append(f"  pages = {{{pages}}}")
            if year:
                fields.append(f"  year = {{{year}}}")
            if doi:
                fields.append(f"  doi = {{{doi}}}")
            fields.append(f"  eprint = {{{aid}}}")
            fields.append(f"  archivePrefix = {{arXiv}}")
            # Conferences use @inproceedings
            lower_venue = venue.lower()
            if any(k in lower_venue for k in ["conference", "proceedings", "neurips", "icml", "iclr", "aaai", "qce", "allerton", "asp-dac", "asplos"]):
                etype = "inproceedings"
                # Rename journal -> booktitle
                fields = [f.replace("journal =", "booktitle =") for f in fields]
            else:
                etype = "article"
            lines.append(f"@{etype}{{{key},\n" + ",\n".join(fields) + "\n}\n")
        else:
            fields = [
                f"  title = {{{title}}}",
                f"  author = {{{author}}}",
            ]
            if year:
                fields.append(f"  year = {{{year}}}")
            fields.append(f"  eprint = {{{aid}}}")
            fields.append(f"  archivePrefix = {{arXiv}}")
            fields.append(f"  note = {{Preprint}}")
            lines.append(f"@misc{{{key},\n" + ",\n".join(fields) + "\n}\n")

    BIB_FILE.write_text("\n".join(lines), encoding="utf-8")


def generate_index(data, existing):
    """Rewrite INDEX.md with venue tags and summary."""
    current = INDEX_FILE.read_text(encoding="utf-8")
    aid_to_key = {(f.get("eprint") or "").strip(): k for k, f in existing.items()}

    # Build aid → classification lookup
    aid_to_cls = {aid: r["classification"] for aid, r in data.items()}

    # Walk the existing INDEX line by line and inject/refresh venue tag
    new_lines = []
    # Accept lines with or without an existing [tag]
    pattern = re.compile(
        r"^(\s*-\s*\*\*(\d{4}-[\w\-]+)\*\*\s*\(arXiv:([\d.]+(?:/\d+)?)\))\s*"
        r"(?:\[[^\]]*\]\s*)?"
        r"(—.*)$"
    )

    for line in current.splitlines():
        m = pattern.match(line)
        if m:
            prefix = m.group(1)
            aid = m.group(3)
            tail = m.group(4)
            cls = aid_to_cls.get(aid)
            if cls:
                tag = cls.get("venue_tag", "").strip()
                status = cls.get("status")
                if status == "PUBLISHED" and tag:
                    venue_part = f"[{tag}]"
                elif status == "WORKSHOP" and tag:
                    venue_part = f"[{tag} workshop]"
                else:
                    venue_part = "[arXiv preprint]"
                line = f"{prefix} {venue_part} {tail}"
        new_lines.append(line)

    body = "\n".join(new_lines)

    # Insert/update the publication-status summary right after "## How to use" block.
    statuses = Counter(r["classification"]["status"] for r in data.values())
    venues = Counter()
    for r in data.values():
        tag = r["classification"].get("short_venue", "")
        if tag and r["classification"]["status"] == "PUBLISHED":
            venues[tag] += 1
    top_venues = venues.most_common(10)

    summary = [
        "## Publication status summary",
        "",
        f"- **PUBLISHED** (peer-reviewed journal/conference): **{statuses.get('PUBLISHED', 0)}** papers",
        f"- **WORKSHOP** / non-archival venue: **{statuses.get('WORKSHOP', 0)}** papers",
        f"- **arXiv preprint only**: **{statuses.get('PREPRINT', 0)}** papers",
        "",
        "**Top venues**: " + ", ".join(f"{v} ({c})" for v, c in top_venues),
        "",
        "Verified via OpenAlex + Semantic Scholar APIs on 2026-04-20. Preprint-only papers are predominantly 2025–2026 arxiv posts that have not yet cycled through peer review; these include many of the most recent SOTA claims and should be read with appropriate skepticism.",
        "",
        "---",
        "",
    ]
    # Strip any previous "## Publication status summary" blocks first
    body = re.sub(
        r"\n## Publication status summary.*?\n---\n",
        "\n",
        body,
        flags=re.DOTALL,
    )

    # Splice summary after the first "---" that follows "## How to use"
    lines = body.splitlines()
    result = []
    inserted = False
    seen_how_to_use = False
    for line in lines:
        result.append(line)
        if line.startswith("## How to use"):
            seen_how_to_use = True
            continue
        if seen_how_to_use and not inserted and line.strip() == "---":
            result.extend([""] + summary)
            inserted = True

    if not inserted:
        result = [summary[0]] + result  # fallback

    INDEX_FILE.write_text("\n".join(result), encoding="utf-8")


def main():
    data = finalize()
    existing = load_existing_bib()
    generate_bib(data, existing)
    generate_index(data, existing)

    statuses = Counter(r["classification"]["status"] for r in data.values())
    venues = Counter()
    for r in data.values():
        tag = r["classification"].get("short_venue", "")
        if tag and r["classification"]["status"] == "PUBLISHED":
            venues[tag] += 1
    print("=" * 60)
    print(f"FINAL STATS: {dict(statuses)}")
    print(f"TOP VENUES: {venues.most_common(15)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
