"""
preprocess.py - Step 1 of the pipeline

Cleans raw 20 Newsgroups documents and outputs a JSONL file.

Design decisions (justified here as per spec):
- Strip ALL routing headers (Path, Message-ID, Date, etc.) — zero semantic value,
  they describe the network journey of the post, not its content.
- Keep Subject: field and prepend it to the body — subject lines are strong
  topical signals, especially for short posts.
- Strip quoted reply lines (> prefix) — they contain the PREVIOUS poster's words,
  not the current author's. Including them would blur document-level semantics.
- Strip signatures (after -- separator) — personal catchphrases, ASCII art, and
  contact info add noise without topical signal.
- Strip "In article ... writes:" threading intros — pure boilerplate.
- Strip email addresses from body — identity markers, not topic markers.
- Drop docs with < 20 words after cleaning — too short to produce a meaningful
  embedding; they would cluster as noise.
- Drop docs where > 70% of lines are quoted — after quote stripping, these become
  near-empty shells with no original voice.
- Store all_newsgroups from the Newsgroups: header — cross-post membership is
  genuine dual-topic signal and will validate fuzzy cluster boundaries in Part 2.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
RAW_DIR      = BASE_DIR / "data" / "raw" / "20_newsgroups"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE  = PROCESSED_DIR / "cleaned_docs.jsonl"

# ── Header fields to strip completely ─────────────────────────────────────────
# These fields describe routing, identity, and threading — not topic content.
STRIP_HEADERS = {
    "path", "message-id", "date", "lines", "from", "sender",
    "xref", "nntp-posting-host", "references", "distribution",
    "x-newsreader", "approved", "organization", "reply-to",
    "followup-to", "archive-name", "alt-atheism-archive-name",
    "last-modified", "version", "expires", "supersedes",
    "summary", "x-posted-to", "posting-frequency",
}

# ── Regex patterns ─────────────────────────────────────────────────────────────
# Quoted lines: lines starting with one or more ">" characters
RE_QUOTED         = re.compile(r"^\s*>+.*$", re.MULTILINE)

# "In article <id>, user writes:" — threading attribution boilerplate
RE_IN_ARTICLE     = re.compile(
    r"^In\s+article\s+.*?(?:writes?|wrote)\s*:.*$",
    re.MULTILINE | re.IGNORECASE
)

# "Someone (email) wrote:" or "Someone wrote:" attribution lines
RE_ATTRIBUTION    = re.compile(
    r"^[\w\s,.<>@\-()]+(?:wrote|writes)\s*:\s*$",
    re.MULTILINE | re.IGNORECASE
)

# Email addresses — identity noise in the body
RE_EMAIL          = re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[a-zA-Z]{2,})+\b")

# PGP / encoded blocks — binary-encoded content, meaningless for NLP
RE_PGP            = re.compile(
    r"-----BEGIN PGP.*?-----END PGP[^-]*-----",
    re.DOTALL
)

# Signature block separator (USENET convention)
RE_SIG_SEP        = re.compile(r"\n--\s*\n")

# Lines that are pure ASCII art / dividers (dashes, equals, stars only)
RE_ASCII_DIVIDER  = re.compile(r"^[\s\-=*#~+_|]{5,}$", re.MULTILINE)

# Collapse 3+ blank lines into 2
RE_EXCESS_BLANK   = re.compile(r"\n{3,}")

# Strip Re:/RE:/Fwd: prefixes from subject lines
RE_SUBJECT_PREFIX = re.compile(r"^\s*(?:Re|RE|Fwd|FWD|AW)\s*:\s*", re.IGNORECASE)


def parse_headers(raw: str) -> Tuple[Dict, str]:
    """
    Split a raw document into (headers_dict, body_text).
    The header block ends at the first blank line.
    """
    # Split at first blank line
    parts = re.split(r"\n\n", raw, maxsplit=1)
    header_block = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    headers = {}
    current_key = None

    for line in header_block.split("\n"):
        # Continuation line (starts with whitespace)
        if line and line[0] in (" ", "\t") and current_key:
            headers[current_key] = headers[current_key] + " " + line.strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            current_key = key.strip().lower()
            headers[current_key] = value.strip()

    return headers, body


def clean_body(body: str) -> str:
    """
    Apply all noise-removal steps to the raw body text.
    Order matters — signature stripping must come before quote stripping.
    """
    # 1. Strip PGP / uuencoded blocks first (before any line-level ops)
    body = RE_PGP.sub("", body)

    # 2. Truncate at signature separator
    #    Everything after "-- " is personal/off-topic
    sig_match = RE_SIG_SEP.search(body)
    if sig_match:
        body = body[: sig_match.start()]

    # 3. Strip "In article ... writes:" threading intros
    body = RE_IN_ARTICLE.sub("", body)

    # 4. Strip attribution lines before quoted blocks
    body = RE_ATTRIBUTION.sub("", body)

    # 5. Strip all quoted lines (> prefix)
    body = RE_QUOTED.sub("", body)

    # 6. Strip email addresses
    body = RE_EMAIL.sub("", body)

    # 7. Strip ASCII art dividers
    body = RE_ASCII_DIVIDER.sub("", body)

    # 8. Collapse excess blank lines
    body = RE_EXCESS_BLANK.sub("\n\n", body)

    return body.strip()


def get_subject(headers: dict) -> Optional[str]:
    """
    Extract and clean the Subject header.
    Strip Re:/Fwd: prefixes — these indicate thread position, not topic.
    """
    subject = headers.get("subject", "").strip()
    if not subject:
        return None
    subject = RE_SUBJECT_PREFIX.sub("", subject).strip()
    # Strip leftover angle brackets and quotes
    subject = subject.strip("<>\"'")
    return subject if subject else None


def get_newsgroups(headers: dict) -> List[str]:
    """
    Parse the Newsgroups: header into a list.
    Cross-post membership is genuine multi-topic signal.
    """
    ng_raw = headers.get("newsgroups", "")
    groups = [g.strip() for g in ng_raw.split(",") if g.strip()]
    return groups


def quote_ratio(body: str) -> float:
    """
    Fraction of non-empty lines that start with >.
    Used to decide if a document is mostly someone else's words.
    """
    lines = [l for l in body.split("\n") if l.strip()]
    if not lines:
        return 1.0
    quoted = sum(1 for l in lines if l.strip().startswith(">"))
    return quoted / len(lines)


def process_document(filepath: Path, category: str) -> Optional[dict]:
    """
    Full processing pipeline for a single document file.
    Returns a dict ready for JSONL output, or None if the doc should be dropped.
    """
    try:
        raw = filepath.read_text(errors="replace")
    except Exception as e:
        log.warning(f"Could not read {filepath}: {e}")
        return None

    # ── Check quote ratio BEFORE cleaning (on raw body) ──────────────────────
    # We want to drop docs that are overwhelmingly quotes from others,
    # since after stripping they'll have almost no original content.
    _, raw_body = parse_headers(raw)
    if quote_ratio(raw_body) > 0.70:
        return None  # Drop: no original voice

    # ── Parse headers ─────────────────────────────────────────────────────────
    headers, raw_body = parse_headers(raw)

    # ── Get metadata ──────────────────────────────────────────────────────────
    subject = get_subject(headers)
    newsgroups = get_newsgroups(headers)

    # ── Clean body ────────────────────────────────────────────────────────────
    cleaned_body = clean_body(raw_body)

    # ── Build final text: Subject prepended to cleaned body ───────────────────
    # Short posts rely heavily on the subject for topical signal.
    if subject:
        full_text = f"{subject}\n\n{cleaned_body}".strip()
    else:
        full_text = cleaned_body

    # ── Length filter ─────────────────────────────────────────────────────────
    # < 20 words → too short for a meaningful embedding
    word_count = len(full_text.split())
    if word_count < 20:
        return None  # Drop: unembeddable

    return {
        "doc_id":           filepath.name,
        "category":         category,
        "all_newsgroups":   newsgroups,
        "is_cross_post":    len(newsgroups) > 1,
        "subject":          subject,
        "text":             full_text,
        "word_count":       word_count,
    }


def run():
    """
    Main entry point.
    Iterates over all 20 category folders, processes every document,
    and writes surviving docs to cleaned_docs.jsonl.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DIR}\n"
            "Please unpack 20_newsgroups.tar.gz into data/raw/"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    categories = sorted([
        d.name for d in RAW_DIR.iterdir() if d.is_dir()
    ])
    log.info(f"Found {len(categories)} categories in {RAW_DIR}")

    total = 0
    kept  = 0
    dropped_short   = 0
    dropped_quoted  = 0
    dropped_error   = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for category in categories:
            cat_dir   = RAW_DIR / category
            doc_files = sorted(cat_dir.iterdir())
            cat_kept  = 0

            for filepath in doc_files:
                if not filepath.is_file():
                    continue
                total += 1

                result = process_document(filepath, category)

                if result is None:
                    # Distinguish drop reason for stats
                    try:
                        raw = filepath.read_text(errors="replace")
                        _, raw_body = parse_headers(raw)
                        if quote_ratio(raw_body) > 0.70:
                            dropped_quoted += 1
                        else:
                            dropped_short += 1
                    except Exception:
                        dropped_error += 1
                    continue

                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                kept     += 1
                cat_kept += 1

            log.info(f"  {category}: {cat_kept} docs kept")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Total documents processed : {total}")
    log.info(f"Kept (written to JSONL)   : {kept}  ({100*kept//total}%)")
    log.info(f"Dropped — too short       : {dropped_short}")
    log.info(f"Dropped — mostly quoted   : {dropped_quoted}")
    log.info(f"Dropped — read error      : {dropped_error}")
    log.info(f"Output → {OUTPUT_FILE}")


if __name__ == "__main__":
    run()