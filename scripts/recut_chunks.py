#!/usr/bin/env python3
"""Recut data/corpus/chunks.jsonl: coalesce small paragraphs, drop snapshot noise.

The original ingestion (``import_w3c_sources.py``) emits one chunk per
``<p>`` / ``<li>`` element. That yields 5879 chunks with a median size
of 207 chars — many too small to carry useful context for dense
retrieval. This script post-processes that JSONL into a smaller,
denser corpus by:

  1. Dropping chunks from ``snapshots/*.html`` (historical Process
     drafts; non-authoritative + carry inline script/style noise).
  2. Coalescing consecutive chunks that share ``(source_url-without-
     fragment, heading_path)`` into ~800-char windows (max ~1500).
  3. Generating deterministic IDs from the merged content hash so
     downstream embedding caches can be rebuilt incrementally.

The original file is moved aside to ``chunks.jsonl.bak``; the new file
is written to ``chunks.jsonl``. Re-run with ``--dry-run`` to inspect
the result without overwriting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "corpus" / "chunks.jsonl"

TARGET_CHARS = 800
MAX_CHARS = 1500
MIN_CHARS = 80  # singletons below this are dropped (boilerplate)

# Drop two categories of non-authoritative noise that the importer
# greedily swept up:
#   - ``snapshots/*`` — historical drafts of the Process Document. They
#     are explicitly superseded by the current Process Document and
#     also leak inline <script>/<style> blocks into the chunk text.
#   - ``issues-*.{txt,html}`` — period-end issue dumps. Not normative;
#     each one is a 10-30KB blob that crowds out useful citations.
NOISE_MARKERS = ("/snapshots/", "snapshots/", "/issues-", "issues-")


def is_noise_chunk(chunk: dict) -> bool:
    file_path = (chunk.get("file_path") or "").lower()
    source_url = (chunk.get("source_url") or "").lower()
    return any(m in file_path for m in NOISE_MARKERS) or any(
        m in source_url for m in NOISE_MARKERS
    )


def looks_like_inline_script(text: str) -> bool:
    """Catch leftover script/style residue that slipped past the importer."""
    if len(text) < 200:
        return False
    code_signals = ("function(", " => ", "var ", "const ", "addEventListener",
                    "document.", "window.", ".prototype.", "}{", "::before",
                    "::after", "@keyframes", "px;")
    hits = sum(text.count(sig) for sig in code_signals)
    return hits >= 4


def group_key(chunk: dict) -> tuple[str, str]:
    """Two chunks coalesce iff same heading section on the same page."""
    raw_url = chunk.get("source_url") or ""
    page, _ = urldefrag(raw_url)
    return page, chunk.get("heading_path") or ""


def coalesce_group(group: list[dict]) -> Iterable[dict]:
    """Emit windowed chunks of TARGET_CHARS, never exceeding MAX_CHARS."""
    if not group:
        return
    buf_text: list[str] = []
    buf_len = 0
    head = group[0]

    def flush() -> dict | None:
        nonlocal buf_text, buf_len
        if not buf_text:
            return None
        text = " ".join(buf_text)
        if len(text) < MIN_CHARS:
            buf_text = []
            buf_len = 0
            return None
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        out = {
            **head,
            "id": f"{head.get('repo_name', 'corpus')}:recut:{digest}",
            "text": text,
        }
        buf_text = []
        buf_len = 0
        return out

    def feed(text: str) -> Iterable[dict]:
        nonlocal buf_text, buf_len
        added_len = len(text) + (1 if buf_text else 0)
        if buf_len and buf_len + added_len > MAX_CHARS:
            out = flush()
            if out:
                yield out
        buf_text.append(text)
        buf_len += len(text) + (1 if len(buf_text) > 1 else 0)
        if buf_len >= TARGET_CHARS:
            out = flush()
            if out:
                yield out

    for chunk in group:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        # Split oversized singletons (e.g. a single <pre> block) at word
        # boundaries before feeding the coalescer, so MAX_CHARS is a hard
        # ceiling rather than an aspiration.
        if len(text) > MAX_CHARS:
            words = text.split(" ")
            slice_text = ""
            for word in words:
                if slice_text and len(slice_text) + 1 + len(word) > MAX_CHARS:
                    yield from feed(slice_text)
                    slice_text = word
                else:
                    slice_text = f"{slice_text} {word}".strip() if slice_text else word
            if slice_text:
                yield from feed(slice_text)
        else:
            yield from feed(text)

    out = flush()
    if out:
        yield out


def recut(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in records:
        if is_noise_chunk(chunk):
            continue
        if looks_like_inline_script(chunk.get("text") or ""):
            continue
        key = group_key(chunk)
        if key not in seen:
            seen.add(key)
            order.append(key)
        groups[key].append(chunk)

    out: list[dict] = []
    for key in order:
        out.extend(coalesce_group(groups[key]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src: Path = args.corpus
    if not src.exists():
        raise SystemExit(f"corpus not found: {src}")

    records: list[dict] = []
    with src.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    recut_records = recut(records)

    sizes_in = [len(r.get("text") or "") for r in records]
    sizes_out = [len(r.get("text") or "") for r in recut_records]
    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0  # noqa: E731
    print(f"input:  {len(records):>5} chunks | median {median(sizes_in):>5}c | max {max(sizes_in):>7}c")
    print(f"output: {len(recut_records):>5} chunks | median {median(sizes_out):>5}c | max {max(sizes_out):>7}c")
    print(f"<200c:  in {sum(1 for s in sizes_in if s < 200):>5}  out {sum(1 for s in sizes_out if s < 200):>5}")
    print(f"<400c:  in {sum(1 for s in sizes_in if s < 400):>5}  out {sum(1 for s in sizes_out if s < 400):>5}")

    if args.dry_run:
        print("dry-run: not writing")
        return

    bak = src.with_suffix(src.suffix + ".bak")
    src.replace(bak)
    print(f"backup: {bak}")

    with src.open("w") as f:
        for rec in recut_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote:  {src}")


if __name__ == "__main__":
    main()
