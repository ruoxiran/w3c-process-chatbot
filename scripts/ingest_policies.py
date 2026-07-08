#!/usr/bin/env python3
"""Incrementally ingest the W3C Policies landing page (depth 2) into the corpus.

The canonical importer (``import_w3c_sources.py``) rewrites the entire
corpus and re-clones repos + re-crawls the 500-page Guidebook. That is
the right tool for a full rebuild, but overkill for adding one source.

This script crawls ONLY ``https://www.w3.org/policies/`` (two levels
deep, reusing the importer's crawl + chunk functions and the same
per-source exclusions), recuts just those chunks with the shared
``recut`` logic, and appends them to ``data/corpus/chunks.jsonl`` in
place. Existing ``repo_name == "policies"`` chunks are dropped first, so
re-running is idempotent — it refreshes the policy content without
touching the other ~6.5k chunks.

    ./.venv/bin/python scripts/ingest_policies.py [--dry-run]

Run ``scripts/build_embedding_cache.py`` afterwards if dense retrieval
is enabled; the lexical index is rebuilt at API startup / via
``POST /refresh-index``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import import_w3c_sources as importer
from recut_chunks import recut


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "chunks.jsonl"
MANIFEST = ROOT / "data" / "corpus" / "manifest.json"
SOURCES_DIR = ROOT / "data" / "sources"

POLICIES_SOURCE = next(s for s in importer.WEB_SOURCES if s["repo_name"] == "policies")


def crawl_policies(indexed_at: str) -> list[dict]:
    """Crawl /policies/ (depth 2) and return recut chunk dicts."""
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    pages = importer.crawl_web_source(
        source=POLICIES_SOURCE,
        sources_dir=SOURCES_DIR,
        max_depth=POLICIES_SOURCE.get("max_depth", 2),
        max_pages=POLICIES_SOURCE.get("max_pages", 120),
        exclude_prefixes=POLICIES_SOURCE.get("exclude_prefixes"),
        fetch_delay=POLICIES_SOURCE.get("fetch_delay", 1.0),
    )
    print(f"crawled {len(pages)} pages under /policies/", flush=True)

    raw: list[dict] = []
    for page in pages:
        chunks = importer.html_chunks(
            html=page["html"],
            title=page["title"],
            url=page["url"],
            source_type=POLICIES_SOURCE["source_type"],
            repo_name=POLICIES_SOURCE["repo_name"],
            repo_url=None,
            commit_sha=None,
            indexed_at=indexed_at,
            page_depth=page["depth"],
            parent_url=page.get("parent_url"),
        )
        raw.extend(asdict(c) for c in chunks)

    print(f"raw chunks before recut: {len(raw)}", flush=True)
    return recut(raw)


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing.")
    args = parser.parse_args()

    if not CORPUS.exists():
        raise SystemExit(f"corpus not found: {CORPUS}  (run import_w3c_sources.py first)")

    indexed_at = datetime.now(timezone.utc).isoformat()
    policy_chunks = crawl_policies(indexed_at)
    print(f"recut policy chunks: {len(policy_chunks)}", flush=True)

    existing = load_jsonl(CORPUS)
    kept = [c for c in existing if c.get("repo_name") != "policies"]
    dropped = len(existing) - len(kept)
    merged = kept + policy_chunks

    print(f"existing: {len(existing)} | dropped old policies: {dropped} "
          f"| new policies: {len(policy_chunks)} | total: {len(merged)}", flush=True)

    # Sample of what was captured, so a dry-run is inspectable.
    pages = sorted({c["source_url"].split("#")[0] for c in policy_chunks})
    print(f"distinct policy pages captured: {len(pages)}", flush=True)
    for page in pages:
        print(f"  {page}", flush=True)

    if args.dry_run:
        print("dry-run: not writing", flush=True)
        return

    backup = CORPUS.with_suffix(CORPUS.suffix + ".prepolicies.bak")
    CORPUS.replace(backup)
    print(f"backup: {backup}", flush=True)
    with CORPUS.open("w", encoding="utf-8") as handle:
        for chunk in merged:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"wrote:  {CORPUS} ({len(merged)} chunks)", flush=True)

    # Keep the manifest honest: register the source and refresh the count.
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["chunk_count"] = len(merged)
        sources = manifest.get("web_sources", [])
        registered = {s.get("repo_name") for s in sources}
        if "policies" not in registered:
            sources.append({
                "name": POLICIES_SOURCE["name"],
                "url": POLICIES_SOURCE["url"],
                "source_type": POLICIES_SOURCE["source_type"],
                "repo_name": POLICIES_SOURCE["repo_name"],
            })
            manifest["web_sources"] = sources
        manifest["policies_ingested_at"] = indexed_at
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated manifest: {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
