from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/api"))

from app.core.config import Settings  # noqa: E402
from app.rag.retriever import chunk_embedding_text, chunk_id  # noqa: E402
from app.services.embeddings import OllamaEmbeddingClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local dense embedding cache for W3C corpus retrieval.")
    parser.add_argument("--corpus", default=None, help="Path to chunks.jsonl")
    parser.add_argument("--output", default=None, help="Path to embedding cache JSONL")
    parser.add_argument("--model", default=None, help="Ollama embedding model, e.g. qwen3-embedding:4b")
    parser.add_argument("--limit", type=int, default=0, help="Optional max chunks for smoke testing")
    parser.add_argument("--resume", action="store_true", help="Reuse existing output entries")
    args = parser.parse_args()

    settings = Settings()
    corpus_path = Path(args.corpus or settings.corpus_path)
    output_path = Path(args.output or settings.retrieval_embedding_cache_path)
    model = args.model or settings.ollama_embedding_model or settings.embedding_model
    client = OllamaEmbeddingClient(settings.ollama_base_url, settings.ollama_timeout_seconds)

    existing = _existing_ids(output_path, model) if args.resume else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    written = 0
    skipped = 0

    with corpus_path.open("r", encoding="utf-8") as corpus, temp_path.open("w", encoding="utf-8") as output:
        if args.resume and output_path.exists():
            with output_path.open("r", encoding="utf-8") as previous:
                for line in previous:
                    if line.strip():
                        output.write(line)

        for line in corpus:
            if args.limit and written >= args.limit:
                break
            if not line.strip():
                continue
            chunk = json.loads(line)
            identifier = chunk_id(chunk)
            if identifier in existing:
                skipped += 1
                continue
            text = chunk_embedding_text(chunk)
            if not text:
                skipped += 1
                continue
            vector = client.embed(model=model, text=text)
            output.write(
                json.dumps(
                    {
                        "chunk_id": identifier,
                        "model": model,
                        "source_url": chunk.get("source_url"),
                        "heading_path": chunk.get("heading_path"),
                        "embedding": vector,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if written % 50 == 0:
                print(f"embedded {written} chunks; skipped {skipped}", flush=True)

    os.replace(temp_path, output_path)
    print(f"wrote {written} new embeddings to {output_path}; skipped {skipped}")


def _existing_ids(path: Path, model: str) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("model") == model and isinstance(payload.get("chunk_id"), str):
                ids.add(str(payload["chunk_id"]))
    return ids


if __name__ == "__main__":
    main()
