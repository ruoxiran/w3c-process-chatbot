from __future__ import annotations

import httpx


class OllamaEmbeddingClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed(self, *, model: str, text: str) -> list[float]:
        payload = self._post_embed(model=model, text=text)
        embedding = _extract_embedding(payload)
        if embedding:
            return _normalize(embedding)

        payload = self._post_legacy_embedding(model=model, text=text)
        embedding = _extract_embedding(payload)
        if not embedding:
            raise ValueError("Ollama embedding response did not include an embedding vector")
        return _normalize(embedding)

    def _post_embed(self, *, model: str, text: str) -> dict[str, object]:
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": text},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def _post_legacy_embedding(self, *, model: str, text: str) -> dict[str, object]:
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}


def _extract_embedding(payload: dict[str, object]) -> list[float]:
    direct = payload.get("embedding")
    if isinstance(direct, list):
        return [float(value) for value in direct if isinstance(value, (int, float))]

    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, list):
            return [float(value) for value in first if isinstance(value, (int, float))]
    return []


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if not norm:
        return vector
    return [value / norm for value in vector]
