"""Persistent short-term, long-term, and vector memory."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from agent.config.settings import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryItem:
    """A remembered conversation or observation."""

    role: str
    content: str
    created_at: str


class HashEmbedding:
    """Dependency-light deterministic embeddings used as a reliable fallback."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                vectors[row, hash(token) % self.dimensions] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors


class VectorMemory:
    """Small persistent vector store with optional FAISS acceleration."""

    def __init__(self, path: Path, embedding_model: str) -> None:
        self.path = path
        self.embedding_model = embedding_model
        self.items: list[MemoryItem] = []
        self.vectors = np.empty((0, 384), dtype=np.float32)
        self._encoder: Any | None = None
        self._faiss: Any | None = None
        self._index: Any | None = None

    async def initialize(self) -> None:
        """Load vectors and initialize embedding backend."""

        self.path.mkdir(parents=True, exist_ok=True)
        self._load_encoder()
        self._load()

    def _load_encoder(self) -> None:
        if self.embedding_model.lower() == "hash":
            self._encoder = HashEmbedding()
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
            LOGGER.info("Loaded sentence-transformers embedding model.")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Using hash embeddings: %s", exc)
            self._encoder = HashEmbedding()
        try:
            import faiss

            self._faiss = faiss
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("FAISS unavailable; using numpy similarity: %s", exc)

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            encoded = self._encoder.encode(texts, show_progress_bar=False)
        except TypeError:
            encoded = self._encoder.encode(texts)
        if not isinstance(encoded, np.ndarray):
            encoded = np.asarray(encoded, dtype=np.float32)
        return encoded.astype(np.float32)

    def _load(self) -> None:
        items_path = self.path / "items.jsonl"
        vectors_path = self.path / "vectors.npy"
        if items_path.exists():
            self.items = [
                MemoryItem(**json.loads(line))
                for line in items_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if vectors_path.exists():
            self.vectors = np.load(vectors_path)
        self._rebuild_index()

    def _save(self) -> None:
        items_path = self.path / "items.jsonl"
        items_path.write_text(
            "\n".join(json.dumps(asdict(item)) for item in self.items),
            encoding="utf-8",
        )
        np.save(self.path / "vectors.npy", self.vectors)

    def _rebuild_index(self) -> None:
        if self._faiss is None or self.vectors.size == 0:
            self._index = None
            return
        self._index = self._faiss.IndexFlatIP(self.vectors.shape[1])
        self._index.add(self.vectors)

    def add(self, item: MemoryItem) -> None:
        """Add a memory item and persist it."""

        vector = self._embed([item.content])
        self.items.append(item)
        self.vectors = np.vstack([self.vectors, vector])
        self._rebuild_index()
        self._save()

    def search(self, query: str, k: int = 5) -> list[MemoryItem]:
        """Return semantically similar memories."""

        if not self.items or self.vectors.size == 0:
            return []
        query_vector = self._embed([query])
        if self._index is not None:
            _, idx = self._index.search(query_vector, min(k, len(self.items)))
            return [self.items[i] for i in idx[0] if i >= 0]
        scores = self.vectors @ query_vector[0]
        indexes = np.argsort(scores)[::-1][:k]
        return [self.items[int(i)] for i in indexes]


class MemoryManager:
    """Coordinates short-term context and persistent long-term memory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.short_term: list[MemoryItem] = []
        self.long_term_path = settings.memory_dir / "long_term.jsonl"
        self.vector_memory = VectorMemory(settings.memory_dir / "vectors", settings.embedding_model)

    async def initialize(self) -> None:
        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        await self.vector_memory.initialize()

    def add(self, role: str, content: str, persist: bool = True) -> MemoryItem:
        """Add a memory entry."""

        item = MemoryItem(
            role=role,
            content=content,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.short_term.append(item)
        self.short_term = self.short_term[-self.settings.short_term_limit :]
        if persist:
            with self.long_term_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(item)) + "\n")
            self.vector_memory.add(item)
        return item

    def load_long_term(self) -> list[MemoryItem]:
        """Load all long-term memories from disk."""

        if not self.long_term_path.exists():
            return []
        return [
            MemoryItem(**json.loads(line))
            for line in self.long_term_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def relevant_context(self, query: str, k: int = 5) -> str:
        """Build a compact context block from relevant memories."""

        memories = self.vector_memory.search(query, k=k)
        if not memories:
            return ""
        return "\n".join(f"{item.role}: {item.content}" for item in memories)
