# app/utils/retriever.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
GUIDE_DIR = BASE_DIR / "rag" / "guidelines"
INDEX_DIR = BASE_DIR / "rag" / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking params (SAFE)
MAX_WORDS = 300
OVERLAP = 30
BATCH_SIZE = 16

SECTION_WEIGHTS = {
    "emergency_signs": 0.65,
    "red_flags": 0.80,
    "overview": 1.00,
    "diagnosis": 1.05,
    "self_care": 1.25,
    "home_care": 1.25,
}


class Retriever:
    def __init__(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(EMBED_MODEL)
        self.index: faiss.IndexFlatL2 | None = None
        self.meta: List[Dict[str, Any]] = []

        if INDEX_PATH.exists() and META_PATH.exists():
            self._load()
        else:
            self._build()

    # ---------- Chunk helpers ----------

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk = words[i : i + MAX_WORDS]
            chunks.append(" ".join(chunk))
            i += MAX_WORDS - OVERLAP

        return chunks

    def _split_sections(self, text: str) -> Dict[str, str]:
        """
        Lightweight medical section splitter.
        """
        sections = {}
        current = "overview"
        buffer = []

        for line in text.splitlines():
            l = line.strip().lower()

            if l.startswith(("emergency", "danger", "call", "seek")):
                sections[current] = "\n".join(buffer)
                current = "emergency_signs"
                buffer = []
            elif l.startswith(("symptom", "sign")):
                sections[current] = "\n".join(buffer)
                current = "symptoms"
                buffer = []
            elif l.startswith(("treatment", "care", "management")):
                sections[current] = "\n".join(buffer)
                current = "care"
                buffer = []
            else:
                buffer.append(line)

        sections[current] = "\n".join(buffer)
        return sections

    # ---------- Build index ----------

    def _build(self) -> None:
        texts: List[str] = []
        meta: List[Dict[str, Any]] = []

        for path in sorted(GUIDE_DIR.glob("*.txt")):
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                continue

            sections = self._split_sections(raw)

            for section, section_text in sections.items():
                for idx, chunk in enumerate(self._chunk_text(section_text)):
                    chunk_id = f"{path.stem}:{section}:{idx}"
                    chunk_text = chunk
                    condition, section, _ = chunk_id.split(":", 2)
                    meta.append(
                        {
                            "doc_id": path.name,
                            "chunk_id": chunk_id,
                            "condition": condition,
                            "section": section,
                            "text": chunk_text,
                        }
                    )
                    texts.append(chunk)

        if not texts:
            raise RuntimeError("No guideline text found")

        print(f"[Retriever] Chunked into {len(texts)} chunks")

        # ---- Embed in batches (CRITICAL) ----
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            emb = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(emb)

        embeddings = np.vstack(embeddings).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, str(INDEX_PATH))
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self.index = index
        self.meta = meta

        print(f"[Retriever] Built FAISS index with {len(meta)} chunks")

    def _load(self) -> None:
        self.index = faiss.read_index(str(INDEX_PATH))
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        print(f"[Retriever] Loaded index with {len(self.meta)} chunks")

    # ---------- Query ----------

    def retrieve(self, query: str, k: int = 3, fetch_k: int = 12) -> List[Dict[str, Any]]:
        """
        Return top-k most relevant guideline chunks.
        Applies:
          - semantic similarity
          - section weighting
          - condition-aware prioritization
        """
        if not query.strip() or self.index is None:
            return []

        # Embed query
        q_emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        q_emb = q_emb.astype("float32")

        # Retrieve more candidates than needed
        D, I = self.index.search(q_emb, fetch_k)

        candidates = []

        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta):
                continue

            meta = self.meta[idx]

            section = meta.get("section", "overview")
            condition = meta.get("condition", "unknown")

            # Section-based weighting
            section_weight = SECTION_WEIGHTS.get(section, 1.0)
            adjusted_score = float(score) * section_weight

            candidates.append({
                "score": adjusted_score,
                "raw_score": float(score),
                "condition": condition,
                "section": section,
                "doc_id": meta["doc_id"],
                "chunk_id": meta["chunk_id"],
                "text": meta["text"],
            })

        if not candidates:
            return []

        # --- CONDITION-AWARE RE-RANKING ---

        # Infer dominant condition from top semantic hit
        dominant_condition = candidates[0]["condition"]

        def final_rank_key(item):
            # Same condition gets highest priority
            same_condition_penalty = 0 if item["condition"] == dominant_condition else 1

            # Emergency sections get a bonus
            emergency_bonus = 0 if item["section"] == "emergency_signs" else 0.25

            return (
                same_condition_penalty,          # primary
                emergency_bonus,                 # secondary
                item["score"],                   # semantic score
            )

        candidates.sort(key=final_rank_key)

        return candidates[:k]


if __name__ == "__main__":
    r = Retriever()
    q = "I hit my head and now I am vomiting"
    hits = r.retrieve(q)
    for h in hits:
        print(h["chunk_id"], h["score"])