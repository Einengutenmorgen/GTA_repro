# utils.py
"""
Shared embedding utilities: single source of truth for turning text into
L2-normalized vectors with a persistent per-text disk cache.

Used by both the codebook-alignment layer (alignment.py) and the
question-similarity index (question_sim.py) so every embedding in the project
comes from the same model and shares one on-disk cache. The cache is keyed by
(model_name, sha256(text)), so different text families (human codes, LLM codes,
interview questions) coexist without collision and reruns never re-embed.

Keep this module dependency-light: numpy always; sentence-transformers only
imported lazily inside embed_texts, so cache-only reruns never load the model.

Dependencies: numpy; sentence-transformers (lazy).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import warnings
from typing import Dict, List, Tuple

import numpy as np

# Single source of truth for the embedding model + cache location.
# Any module that embeds MUST use these (or be passed them) so results are
# comparable across the alignment and question-similarity layers.
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBED_CACHE = ".embedding_cache.pkl"


def text_key(model_name: str, text: str) -> Tuple[str, str]:
    """Cache key for one text under one model: (model, sha256 hexdigest)."""
    return (model_name, hashlib.sha256(text.encode("utf-8")).hexdigest())


def load_disk_cache(path: str) -> Dict[Tuple[str, str], np.ndarray]:
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:  # corrupt cache is not fatal, just recompute
            warnings.warn(f"Could not read embedding cache {path!r} ({e}); recomputing.")
    return {}


def save_disk_cache(path: str, cache: Dict[Tuple[str, str], np.ndarray]) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp, path)


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    disk_cache_path: str = DEFAULT_EMBED_CACHE,
) -> np.ndarray:
    """Embed texts, L2-normalized, with a persistent per-text disk cache.

    Returns an (N, d) float32 array of unit-norm row vectors, aligned to
    `texts`. Empty input returns a (0, 0) array. Only uncached texts hit the
    model; the model is imported lazily so a fully-cached call needs no GPU
    and no sentence-transformers load.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    disk = load_disk_cache(disk_cache_path)
    missing_idx = [i for i, t in enumerate(texts)
                   if text_key(model_name, t) not in disk]

    if missing_idx:
        # Import lazily so cache-only reruns never touch the model.
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        new_vecs = model.encode(
            [texts[i] for i in missing_idx],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        for i, vec in zip(missing_idx, new_vecs):
            disk[text_key(model_name, texts[i])] = vec
        save_disk_cache(disk_cache_path, disk)

    vecs = np.stack([disk[text_key(model_name, t)] for t in texts]).astype(np.float32)
    # Re-normalize defensively (cached vectors are already normalized).
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms
