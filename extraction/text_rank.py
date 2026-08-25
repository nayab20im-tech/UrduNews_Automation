"""
extraction/text_rank.py
==========================
Shared TextRank sentence-ranking utility used by both keypoint
extraction (Stage 2.4) and summarization (Stage 2.5).

Approach: build a sentence-similarity graph with TF-IDF + cosine
similarity (same toolkit already used in duplicate detection --
consistent, dependency-light, offline, reproducible), then rank
sentences with `networkx`'s PageRank implementation, which is exactly
the TextRank algorithm (Mihalcea & Tarau, 2004) applied to that graph.

No summarization model is downloaded -- this is a practical, resource-
appropriate approach for a local pipeline (Requirement: "prefer a
practical model that can run in the current environment").
"""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_sentences(sentences: List[str]) -> List[Tuple[int, float]]:
    """Return [(sentence_index, score), ...] sorted by descending score.

    Falls back to input order (uniform score) for 0-2 sentences, where a
    similarity graph isn't meaningful.
    """
    n = len(sentences)
    if n == 0:
        return []
    if n <= 2:
        return [(i, 1.0) for i in range(n)]

    try:
        vectorizer = TfidfVectorizer(lowercase=True, min_df=1)
        matrix = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(matrix)
        # Zero the diagonal so a sentence's self-similarity doesn't
        # dominate its own PageRank mass.
        for i in range(n):
            sim_matrix[i, i] = 0.0

        graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(graph, max_iter=200)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked
    except Exception:
        # Degenerate input (e.g. all-identical sentences causing a
        # singular graph) -- fall back to original order.
        return [(i, 1.0) for i in range(n)]


def top_sentences_in_order(sentences: List[str], k: int) -> List[str]:
    """Pick the top-k ranked sentences, but return them in their original
    reading order (summaries should read coherently, not by score)."""
    if not sentences:
        return []
    k = min(k, len(sentences))
    ranked = rank_sentences(sentences)
    top_indices = sorted(idx for idx, _ in ranked[:k])
    return [sentences[i] for i in top_indices]
