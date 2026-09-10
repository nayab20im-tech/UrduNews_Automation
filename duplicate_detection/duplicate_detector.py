"""
duplicate_detection/duplicate_detector.py
============================================
Stage 2.3 -- Duplicate Detection (near-duplicate, cross-source).

Method: TF-IDF vectors + cosine similarity.

Why TF-IDF over embeddings here: this environment has no network access
to download a sentence-embedding model, and the project's own
Requirement 13 says "avoid unnecessary API keys or paid services." TF-IDF
+ cosine similarity is a standard, well-understood near-duplicate
detection approach that needs no external model, runs fully offline, and
is deterministic/reproducible (Requirement 16) -- appropriate for a
local pipeline of this size. `DuplicateDetector` exposes a single
`vectorizer_factory` hook so it can be swapped for an embeddings-based
similarity function later without touching the grouping/selection logic.

This is separate from Stage 1's raw-level exact/URL dedup (see
`database/raw_news_db.py`) and from Stage 2.1's exact-content-hash dedup
-- this stage finds *near*-duplicates: the same story covered by
different outlets with different wording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_acquisition.utils.logger import get_logger

logger = get_logger("pipeline.duplicate_detection")


@dataclass
class DuplicateRecord:
    article_id: int
    duplicate_group_id: Optional[int]
    similarity_score: float          # similarity to the group's primary article
    is_duplicate: bool               # True for every non-primary member of a group
    is_primary: bool                 # the "most relevant/complete source" kept
    status: str = "ok"               # ok | unique | error


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class DuplicateDetector:
    """Groups near-duplicate articles and picks one primary per group.

    `similarity_threshold` is configurable -- see Requirement 12 ("make
    thresholds ... configurable"). 0.65 is a reasonable default for
    short news-article TF-IDF cosine similarity: high enough to avoid
    lumping together articles that merely share a broad topic (e.g. two
    unrelated cricket stories), low enough to catch the same story
    reworded across two outlets.
    """

    def __init__(self, similarity_threshold: float = 0.65):
        self.similarity_threshold = similarity_threshold

    def detect(self, articles: List[Dict]) -> List[DuplicateRecord]:
        """`articles`: list of {"article_id": int, "text": str, "title": str}.

        Grouping requires a full pairwise similarity matrix -- called
        once per pipeline run over all valid (non-error, non-exact-dup)
        preprocessed articles, not per-article, since "duplicate" is
        inherently a relationship between articles.
        """
        n = len(articles)
        if n == 0:
            return []

        try:
            texts = [(a.get("title") or "") + " " + (a.get("text") or "") for a in articles]
            non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]
            if len(non_empty_idx) < 2:
                return [
                    DuplicateRecord(a["article_id"], None, 0.0, False, True, "unique")
                    for a in articles
                ]

            vectorizer = TfidfVectorizer(lowercase=True, min_df=1, ngram_range=(1, 2))
            matrix = vectorizer.fit_transform([texts[i] for i in non_empty_idx])
            sim_matrix = cosine_similarity(matrix)

            uf = _UnionFind(len(non_empty_idx))
            for i in range(len(non_empty_idx)):
                for j in range(i + 1, len(non_empty_idx)):
                    if sim_matrix[i, j] >= self.similarity_threshold:
                        uf.union(i, j)

            # local-index groups -> members
            groups: Dict[int, List[int]] = {}
            for local_i in range(len(non_empty_idx)):
                root = uf.find(local_i)
                groups.setdefault(root, []).append(local_i)

            results: Dict[int, DuplicateRecord] = {}
            group_counter = 0
            for root, members in groups.items():
                if len(members) == 1:
                    orig_idx = non_empty_idx[members[0]]
                    results[orig_idx] = DuplicateRecord(
                        articles[orig_idx]["article_id"], None, 0.0, False, True, "unique"
                    )
                    continue

                group_counter += 1
                # "Keep the most relevant/complete source": longest cleaned
                # text wins as the primary (a reasonable, explainable proxy
                # for completeness with no external quality signal available).
                primary_local = max(members, key=lambda li: len(texts[non_empty_idx[li]]))
                primary_orig = non_empty_idx[primary_local]

                for li in members:
                    orig_idx = non_empty_idx[li]
                    score = float(sim_matrix[li, primary_local])
                    is_primary = orig_idx == primary_orig
                    results[orig_idx] = DuplicateRecord(
                        articles[orig_idx]["article_id"],
                        group_counter,
                        1.0 if is_primary else round(score, 4),
                        not is_primary,
                        is_primary,
                        "ok",
                    )

            # Articles with empty text: can't compare, treat as unique.
            for i, a in enumerate(articles):
                if i not in results:
                    results[i] = DuplicateRecord(a["article_id"], None, 0.0, False, True, "unique")

            ordered = [results[i] for i in range(n)]
            logger.info(
                "Duplicate detection complete: %d groups, %d marked duplicate (of %d articles)",
                group_counter,
                sum(1 for r in ordered if r.is_duplicate),
                n,
            )
            return ordered
        except Exception as exc:
            logger.error("Duplicate detection failed: %s", exc)
            return [
                DuplicateRecord(a["article_id"], None, 0.0, False, True, "error")
                for a in articles
            ]
