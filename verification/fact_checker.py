"""
verification/fact_checker.py
================================
Stage 2.8 -- Fact Verification & Evidence Retrieval (internal sources).

No external fact-check API is used (Requirement 13: no paid services /
API keys; fully offline). "Verification" here means *cross-source
corroboration inside the acquired corpus*:

1. **Evidence retrieval** -- every Stage-2.4 claim of an article is
   compared (TF-IDF cosine, same vectorization approach as Stage 2.3)
   against every *other* article in the corpus; articles above the
   configurable similarity threshold count as supporting evidence.
2. **Source credibility scoring** -- each source has a configurable
   credibility score (`TRUSTED_SOURCES_JSON`, default 0.5 for unknown
   sources); supporting evidence from credible outlets weighs more.
3. **Confidence score** -- per claim:
   ``0.5 * best_similarity + 0.3 * mean_supporting_credibility +
   0.2 * min(1, n_supporting / min_corroborating_sources)``,
   and the article confidence is the mean over its claims.
4. **Verdict** -- a claim seen in >= `min_corroborating_sources`
   *distinct other sources* is "corroborated"; seen in one other
   article "single_source"; otherwise "unverified". Articles carry an
   aggregate verdict (corroborated / partially_corroborated /
   unverified / no_claims).

Like every other stage this only *annotates* -- nothing is deleted or
rewritten -- and Stage 2.9 uses the verdicts to hedge uncorroborated
claims in the broadcast script instead of silently dropping them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_acquisition.utils.logger import get_logger

logger = get_logger("pipeline.verification")

DEFAULT_TRUSTED_SOURCES: Dict[str, float] = {
    "bbc": 0.9, "reuters": 0.9, "ap": 0.9, "associated press": 0.9,
    "afp": 0.85, "aljazeera": 0.8, "cnn": 0.75, "dawn": 0.75,
    "geo news": 0.7, "bbc urdu": 0.9,
}


@dataclass
class VerificationRecord:
    article_id: int
    source_credibility: float = 0.5
    claim_results: List[Dict[str, Any]] = field(default_factory=list)
    evidence_count: int = 0            # total supporting articles across claims
    confidence: float = 0.0            # mean claim confidence, 0..1
    verdict: str = "no_claims"         # corroborated | partially_corroborated | unverified | no_claims
    status: str = "ok"                 # ok | no_claims | error


class FactChecker:
    def __init__(
        self,
        evidence_similarity_threshold: float = 0.15,
        min_corroborating_sources: int = 2,
        trusted_sources: Dict[str, float] | None = None,
        default_credibility: float = 0.5,
    ):
        self.evidence_similarity_threshold = evidence_similarity_threshold
        self.min_corroborating_sources = min_corroborating_sources
        self.trusted_sources = {k.lower(): v for k, v in (trusted_sources or DEFAULT_TRUSTED_SOURCES).items()}
        self.default_credibility = default_credibility

    def source_credibility(self, source: str) -> float:
        return self.trusted_sources.get((source or "").strip().lower(), self.default_credibility)

    # -- public API ---------------------------------------------------------
    def verify(self, articles: List[Dict]) -> List[VerificationRecord]:
        """`articles`: list of {"article_id", "source", "title", "text", "claims"}.

        Corpus-level stage (like 2.3): evidence is a relationship between
        articles, so the similarity matrix is built once per run.
        """
        if not articles:
            return []

        try:
            corpus_texts = [(a.get("title") or "") + " " + (a.get("text") or "") for a in articles]
            usable = [i for i, t in enumerate(corpus_texts) if t.strip()]

            matrix = None
            self._vectorizer = None
            if len(usable) >= 2:
                vectorizer = TfidfVectorizer(lowercase=True, min_df=1, ngram_range=(1, 2))
                matrix = vectorizer.fit_transform([corpus_texts[i] for i in usable])
                self._vectorizer = vectorizer

            results = [
                self._verify_one(a, i, articles, usable, matrix)
                for i, a in enumerate(articles)
            ]
            logger.info(
                "Fact verification complete: %d corroborated, %d partial, %d unverified, %d no-claims (of %d)",
                sum(1 for r in results if r.verdict == "corroborated"),
                sum(1 for r in results if r.verdict == "partially_corroborated"),
                sum(1 for r in results if r.verdict == "unverified"),
                sum(1 for r in results if r.verdict == "no_claims"),
                len(results),
            )
            return results
        except Exception as exc:
            logger.error("Fact verification failed: %s", exc)
            return [VerificationRecord(article_id=a["article_id"], status="error") for a in articles]

    # -- internals -----------------------------------------------------------
    def _verify_one(self, article: Dict, idx: int, articles: List[Dict],
                    usable: List[int], matrix) -> VerificationRecord:
        article_id = article["article_id"]
        credibility = self.source_credibility(article.get("source"))
        claims: List[str] = article.get("claims") or []
        if not claims:
            return VerificationRecord(article_id, credibility, [], 0, 0.0, "no_claims", "no_claims")

        claim_results: List[Dict[str, Any]] = []
        evidence_total = 0

        for claim in claims:
            sims = self._claim_similarities(claim, idx, usable, matrix)
            max_sim = 0.0
            supporting: List[Dict[str, Any]] = []
            if sims:
                max_sim = max(sims.values())
                for other_idx, sim in sims.items():
                    if sim >= self.evidence_similarity_threshold:
                        supporting.append({"article_id": articles[other_idx]["article_id"],
                                           "source": articles[other_idx].get("source") or "unknown",
                                           "similarity": round(sim, 4)})

            supporting_sources = sorted({s["source"] for s in supporting})
            avg_cred = (
                sum(self.source_credibility(s["source"]) for s in supporting) / len(supporting)
                if supporting else 0.0
            )
            confidence = round(min(
                1.0,
                0.5 * max_sim
                + 0.3 * avg_cred
                + 0.2 * min(1.0, len(supporting) / max(self.min_corroborating_sources, 1)),
            ), 4)

            if len(supporting_sources) >= self.min_corroborating_sources:
                verdict = "corroborated"
            elif supporting:
                verdict = "single_source"
            else:
                verdict = "unverified"

            evidence_total += len(supporting)
            claim_results.append(
                {
                    "claim": claim,
                    "max_similarity": round(max_sim, 4),
                    "supporting": supporting,
                    "supporting_sources": supporting_sources,
                    "verdict": verdict,
                    "confidence": confidence,
                }
            )

        article_confidence = round(
            sum(c["confidence"] for c in claim_results) / len(claim_results), 4
        ) if claim_results else 0.0
        verdicts = {c["verdict"] for c in claim_results}
        if verdicts == {"corroborated"}:
            article_verdict = "corroborated"
        elif "corroborated" in verdicts or "single_source" in verdicts:
            article_verdict = "partially_corroborated"
        else:
            article_verdict = "unverified"

        return VerificationRecord(
            article_id, credibility, claim_results, evidence_total,
            article_confidence, article_verdict, "ok",
        )

    def _claim_similarities(self, claim: str, idx: int, usable: List[int], matrix) -> Dict[int, float]:
        """Cosine similarity of one claim against every other usable article."""
        if matrix is None or not (claim or "").strip():
            return {}
        # The vectorizer instance is stashed by verify() for reuse here.
        vectorizer = getattr(self, "_vectorizer", None)
        if vectorizer is None:
            return {}
        try:
            claim_vec = vectorizer.transform([claim])
        except Exception:
            return {}
        sims = cosine_similarity(claim_vec, matrix)[0]
        out: Dict[int, float] = {}
        for local_i, orig_idx in enumerate(usable):
            if orig_idx != idx:
                out[orig_idx] = float(sims[local_i])
        return out
