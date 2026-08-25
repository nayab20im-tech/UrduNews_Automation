"""
extraction/claim_extractor.py
================================
Stage 2.4 -- Claim / fact identification.

A sentence is treated as a factual claim/statement worth surfacing if it
contains at least one of:
  - a reporting verb ("said", "announced", "according to", ...) --
    who-said-what,
  - a number/statistic (also catches dates and money mentions from the
    NER stage),
  - a quoted attribution.

This never invents text -- every returned claim is a verbatim sentence
already present in the article (Requirement: "Do NOT invent facts that
are not present in the source article").
"""

from __future__ import annotations

import re
from typing import List

_REPORTING_VERBS_EN = re.compile(
    r"\b(said|says|announced|reported|according to|stated|confirmed|declared|"
    r"revealed|claimed|told|warned|added|noted)\b",
    re.IGNORECASE,
)
_REPORTING_VERBS_UR = re.compile(
    r"(کہا|کہتے ہیں|اعلان کیا|کے مطابق|بتایا|تصدیق کی|دعویٰ کیا|خبردار کیا)"
)
_NUMBER_RE = re.compile(r"\d")
_QUOTE_RE = re.compile(r"[\"“”]")

DEFAULT_MAX_CLAIMS = 5


def extract_claims(sentences: List[str], max_claims: int = DEFAULT_MAX_CLAIMS) -> List[str]:
    if not sentences:
        return []

    claims = []
    for sentence in sentences:
        if (
            _REPORTING_VERBS_EN.search(sentence)
            or _REPORTING_VERBS_UR.search(sentence)
            or _NUMBER_RE.search(sentence)
            or _QUOTE_RE.search(sentence)
        ):
            claims.append(sentence)

    return claims[:max_claims]
