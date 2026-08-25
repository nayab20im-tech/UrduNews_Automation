"""
extraction/keypoint_extractor.py
===================================
Stage 2.4 -- Keypoint extraction.

Ranks an article's sentences with TextRank (see `text_rank.py`) and
returns the top-N as the article's keypoints, in original reading order.
Genuinely important sentences tend to be well-connected to the rest of
the article's content, which is exactly what TextRank's graph centrality
measures -- a lightweight, model-free proxy for "importance."
"""

from __future__ import annotations

from typing import List

from .text_rank import top_sentences_in_order

DEFAULT_KEYPOINT_COUNT = 3


def extract_keypoints(sentences: List[str], max_points: int = DEFAULT_KEYPOINT_COUNT) -> List[str]:
    if not sentences:
        return []
    return top_sentences_in_order(sentences, max_points)
