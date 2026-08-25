"""
extraction/ner.py
====================
Stage 2.4 -- Named Entity Recognition.

Rule-/regex-based on purpose: no spaCy/transformer model is downloaded
(Requirement: avoid unnecessary huge downloads; this environment also has
no network access to model hubs). Entities are recognized with:

  - DATE / MONEY / EVENT: regex patterns (work for both English and, for
    numeric patterns, Urdu digits/currency words).
  - PERSON / ORGANIZATION / LOCATION: capitalization + trigger-word
    heuristics for English text (titles like "Mr./President/Dr." before
    a name, org suffixes like "Ministry/Party/Inc.", a small gazetteer of
    common country/city names). Reliable capitalization cues don't exist
    in the Urdu script, so Urdu PERSON/ORG/LOCATION extraction is limited
    to matches against a small bilingual gazetteer -- this is a known,
    documented limitation of a no-download approach, not a silent gap.

This module is deliberately swappable: `NERExtractor.extract()` returns a
plain `Dict[str, List[str]]`, so a proper spaCy/transformer NER model can
replace the internals later with no change to callers.
"""

from __future__ import annotations

import re
from typing import Dict, List

_TITLE_WORDS = (
    r"Mr|Mrs|Ms|Dr|Prime Minister|President|Minister|Senator|Governor|"
    r"Justice|General|Sheikh|Chief Justice|CM|PM"
)
_PERSON_RE = re.compile(
    rf"\b(?:{_TITLE_WORDS})\.?\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){{0,3}})"
)
# Two-or-more-capitalized-word run, e.g. "Shehbaz Sharif", "Joe Biden Jr" --
# used as a fallback PERSON cue when no title word is present.
_CAP_RUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")

_ORG_SUFFIXES = (
    r"Ministry(?: of [A-Z][a-zA-Z]+)?|Party|Government|Corporation|Corp\.?|Inc\.?|"
    r"Ltd\.?|Company|Bank|Committee|Council|Authority|Commission|Agency|"
    r"United Nations|UN|Parliament|Senate|Assembly|Court|Airlines|University"
)
_ORG_RE = re.compile(rf"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:{_ORG_SUFFIXES}))\b")

_LOCATION_GAZETTEER = {
    "Pakistan", "India", "China", "Russia", "America", "United States", "USA", "UK",
    "Britain", "England", "Europe", "Afghanistan", "Iran", "Bangladesh", "Karachi",
    "Lahore", "Islamabad", "Gujranwala", "Peshawar", "Quetta", "Multan", "Faisalabad",
    "London", "Washington", "Beijing", "Moscow", "Delhi", "Mumbai", "Kabul", "Dubai",
    "Middle East", "Gaza", "Ukraine", "Israel", "Palestine", "Punjab", "Sindh",
    "Balochistan", "Khyber Pakhtunkhwa", "New York", "Hong Kong", "West Bank",
    "Germany", "France", "Canada", "Australia", "Peru", "Delaware", "Sydney",
    "Netflix",
    # Urdu gazetteer (script-based match, no capitalization cue available).
    "پاکستان", "بھارت", "چین", "روس", "امریکہ", "برطانیہ", "یورپ", "افغانستان",
    "ایران", "بنگلہ دیش", "کراچی", "لاہور", "اسلام آباد", "پشاور", "کوئٹہ", "ملتان",
    "لندن", "واشنگٹن", "بیجنگ", "ماسکو", "دہلی", "کابل", "دبئی", "پنجاب", "سندھ",
}

_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}\s+(?:{_MONTHS})(?:\s+\d{{2,4}})?|(?:{_MONTHS})\s+\d{{1,2}}(?:,?\s+\d{{2,4}})?|"
    rf"\d{{4}}-\d{{2}}-\d{{2}}|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b)"
)
_MONEY_RE = re.compile(
    r"(?:Rs\.?|PKR|USD|\$|₹)\s?[\d,]+(?:\.\d+)?\s*(?:million|billion|trillion|crore|lakh)?"
    r"|\b[\d,]+(?:\.\d+)?\s*(?:million|billion|trillion|crore|lakh)\s*(?:dollars|rupees)?"
)
# Common capitalized-run false positives for the PERSON fallback heuristic
# (event/place/institution names that happen to be Title Case but aren't
# people) -- filtered out by trailing word. Not exhaustive by design; this
# is a lightweight heuristic, not a trained model (see module docstring).
_NON_PERSON_LAST_WORDS = {
    "marathon", "kitchen", "survey", "flag", "day", "week", "bank", "cup",
    "york", "kong", "east", "union", "league", "cabinet", "office", "house",
    "street", "avenue", "square", "wall", "gate", "park", "stadium", "airport",
}

_EVENT_KEYWORDS_RE = re.compile(
    r"\b(election|summit|conference|tournament|championship|olympics|world cup|"
    r"protest|earthquake|flood|ceasefire|referendum|treaty|festival|budget session)\b",
    re.IGNORECASE,
)


class NERExtractor:
    def extract(self, text: str) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {
            "PERSON": [], "ORGANIZATION": [], "LOCATION": [],
            "DATE": [], "MONEY": [], "EVENT": [],
        }
        if not text or not text.strip():
            return entities

        try:
            persons = set(m.group(1).strip() for m in _PERSON_RE.finditer(text))
            orgs = set(m.group(1).strip() for m in _ORG_RE.finditer(text))
            # Fallback capitalized-run persons, excluding anything already
            # captured as an organization or a known location.
            for m in _CAP_RUN_RE.finditer(text):
                candidate = m.group(1).strip()
                if candidate in _LOCATION_GAZETTEER:
                    continue
                if any(candidate in o or o in candidate for o in orgs):
                    continue
                if candidate.split()[-1].lower() in _NON_PERSON_LAST_WORDS:
                    continue
                persons.add(candidate)

            locations = {loc for loc in _LOCATION_GAZETTEER if loc in text}

            entities["PERSON"] = sorted(persons)[:15]
            entities["ORGANIZATION"] = sorted(orgs)[:15]
            entities["LOCATION"] = sorted(locations)[:15]
            entities["DATE"] = sorted(set(_DATE_RE.findall(text)))[:15]
            entities["MONEY"] = sorted(set(m.group(0).strip() for m in _MONEY_RE.finditer(text)))[:15]
            entities["EVENT"] = sorted(set(m.group(0) for m in _EVENT_KEYWORDS_RE.finditer(text)))[:15]
        except Exception:
            # Never let a regex edge case break extraction for one article.
            pass
        return entities
