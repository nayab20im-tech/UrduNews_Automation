"""
translation/lexicon.py
=========================
Bilingual (English -> Urdu) news-domain lexicon used by Stage 2.6.

Three layers, consulted in order by `translator.py`:

1. `PHRASE_MAP`  -- multi-word expressions translated as a unit
   ("prime minister" -> "وزیرِ اعظم"); longest-match-first.
2. `WORD_MAP`    -- single-word vocabulary covering the highest-frequency
   news terms (politics, business, sports, international, ...).
3. `transliterator.transliterate` -- fallback for anything unknown
   (mostly proper nouns).

The lexicon is intentionally a plain dict: it is data, not logic, so a
larger dictionary or a downloaded bilingual glossary can replace it
without touching the translator (same swap-out philosophy the project
uses for the classifier and duplicate detector).

`STYLE_MAP` lives here too because it is also just data: the Urdu
refiner (Stage 2.6 "Style & Readability Enhancement") uses it to lift
colloquial wording to a formal news register.
"""

from __future__ import annotations

# -- multi-word expressions (keys are lower-case English) ----------------
PHRASE_MAP = {
    "prime minister": "وزیرِ اعظم",
    "foreign minister": "وزیرِ خارجہ",
    "interior minister": "وزیرِ داخلہ",
    "chief minister": "وزیرِ اعلیٰ",
    "supreme court": "سپریم کورٹ",
    "high court": "ہائی کورٹ",
    "national assembly": "قومی اسمبلی",
    "united nations": "اقوامِ متحدہ",
    "according to": "کے مطابق",
    "the government": "حکومت",
    "government of": "حکومتِ",
    "human rights": "انسانی حقوق",
    "security forces": "سیکیورٹی فورسز",
    "foreign office": "دفترِ خارجہ",
    "general elections": "عام انتخابات",
    "international community": "بین الاقوامی برادری",
    "economic growth": "معاشی ترقی",
    "climate change": "ماحولیاتی تبدیلی",
    "artificial intelligence": "مصنوعی ذہانت",
    "stock market": "اسٹاک مارکیٹ",
    "cease fire": "جنگ بندی",
    "press conference": "پریس کانفرنس",
    "imran khan": "عمران خان",
    "the united states": "امریکہ",
    "united states": "امریکہ",
    "saudi arabia": "سعودی عرب",
    "new delhi": "نئی دہلی",
}

# -- single words ----------------------------------------------------------
WORD_MAP = {
    # politics / governance
    "government": "حکومت", "president": "صدر", "minister": "وزیر",
    "parliament": "پارلیمنٹ", "senate": "سینیٹ", "election": "انتخاب",
    "elections": "انتخابات", "vote": "ووٹ", "votes": "ووٹ",
    "party": "پارٹی", "opposition": "اپوزیشن", "law": "قانون",
    "court": "عدالت", "judge": "جج", "officials": "حکام",
    "official": "سرکاری", "sources": "ذرائع", "source": "ذریعہ",
    "meeting": "اجلاس", "visit": "دورہ", "talks": "مذاکرات",
    "agreement": "معاہدہ", "deal": "معاہدہ", "policy": "پالیسی",
    "statement": "بيان", "announcement": "اعلان",
    # security / conflict
    "army": "فوج", "military": "فوج", "police": "پولیس", "war": "جنگ",
    "peace": "امن", "security": "سیکیورٹی", "attack": "حملہ",
    "attacks": "حملے", "violence": "تشدد", "protest": "احتجاج",
    "protests": "احتجاج", "arrested": "گرفتار", "arrest": "گرفتاری",
    "killed": "ہلاک", "dead": "ہلاک", "injured": "زخمی",
    # economy / business
    "economy": "معیشت", "economic": "معاشی", "market": "مارکیٹ",
    "trade": "تجارت", "business": "کاروبار", "company": "کمپنی",
    "investment": "سرمایہ کاری", "budget": "بجٹ", "tax": "ٹیکس",
    "taxes": "ٹیکس", "price": "قیمت", "prices": "قیمتیں",
    "inflation": "مہنگائی", "bank": "بینک", "rupee": "روپیہ",
    "dollar": "ڈالر", "million": "ملین", "billion": "بلین",
    # society / general
    "people": "عوام", "citizens": "شہری", "children": "بچے",
    "women": "خواتین", "men": "مرد", "country": "ملک",
    "countries": "ممالک", "world": "دنیا", "news": "خبر",
    "health": "صحت", "hospital": "ہسپتال", "education": "تعلیم",
    "school": "اسکول", "university": "یونیورسٹی", "weather": "موسم",
    "rain": "بارش", "flood": "سیلاب", "earthquake": "زلزلہ",
    "crisis": "بحران", "problem": "مسئلہ", "problems": "مسائل",
    # sports
    "cricket": "کرکٹ", "football": "فٹبال", "match": "میچ",
    "team": "ٹیم", "player": "کھلاڑی", "players": "کھلاڑی",
    "sports": "کھیل", "championship": "چیمپئن شپ", "series": "سیریز",
    # technology
    "technology": "ٹیکنالوجی", "internet": "انٹرنیٹ", "phone": "فون",
    "app": "ایپ", "data": "ڈیٹا", "software": "سافٹ ویئر",
    # verbs / reporting
    "said": "کہا", "says": "کہا", "told": "بتایا", "announced": "اعلان کیا",
    "announces": "اعلان کیا", "warned": "خبردار کیا", "warns": "خبردار کیا",
    "confirmed": "تصدیق کی", "confirms": "تصدیق کی", "reported": "رپورٹ کیا",
    "claimed": "دعویٰ کیا", "stated": "بیان دیا", "visited": "دورہ کیا",
    "visits": "دورہ کیا", "won": "جیت لیا", "wins": "جیت",
    # common adverbs / adjectives
    "today": "آج", "yesterday": "گزشتہ کل", "tomorrow": "کل",
    "week": "ہفتہ", "month": "مہینہ", "year": "سال", "new": "نیا",
    "old": "پرانا", "big": "بڑا", "small": "چھوٹا", "important": "اہم",
    "major": "بڑا", "national": "قومی", "international": "بین الاقوامی",
    "local": "مقامی", "political": "سیاسی", "latest": "تازہ ترین",
    # countries / places (conventional Urdu spellings)
    "pakistan": "پاکستان", "india": "بھارت", "china": "چین",
    "america": "امریکہ", "usa": "امریکہ", "uk": "برطانیہ",
    "britain": "برطانیہ", "russia": "روس", "afghanistan": "افغانستان",
    "iran": "ایران", "israel": "اسرائیل", "turkey": "ترکی",
    "kashmir": "کشمیر", "punjab": "پنجاب", "sindh": "سندھ",
    "balochistan": "بلوچستان", "islamabad": "اسلام آباد",
    "karachi": "کراچی", "lahore": "لاہور", "peshawar": "پشاور",
    "quetta": "کوئٹہ", "delhi": "دہلی", "kabul": "کابل",
    "taliban": "طالبان",
}

# -- English function words: None => dropped, else mapped ------------------
# Articles and copula/auxiliary noise are dropped (they have no useful
# Urdu equivalent in a gloss translation); conjunctions, prepositions,
# and pronouns map to their common Urdu counterparts.
FUNCTION_MAP = {
    "the": None, "a": None, "an": None, "of": None, "to": None,
    "is": None, "are": None, "was": None, "were": None, "be": None,
    "been": None, "being": None, "has": None, "have": None, "had": None,
    "will": None, "would": None, "could": None, "should": None,
    "may": None, "might": None, "must": None, "do": None, "does": None,
    "did": None, "its": None, "it's": None, "that's": None,
    "and": "اور", "or": "یا", "but": "لیکن", "because": "کیونکہ",
    "if": "اگر", "while": "جبکہ", "after": "کے بعد", "before": "سے پہلے",
    "during": "کے دوران", "from": "سے", "with": "کے ساتھ",
    "for": "کے لیے", "in": "میں", "on": "پر", "at": "پر",
    "by": "کی طرف سے", "about": "کے بارے میں", "over": "پر",
    "under": "کے تحت", "between": "کے درمیان", "against": "کے خلاف",
    "into": "میں", "through": "کے ذریعے", "not": "نہیں", "no": "کوئی",
    "all": "تمام", "more": "مزید", "also": "بھی", "still": "ابھی بھی",
    "now": "اب", "soon": "جلد", "later": "بعد میں", "where": "جہاں",
    "when": "جب", "why": "کیوں", "how": "کیسے", "what": "کیا",
    "who": "کون", "which": "جو", "that": "کہ", "this": "یہ",
    "these": "یہ", "those": "وہ", "there": "وہاں", "here": "یہاں",
    "he": "انہوں", "she": "انہوں", "they": "انہوں", "it": "یہ",
    "his": "ان کے", "her": "ان کے", "their": "ان کے", "him": "انہیں",
    "them": "انہیں", "we": "ہم", "our": "ہمارے", "you": "آپ",
    "your": "آپ کے", "i": "میں", "me": "مجھے", "my": "میرا",
    "us": "ہمیں",
}

# -- Urdu register lift: colloquial -> formal news style -------------------
STYLE_MAP = {
    "بولا": "کہا",
    "بولے": "کہا",
    "پکڑا": "گرفتار کیا",
    "پکڑے": "گرفتار کیے",
    "مر گیا": "ہلاک ہو گیا",
    "مر گئے": "ہلاک ہو گئے",
}

# Sorted phrase keys, longest first, so multi-word matches win over the
# single-word map (e.g. "prime minister" must not become "وزیرِ" + ...).
SORTED_PHRASES = sorted(PHRASE_MAP.keys(), key=lambda p: -len(p.split()))
