"""
bias_detection/bias_lexicon.py
=================================
Bilingual lexicons and patterns for Stage 2.7 (Bias & Sensational
Language Detection). Pure data, same convention as
`classification/category_keywords.py` -- the detector logic lives in
`bias_detector.py`.

The neutral-replacement maps are also consumed by Stage 2.9 (Final
Urdu News Script Generation) to tone sensational wording down to a
professional news register before it reaches the broadcast script.
"""

from __future__ import annotations

import re

# -- clickbait headline patterns (English) ----------------------------------
CLICKBAIT_PATTERNS_EN = [
    re.compile(r"\byou (won'?t|will not) believe\b", re.IGNORECASE),
    re.compile(r"\bwhat happened next\b", re.IGNORECASE),
    re.compile(r"\bthis (one|simple) trick\b", re.IGNORECASE),
    re.compile(r"\b\d+ (ways|reasons|things|secrets) (to|why|you)\b", re.IGNORECASE),
    re.compile(r"\bthe (truth|real reason|secret) about\b", re.IGNORECASE),
    re.compile(r"\b(shocking|unbelievable|mind-?blowing|jaw-?dropping)\b", re.IGNORECASE),
    re.compile(r"\bgoes viral\b", re.IGNORECASE),
    re.compile(r"\bwatch (this|now)\b", re.IGNORECASE),
]

# -- clickbait headline patterns (Urdu) --------------------------------------
CLICKBAIT_PATTERNS_UR = [
    re.compile(r"(سنسنی خیز|دھماکہ خیز انکشاف)"),
    re.compile(r"(جان کر آپ کے ہوش اڑ جائیں گے)"),
    re.compile(r"(ویڈیو دیکھئے|ویڈیو دیکھیں اور)"),
]

# -- sensational / hype vocabulary -------------------------------------------
SENSATIONAL_WORDS_EN = frozenset(
    """
    shocking explosive sensational unbelievable incredible outrageous
    horrifying horrific bombshell nightmare brutal savage epic insane
    madness fury rampage chaos turmoil catastrophe disaster doomed
    slams destroys destroys humiliates obliterates stuns baffles
    """.split()
)
SENSATIONAL_WORDS_UR = frozenset(
    """
    سنسنی خیز دھماکہ خیز خوفناک ہولناک حیران کن سنسنی فضیحت
    تہلکہ خیز خوف ناک خوفناک ترین افسانوی
    """.split()
)

# -- loaded / bias-indicating wording ------------------------------------------
LOADED_WORDS_EN = frozenset(
    """
    regime so-called radical extremist fanatic corrupt crooked
    dishonest sinister ruthless merciless traitor conspiracy plot
    obviously clearly of course undoubtedly certainly shameless
    """.split()
)
LOADED_WORDS_UR = frozenset(
    """
    سازش بدنام غنڈے کرپٹ بدعنوان ظالم سازشی خفیہ منصوبہ
    """.split()
)

# -- sentiment lexicons (small, high-precision) ---------------------------------
POSITIVE_WORDS_EN = frozenset(
    """
    win wins won victory success successful growth peace celebrate
    celebrated progress improvement improved record triumph agreement
    recovery development hope
    """.split()
)
POSITIVE_WORDS_UR = frozenset(
    """
    کامیابی جیت خوشی امن ترقی بہتری امید جشن ریکارڈ معاہدہ
    """.split()
)
NEGATIVE_WORDS_EN = frozenset(
    """
    kill killed kills death dead attack attacks crisis fear violence
    protest protests arrest arrested war conflict tension threat
    warning damage destroyed injured collapse fail failed failure
    """.split()
)
NEGATIVE_WORDS_UR = frozenset(
    """
    قتل ہلاک حملہ بحران خوف تشدد گرفتار جنگ تنازع کشیدگی دھمکی
    تباہ زخمی ناکام
    """.split()
)

# -- sensational -> neutral replacements (professional news register) -----------
NEUTRAL_REPLACEMENTS_EN = {
    "shocking": "غیر متوقع",          # unexpected
    "explosive": "اہم",               # significant
    "sensational": "متنازع",          # controversial
    "unbelievable": "غیر متوقع",
    "incredible": "غیر معمولی",
    "outrageous": "متنازع",
    "horrifying": "تشویشناک",
    "bombshell": "اہم اعلان",
    "nightmare": "مشکل صورت حال",
    "brutal": "شدید",
    "savage": "شدید",
    "slams": "تنقید کرتا ہے",
    "destroys": "تنقید کا نشانہ بناتا ہے",
    "humiliates": "تنقید کا نشانہ بناتا ہے",
    "viral": "وسیع پیمانے پر شیئر",
}
NEUTRAL_REPLACEMENTS_UR = {
    "دھماکہ خیز": "اہم",
    "سنسنی خیز": "متنازع",
    "خوفناک": "شدید",
    "ہولناک": "شدید",
    "حیران کن": "غیر متوقع",
    "تہلکہ خیز": "اہم",
    "فضیحت": "تنقید",
}

# -- English hype word -> plain English stand-in (pre-translation toning) -----
EN_TONE_DOWN = {
    "shocking": "unexpected", "explosive": "significant", "sensational": "controversial",
    "unbelievable": "unexpected", "incredible": "remarkable", "outrageous": "controversial",
    "horrifying": "disturbing", "bombshell": "major", "nightmare": "difficult",
    "brutal": "severe", "savage": "severe", "slams": "criticizes",
    "destroys": "criticizes", "humiliates": "criticizes", "viral": "widely shared",
}
