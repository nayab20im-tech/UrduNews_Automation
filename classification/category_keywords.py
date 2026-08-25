"""
classification/category_keywords.py
======================================
Bilingual (English + Urdu) keyword lexicons for the 7 required news
categories. Used to build a TF-IDF "profile document" per category so
articles can be classified by similarity without a labeled training set
or a downloaded model (see classifier.py for why).
"""

from __future__ import annotations

CATEGORIES = [
    "Politics",
    "Sports",
    "Business",
    "Technology",
    "Entertainment",
    "International",
    "Others",
]

CATEGORY_KEYWORDS = {
    "Politics": [
        "election", "elections", "government", "minister", "parliament", "president",
        "prime minister", "senate", "assembly", "policy", "vote", "voting", "party",
        "opposition", "cabinet", "constitution", "law", "bill", "legislation", "campaign",
        "governor", "coalition", "referendum", "political",
        "حکومت", "وزیر", "پارلیمنٹ", "صدر", "وزیراعظم", "سینیٹ", "اسمبلی", "پالیسی",
        "ووٹ", "انتخابات", "پارٹی", "اپوزیشن", "کابینہ", "آئین", "قانون", "بل", "سیاسی",
    ],
    "Sports": [
        "match", "tournament", "cricket", "football", "hockey", "olympics", "league",
        "player", "team", "coach", "goal", "wicket", "score", "championship", "final",
        "athlete", "stadium", "world cup", "series", "innings", "medal", "sport", "sports",
        "میچ", "کرکٹ", "فٹبال", "ہاکی", "اولمپکس", "لیگ", "کھلاڑی", "ٹیم", "کوچ", "گول",
        "وکٹ", "اسکور", "چیمپئن شپ", "فائنل", "کھیل", "ورلڈ کپ", "سیریز",
    ],
    "Business": [
        "market", "stock", "economy", "economic", "trade", "investment", "investor",
        "company", "companies", "bank", "finance", "financial", "revenue", "profit",
        "inflation", "gdp", "budget", "tax", "export", "import", "currency", "rupee",
        "dollar", "startup", "industry", "shares", "business",
        "مارکیٹ", "اسٹاک", "معیشت", "اقتصادی", "تجارت", "سرمایہ کاری", "کمپنی", "بینک",
        "مالیاتی", "منافع", "مہنگائی", "بجٹ", "ٹیکس", "برآمد", "درآمد", "روپیہ", "ڈالر",
        "کاروبار",
    ],
    "Technology": [
        "technology", "tech", "software", "app", "internet", "ai", "artificial intelligence",
        "smartphone", "computer", "startup", "google", "microsoft", "apple", "meta",
        "chip", "semiconductor", "robot", "robotics", "cybersecurity", "data", "cloud",
        "gadget", "innovation", "digital", "5g",
        "ٹیکنالوجی", "سافٹ ویئر", "ایپ", "انٹرنیٹ", "مصنوعی ذہانت", "اسمارٹ فون", "کمپیوٹر",
        "ڈیجیٹل", "روبوٹ", "ڈیٹا", "کلاؤڈ",
    ],
    "Entertainment": [
        "movie", "film", "actor", "actress", "music", "song", "album", "concert", "drama",
        "celebrity", "director", "hollywood", "bollywood", "lollywood", "tv show",
        "television", "singer", "award", "festival", "entertainment", "series", "cinema",
        "فلم", "اداکار", "اداکارہ", "موسیقی", "گانا", "البم", "کنسرٹ", "ڈرامہ", "سیلیبرٹی",
        "ہدایتکار", "ٹیلی ویژن", "گلوکار", "ایوارڈ", "میلہ", "تفریح", "سنیما",
    ],
    "International": [
        "world", "global", "united nations", "un ", "foreign", "diplomat", "diplomatic",
        "embassy", "treaty", "border", "war", "conflict", "ceasefire", "sanctions",
        "united states", "china", "india", "russia", "europe", "middle east", "international",
        "دنیا", "عالمی", "اقوام متحدہ", "غیر ملکی", "سفارتی", "سفارت خانہ", "معاہدہ", "سرحد",
        "جنگ", "تنازعہ", "پابندیاں", "امریکہ", "چین", "بھارت", "روس", "یورپ",
    ],
    "Others": [
        "weather", "health", "education", "culture", "religion", "science", "environment",
        "climate", "accident", "obituary", "lifestyle", "opinion", "editorial",
        "موسم", "صحت", "تعلیم", "ثقافت", "مذہب", "سائنس", "ماحول", "حادثہ",
    ],
}
