"""
preprocessing/stopwords.py
===========================
Small, dependency-free stopword lists for English and Urdu.

No NLTK/spaCy download is used on purpose (Requirement: avoid unnecessary
downloads / heavy models -- see task instructions). These lists are
intentionally short and focused on the highest-frequency function words;
good enough for stopword-removal used as an NLP *feature* input, never
applied to the text kept for summarization.
"""

from __future__ import annotations

ENGLISH_STOPWORDS = frozenset(
    """
    a an the and or but if while is are was were be been being
    to of in on at by for with about against between into through
    during before after above below from up down out off over under
    again further then once here there when where why how all any
    both each few more most other some such no nor not only own same
    so than too very s t can will just don should now i me my myself
    we our ours ourselves you your yours yourself yourselves he him
    his himself she her hers herself it its itself they them their
    theirs themselves what which who whom this that these those am
    have has had having do does did doing would could might must
    shall as it's that's have's it says said also new one two three
    """.split()
)

# Common Urdu function words / particles (script-based, high frequency).
URDU_STOPWORDS = frozenset(
    """
    کے کا کی کو میں نے سے پر ہے ہیں ہو تھا تھی تھے گا گی گے
    اور یا اگر لیکن کہ جو جس جن کیا کیوں کب کہاں کیسے اس ان
    یہ وہ ہم تم آپ میرا میری میرے تیرا تیری تیرے اپنا اپنی اپنے
    نہیں نہ بھی ہی صرف تک بعد پہلے دوران درمیان اوپر نیچے باہر
    اندر ایک دو تین سب کچھ کوئی ہر بغیر ساتھ لیے کر کرتے کرتی
    کرنے دیا دی دیں دے رہا رہی رہے تھی۔ گیا گئی گئے
    """.split()
)


def stopwords_for(language: str) -> frozenset:
    if language == "ur":
        return URDU_STOPWORDS
    # default to English for 'en', 'unknown', or anything else --
    # removing a handful of English function words from non-English
    # text is harmless (they simply won't match).
    return ENGLISH_STOPWORDS
