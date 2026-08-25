# UrduNewsAI — AI Processing Pipeline (Section 2: Stages 2.1–2.5)

Continues directly from the existing **Data Acquisition** layer
(`data_acquisition/`, unmodified except for one additive method — see
below). Implements:

```
Data Acquisition (existing)
        │  RawNewsDatabase.get_all(status="raw")
        ▼
2.1  Data Preprocessing        preprocessing/
        ▼
2.2  News Classification       classification/
        ▼
2.3  Duplicate Detection       duplicate_detection/
        ▼
2.4  Key Info / Claim Extraction  extraction/
        ▼
2.5  English Summarization     summarization/
        ▼
Final Processed News Dataset   pipeline/  (orchestrator + storage + CLI)
```

Each stage's output becomes the next stage's input, via new SQLite
tables in the **same** database file the acquisition layer already
uses (`data_acquisition/data/raw_news.db`) — see `pipeline/processed_news_db.py`.
The raw table is only ever touched for its `status` column; all raw
scraped data is preserved unmodified.

---

## 1. What was created / modified

### Created (all new, nothing here existed before)

| Path | Purpose |
|---|---|
| `preprocessing/stopwords.py` | English + Urdu stopword lists |
| `preprocessing/html_cleaner.py` | HTML stripping, ad/boilerplate removal |
| `preprocessing/normalizer.py` | Unicode normalization, sentence splitting, stopword removal |
| `preprocessing/preprocessor.py` | Stage 2.1 orchestration (`Preprocessor`) |
| `classification/category_keywords.py` | Bilingual keyword lexicon for the 7 categories |
| `classification/classifier.py` | Stage 2.2 (`NewsClassifier`, TF-IDF similarity) |
| `duplicate_detection/duplicate_detector.py` | Stage 2.3 (`DuplicateDetector`, TF-IDF + cosine + Union-Find) |
| `extraction/ner.py` | Rule-based Named Entity Recognition |
| `extraction/text_rank.py` | Shared TextRank sentence-ranking utility |
| `extraction/keypoint_extractor.py` | Keypoint extraction |
| `extraction/claim_extractor.py` | Claim/fact sentence identification |
| `extraction/extractor.py` | Stage 2.4 orchestration (`InfoExtractor`) |
| `summarization/summarizer.py` | Stage 2.5 (`Summarizer`, extractive TextRank) |
| `pipeline/config.py` | All configurable thresholds/paths for stages 2.1–2.5 |
| `pipeline/processed_news_db.py` | SQLite storage for every stage's output |
| `pipeline/pipeline_orchestrator.py` | Chains all 5 stages, per-stage + full-run reports |
| `pipeline/run_pipeline.py` | CLI entry point |
| `pipeline/tests/test_pipeline.py` | 39 offline unit tests covering all 5 stages + edge cases |

### Modified

| Path | Change | Why |
|---|---|---|
| `data_acquisition/database/raw_news_db.py` | Added one method, `update_status(article_id, status)` | This is exactly the handoff hook the existing README's "What to wire up next (Section 2)" section asked for. Nothing else in this file, or anywhere else in `data_acquisition/`, was changed. |

No existing Data Acquisition scraper, config, schema, or test was altered or replaced.

---

## 2. How each stage works

### 2.1 Data Preprocessing (`preprocessing/`)
For each `status="raw"` article: strip HTML/entities (BeautifulSoup) →
remove ad/boilerplate lines (pattern list) → check for an **exact**
content duplicate (SHA-256 of title+content, separate from Stage 1's
URL-based dedup and from Stage 2.3's near-duplicate detection) →
detect language (reuses the acquisition layer's `detect_language`,
trusting an existing source-config language first) → Unicode-normalize
text (NFKC, punctuation, whitespace) → split into sentences (regex,
handles both Latin and Urdu sentence-enders `۔؟`) → remove stopwords
into a **separate** `processed_text` field, leaving `cleaned_text`
untouched for summarization. Every article gets a `preprocessing_status`
(`ok` / `empty_content` / `duplicate` / `error`) instead of being
silently dropped.

### 2.2 News Classification (`classification/`)
No labeled training data exists in this project, so training/downloading
a classifier wasn't appropriate. Instead: each of the 7 categories gets
a bilingual (English + Urdu) keyword "profile"; a single `TfidfVectorizer`
is fit across the 7 profiles; each article's stopword-removed text is
compared to every profile with cosine similarity; the best match wins,
or `Others` if below a configurable confidence threshold. `NewsClassifier.classify_text()`
is a plain string-in/category-out function, so a properly trained model
can replace the internals later with zero changes elsewhere.

### 2.3 Duplicate Detection (`duplicate_detection/`)
TF-IDF vectors + cosine similarity (chosen over embeddings: no network
access to an embedding-model hub in this environment, and TF-IDF is
deterministic/reproducible and needs no API key — Requirements 13 & 16).
All valid articles are vectorized together, pairwise cosine similarity
is computed, and any pair above a configurable threshold (default 0.65)
is unioned into a group (Union-Find). Within each group the **longest**
cleaned text is kept as the primary/most-complete source; others are
marked `is_duplicate=True` with their similarity score to the primary —
never deleted, so traceability is preserved.

### 2.4 Key Information / Claim Extraction (`extraction/`)
- **NER**: regex/heuristic-based (title-word + capitalization-run cues
  for PERSON/ORGANIZATION in English, a gazetteer for LOCATION in both
  scripts, regex for DATE/MONEY, keyword match for EVENT). Documented
  limitation: Urdu has no capitalization signal, so Urdu PERSON/ORG
  recall is limited to gazetteer matches — a real gap of the no-download
  approach, not a silent one.
- **Keypoints**: TextRank (networkx PageRank over a TF-IDF sentence-
  similarity graph) picks the most central sentences.
- **Claims**: sentences containing a reporting verb ("said", "announced",
  "کہا", "کے مطابق"...), a number, or a quotation are surfaced verbatim
  — nothing is paraphrased or invented.

### 2.5 Summarization (`summarization/`)
Extractive TextRank (same ranking utility as 2.4) over English-language
articles only (`language == "en"`; non-English articles are marked
`status="skipped_non_english"`, never mistranslated). Extractive by
construction, so it cannot hallucinate: every summary sentence, key
point, and bullet is a verbatim sentence from the article. Very short
articles (this dataset is RSS teaser text, often 1 sentence) are
returned as-is rather than artificially shortened further.

---

## 3. Execution instructions

From the project root (`UrduNewsAI_DataAcquisition/`, one level above
`data_acquisition/` — same convention the existing `data_acquisition/main.py`
uses):

```bash
# 1. Install (adds sklearn/networkx to the existing requirements)
pip install -r data_acquisition/requirements.txt
pip install scikit-learn networkx      # only if not already present

# 2. Make sure raw data exists (skip if you already ran Data Acquisition)
python -m data_acquisition.main --sources rss

# 3. Run the full pipeline (Stages 2.1 -> 2.5) and export the final dataset
python -m pipeline.run_pipeline --export

# Or run one stage at a time:
python -m pipeline.run_pipeline --stage preprocess
python -m pipeline.run_pipeline --stage classify
python -m pipeline.run_pipeline --stage dedup
python -m pipeline.run_pipeline --stage extract
python -m pipeline.run_pipeline --stage summarize

# Check output counts without re-running anything:
python -m pipeline.run_pipeline --stats
```

Output: `data_acquisition/data/processed_news_export.json` (one joined
record per article — preprocessing + classification + duplicate status +
extraction + summary).

### Testing

```bash
pytest pipeline/tests -v          # 39 new tests, fully offline
pytest data_acquisition/tests -v  # original 23 tests, unaffected
```

Both suites pass (62/62) as of this implementation.

---

## 4. Results on the actual acquired dataset (52 BBC RSS articles, 31 EN / 21 UR)

```
2.1 Preprocessing:  52 ok, 0 duplicate, 0 empty, 0 error
2.2 Classification: Politics 11, Sports 6, Technology 3, International 5,
                     Business 1, Others 26 (short RSS teasers -> low
                     keyword overlap for many; expected, not a bug)
2.3 Duplicate Detection: 0 groups found (single-source dataset -- see
                     unit tests for grouping proven on synthetic
                     near-duplicate pairs)
2.4 Extraction: 52/52 processed
2.5 Summarization: 31 ok (English), 21 correctly skipped (Urdu)
```

Sample record (English, Politics):

```json
{
  "title": "Israel re-establishes closed West Bank settlement, defying growing international protests",
  "category": "Politics",
  "classification_confidence": 0.1508,
  "is_duplicate": false,
  "entities": {"LOCATION": ["Israel"]},
  "summary": "Thirty \"pioneer families\" have arrived on a wave of nationalism driven by Israel's government, but the rapid change has left nearby Palestinian residents fearful."
}
```

Sample record (Urdu — summarization correctly skipped, keypoints/entities still populated):

```json
{
  "title": "عمران خان کی ہسپتال منتقلی کا معاملہ، تحریکِ انصاف نے توہینِ عدالت کی درخواست دائر کر دی",
  "category": "Politics",
  "entities": {"LOCATION": ["اسلام آباد", "پاکستان"]},
  "keypoints": [
    "پاکستان تحریکِ انصاف (پی ٹی آئی) نے سابق وزیرِ اعظم عمران خان کو ہسپتال منتقل کرنے کے معاملے پر حکومت کے خلاف توہینِ عدالت کی درخواست دائر کر دی ہے۔"
  ],
  "summary": ""
}
```

---

## 5. Final verification

Pipeline order confirmed as run: **Data Acquisition → Data Preprocessing
→ News Classification → Duplicate Detection → Key Information/Claim
Extraction → English Summarization**, each stage reading only the prior
stage's output table. Every stage tolerates a single bad article without
stopping the batch (each `*.run()` method catches per-item exceptions);
verified against missing title, missing body, very short/long content,
malformed text, and non-English input in `pipeline/tests/test_pipeline.py`.
