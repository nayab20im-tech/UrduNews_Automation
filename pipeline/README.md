# UrduNewsAI — AI Processing Pipeline (Section 2: Stages 2.1–2.12)

Continues directly from the existing **Data Acquisition** layer
(`data_acquisition/`, unmodified except for one additive method — see
below). Implements:

```
Data Acquisition (existing)
        │  RawNewsDatabase.get_all(status="raw")
        ▼
2.1  Data Preprocessing           preprocessing/
        ▼
2.2  News Classification          classification/
        ▼
2.3  Duplicate Detection          duplicate_detection/
        ▼
2.4  Key Info / Claim Extraction  extraction/
        ▼
2.5  English Summarization        summarization/
        ▼
2.6  Translation & Urdu Refinement   translation/
        ▼
2.7  Bias & Sensational Detection    bias_detection/
        ▼
2.8  Fact Verification & Evidence    verification/
        ▼
2.9  Final Urdu Script Generation    script_generation/
        ▼
2.10 Script Structuring for News     script_generation/
        ▼
2.11 Text-to-Speech (Urdu voice)   tts/
        ▼
2.12 Audio Post-Processing          audio_processing/
        ▼
Final Processed News Dataset     pipeline/  (orchestrator + storage + CLI)
  + WAV audio per article         data_acquisition/data/media/{audio,audio_processed}/
        ▼  (hand-off to Section 3 Video Production: 3.1 Avatar / 3.4 Composition)
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
| `translation/transliterator.py` | Rule-based English→Urdu-script transliteration (fallback for unknown tokens) |
| `translation/lexicon.py` | Bilingual news lexicon: phrase map, word map, function words, register map |
| `translation/translator.py` | Stage 2.6 (`UrduTranslator`: pattern + dictionary + transliteration; optional local-LLM backend) |
| `translation/refiner.py` | Stage 2.6 (`UrduRefiner`: grammar correction, style/readability enhancement) |
| `bias_detection/bias_lexicon.py` | Bilingual clickbait patterns, sensational/loaded/sentiment lexicons, neutral replacements |
| `bias_detection/bias_detector.py` | Stage 2.7 (`BiasDetector`) |
| `verification/fact_checker.py` | Stage 2.8 (`FactChecker`: internal evidence retrieval + source credibility + confidence) |
| `script_generation/script_generator.py` | Stage 2.9 (`ScriptGenerator`: evidence-grounded Urdu script, template + optional local-LLM backend) |
| `script_generation/script_structurer.py` | Stage 2.10 (`ScriptStructurer`: headline/intro/main story/key points/ending-CTA) |
| `audio_processing/wav_io.py` | Dependency-free WAV read/write (stdlib `wave`+`array`, 8/16/24/32-bit in, 16-bit out) shared by 2.11/2.12 |
| `tts/wav_synth.py` | Deterministic offline placeholder voice (tone-burst WAV per script, length ∝ word count) |
| `tts/tts_engine.py` | Stage 2.11 (`TTSEngine`: auto-selects XTTS/Coqui → Piper → espeak-ng → placeholder) |
| `audio_processing/post_processor.py` | Stage 2.12 (`AudioPostProcessor`: noise gate, silence adjust, peak normalize, optional BGM mix) |
| `pipeline/config.py` | All configurable thresholds/paths for stages 2.1–2.12 |
| `pipeline/processed_news_db.py` | SQLite storage for every stage's output |
| `pipeline/pipeline_orchestrator.py` | Chains all 12 stages, per-stage + full-run reports |
| `pipeline/run_pipeline.py` | CLI entry point |
| `pipeline/tests/test_pipeline.py` | 82 offline unit/e2e tests covering all 12 stages + edge cases |

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

### 2.6 Translation & Urdu Refinement (`translation/`)
Offline-first, same swap-out philosophy as 2.2/2.3. `UrduTranslator`
translates English → Urdu in three layers: (1) a few news-syntax
patterns are reordered into natural Urdu ("X said Y" → "X نے کہا کہ Y",
"According to X, Y" → "X کے مطابق، Y"); (2) a bilingual news lexicon
(~200 entries: phrases longest-first, then words, then function words);
(3) unknown tokens — mostly proper nouns — are transliterated into
Urdu script by a rule-based grapheme mapper so no entity is dropped.
If `TRANSLATION_BACKEND=llm` and `LOCAL_LLM_ENDPOINT` point at a local
OpenAI-compatible server (Ollama / llama.cpp), it is tried first and
the dictionary backend is the automatic fallback on any failure — the
pipeline never depends on the network. Urdu-source articles pass
through untranslated. `UrduRefiner` then applies form-level fixes only
(NFKC + punctuation normalization, colloquial→formal register lift e.g.
"بولا"→"کہا", duplicate-token collapse, punctuation spacing, over-long
sentence splitting, terminal "۔"): it never rewrites meaning, so the
no-hallucination requirement holds by construction. Every applied fix
is recorded in `refinement_notes` for auditability. Documented
limitation: pattern-unmatched English sentences come out gloss-style
(English word order) — a deterministic baseline that the optional local
LLM backend upgrades to fluent NMT with zero other code changes.

### 2.7 Bias & Sensational Language Detection (`bias_detection/`)
Three explainable lexicon/rule signals, bilingual (English + Urdu):
**clickbait detection** (headline patterns like "You won't believe…",
"X ways to…", Urdu equivalents, ALL-CAPS words, exclamation abuse),
**sensational-word score** (hype-vocabulary density per 100 words,
0..1), and **sentiment/bias analysis** (high-precision sentiment
lexicons give −1..+1; loaded words like "regime", "so-called", "سازش"
give a 0..1 loaded score). Composite `bias_score` =
0.4·clickbait + 0.3·sensational + 0.3·loaded, labelled
neutral/mild/sensational against configurable thresholds. Every hit is
stored in `flags` so a human can audit *why* an article was flagged.
The stage only scores — toning sensational wording down happens in 2.9.

### 2.8 Fact Verification & Evidence Retrieval (`verification/`)
Fully offline cross-source corroboration inside the acquired corpus
(no external fact-check API — Requirements 13/16). Each Stage-2.4 claim
is compared (TF-IDF cosine, same approach as 2.3) against every *other*
article; articles above `EVIDENCE_SIMILARITY_THRESHOLD` count as
supporting evidence. **Source credibility scoring** uses a configurable
per-source map (`TRUSTED_SOURCES_JSON`, default 0.5 for unknown
sources). Per-claim **confidence** =
0.5·best_similarity + 0.3·mean_supporting_credibility +
0.2·min(1, n_supporting/min_corroborating_sources); a claim seen in
≥ `MIN_CORROBORATING_SOURCES` distinct other sources is "corroborated",
in one other article "single_source", else "unverified". Articles carry
an aggregate verdict + mean confidence. Nothing is deleted or rewritten
— 2.9 uses the verdicts to hedge uncorroborated claims.

### 2.9 Final Urdu News Script Generation (`script_generation/`)
Evidence-grounded, professional-register Urdu broadcast script. The
default **template** backend is extractive by construction (every
script sentence is a verbatim Urdu sentence or a dictionary translation
of a verbatim English sentence → cannot hallucinate): sensational
wording is toned down to a neutral register before/while translating
(`bias_lexicon.EN_TONE_DOWN` / `NEUTRAL_REPLACEMENTS_UR`, guided by
2.7), and articles whose claims are unverified/partially corroborated
(2.8) carry an explicit editorial clarification line instead of
presenting unverified claims as fact. The optional **llm** backend
(`SCRIPT_BACKEND=llm` + `LOCAL_LLM_ENDPOINT`) asks a local
Llama-3/Mistral/DeepSeek server to write the script from the extracted
evidence only, falling back to the template on any failure.

### 2.10 Script Structuring for News (`script_generation/`)
Pure restructuring of the 2.9 script into the broadcast layout the
downstream Video Production pipeline (Section 3) consumes:
`ہیڈلائن` (Urdu headline), `تعارف` (intro lead), `مرکزی خبر` (main
story), `اہم نکات` (key-point bullets — verbatim keypoints, translated
for English sources), `اختتامیہ` (ending/CTA, a fixed channel-level
line from `SCRIPT_CTA_TEXT`). `full_script` renders all sections in
broadcast order for direct hand-off to TTS (2.11) / avatar (3.1). No
article text is invented or rewritten here.

### 2.11 Text-to-Speech — Urdu Voice Generation (`tts/`)
Reads each article's `structured_scripts.full_script` and writes one WAV
per article into the Media Storage area (`data/media/audio/`). Backend
selection mirrors the 2.6/2.9 swap-out philosophy: `TTS_BACKEND=auto`
tries, in order, **xtts** (Coqui TTS / XTTS-v2 with a local model dir +
speaker sample — `XTTS_MODEL_PATH`, `XTTS_SPEAKER_WAV`), **piper**
(local Piper binary + model — `PIPER_MODEL_PATH`), **espeak**
(espeak-ng ships an Urdu voice `-v ur` and needs no downloads), and
finally a deterministic **placeholder** synth (`wav_synth`: one
speech-like tone burst per sentence, length proportional to word count,
prosody-like pitch variation) so the pipeline always produces a valid
WAV artifact even on a bare host with no TTS engine. Emotion/tone
control maps to `TTS_SPEED`/`TTS_PITCH` (espeak `-s`/`-p`). Every
record stores `backend_used`, so placeholder audio is never mistaken
for a natural voice. On hosts with `espeak-ng` installed it is
auto-detected and used with zero config changes.

### 2.12 Audio Post-Processing (`audio_processing/`)
Pure-stdlib DSP over 16-bit PCM (`wave` + `array`; no numpy/audioop so
it survives Python 3.13), applied in the architecture's order:
(1) **noise reduction** — a hard noise gate zeroes samples below
`AUDIO_NOISE_GATE` (removes hiss between speech); (2) **pause/silence
adjustment** — silent runs longer than `AUDIO_MAX_SILENCE_MS` are
collapsed to it, then leading/trailing silence is trimmed with a small
margin; (3) **volume normalization** — peak-normalized to
`AUDIO_TARGET_PEAK` (default 0.85, broadcast-safe) so every article
plays at the same loudness regardless of backend; (4) **background
music (optional)** — if `BACKGROUND_MUSIC_PATH` points at a same-rate
WAV it is looped under the voice at `BACKGROUND_MUSIC_GAIN`. Processed
files land in `data/media/audio_processed/` (raw TTS output kept
intact); duration/peak before-after and removed silence are stored per
article for audit.

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

# 3. Run the full pipeline (Stages 2.1 -> 2.12) and export the final dataset
python -m pipeline.run_pipeline --export

# Or run one stage at a time:
python -m pipeline.run_pipeline --stage preprocess
python -m pipeline.run_pipeline --stage classify
python -m pipeline.run_pipeline --stage dedup
python -m pipeline.run_pipeline --stage extract
python -m pipeline.run_pipeline --stage summarize
python -m pipeline.run_pipeline --stage translate
python -m pipeline.run_pipeline --stage bias
python -m pipeline.run_pipeline --stage verify
python -m pipeline.run_pipeline --stage generate
python -m pipeline.run_pipeline --stage structure
python -m pipeline.run_pipeline --stage tts
python -m pipeline.run_pipeline --stage audio

# Check output counts without re-running anything:
python -m pipeline.run_pipeline --stats
```

Output: `data_acquisition/data/processed_news_export.json` (one joined
record per article — preprocessing + classification + duplicate status +
extraction + summary + translation + bias + verification + script +
structured script + TTS/processed-audio paths) plus WAV audio per
article under `data_acquisition/data/media/{audio,audio_processed}/`.

### Optional local-LLM upgrade (still fully offline / local)

```bash
export TRANSLATION_BACKEND=llm          # fluent NMT for 2.6
export SCRIPT_BACKEND=llm               # LLM script writing for 2.9
export LOCAL_LLM_ENDPOINT=http://localhost:11434   # e.g. Ollama / llama.cpp
export LOCAL_LLM_MODEL=llama3
```

Without these, the deterministic dictionary/template backends run and
the LLM is only attempted (and skipped) if configured.

### Optional real-voice TTS upgrade (still fully offline / local)

```bash
# Option A (easiest, real Urdu voice, no downloads): install the system binary
sudo apt install espeak-ng              # auto-detected by TTS_BACKEND=auto

# Option B: Piper neural TTS
export TTS_BACKEND=piper
export PIPER_MODEL_PATH=/path/to/urdu-voice.onnx

# Option C: Coqui XTTS-v2 (natural voice cloning)
export TTS_BACKEND=xtts
export XTTS_MODEL_PATH=/path/to/xtts-model-dir
export XTTS_SPEAKER_WAV=/path/to/speaker-sample.wav

# Audio tuning (all stages honor these):
export TTS_SPEED=160 TTS_PITCH=50      # espeak speed (wpm) / pitch (0-99)
export AUDIO_TARGET_PEAK=0.85 AUDIO_MAX_SILENCE_MS=700
export BACKGROUND_MUSIC_PATH=/path/to/music.wav BACKGROUND_MUSIC_GAIN=0.08
```

Without any of these, the deterministic placeholder synth runs and the
pipeline still emits valid, well-formed WAV artifacts.

### Testing

```bash
pytest pipeline/tests -v          # 82 offline tests, all 12 stages + e2e run_all
pytest data_acquisition/tests -v  # original 23 tests, unaffected
```

Both suites pass (105/105) as of this implementation.

---

## 4. Results on the actual acquired dataset (52 BBC RSS articles, 31 EN / 21 UR)

```
2.1  Preprocessing:        52 ok, 0 duplicate, 0 empty, 0 error
2.2  Classification:       Politics 11, Sports 6, Technology 3, International 5,
                            Business 1, Others 26 (short RSS teasers -> low
                            keyword overlap for many; expected, not a bug)
2.3  Duplicate Detection:  0 groups (single-source dataset -- grouping proven
                            on synthetic near-duplicate pairs in unit tests)
2.4  Extraction:           52/52 processed
2.5  Summarization:        31 ok (English), 21 correctly skipped (Urdu)
2.6  Translation:          52 ok (31 translated EN->UR, 21 Urdu passthrough+refined)
2.7  Bias Detection:       32 neutral, 20 mild, 0 sensational (BBC is
                            wire-style; mild flags come from sentiment words
                            like "killed"/"protest" in conflict news)
2.8  Fact Verification:    32 no_claims (1-sentence teasers), 19 unverified,
                            1 partially_corroborated (single-source dataset --
                            corroboration proven cross-source in unit tests)
2.9  Script Generation:    52/52 (template backend; unverified claims hedged
                            with an editorial clarification line)
2.10 Script Structuring:   52/52 structured broadcast scripts
2.11 Text-to-Speech:       52/52 WAV files (placeholder backend on this
                            host; espeak-ng auto-detected when installed)
2.12 Audio Post-Processing: 52/52 (peaks normalized to 0.85, ~12 s of
                            dead silence removed across all articles)
```

Sample structured script (Urdu source — passthrough + refinement path):

```
ہیڈلائن: عمران خان کی ہسپتال منتقلی کا معاملہ، تحریکِ انصاف نے توہینِ عدالت کی درخواست دائر کر دی۔

تعارف: پاکستان تحریکِ انصاف (پی ٹی آئی) نے سابق وزیرِ اعظم عمران خان کو ہسپتال منتقل کرنے کے
معاملے پر حکومت کے خلاف توہینِ عدالت کی درخواست دائر کر دی ہے۔

مرکزی خبر: درخواست عزیر بھنڈاری اور سلمان اکرم راجہ کی جانب سے دائر کی گئی۔ توہینِ عدالت کی درخواست
میں عمران خان کو فوری طور پر شفا انٹرنیشنل ہسپتال اسلام آباد منتقل کرنے کی استدعا کی گئی ہے۔

اہم نکات:
- پاکستان تحریکِ انصاف (پی ٹی آئی) نے سابق وزیرِ اعظم عمران خان کو ہسپتال منتقل کرنے کے معاملے پر
  حکومت کے خلاف توہینِ عدالت کی درخواست دائر کر دی ہے۔
- درخواست عزیر بھنڈاری اور سلمان اکرم راجہ کی جانب سے دائر کی گئی۔

اختتامیہ: مزید خبروں اور تازہ ترین اپڈیٹس کے لیے ہمارے چینل سے جڑے رہیں۔
```

Sample structured script (English source — dictionary-translation path;
gloss-style word order is the documented offline baseline, upgraded by
the optional local-LLM backend):

```
ہیڈلائن: کانادا نے کہا کہ یہ میچ ہمیں تاریففس کے لیے ڈالر اس تجارت مذاکرات بریک دوون

تعارف: نیا 50 لیوی پر 20 بن کانادین یمپورتس کومیس میں فورکی کے بعد لاست مینوتی بریکدوون میں تجارت مذاکرات ۔

مرکزی خبر: وضاحت: اس خبر کے بعض دعوے آزاد ذرائع سے مکمل طور پر تصدیق شدہ نہیں ہیں۔

اختتامیہ: مزید خبروں اور تازہ ترین اپڈیٹس کے لیے ہمارے چینل سے جڑے رہیں۔
```

---

## 5. Final verification

Pipeline order confirmed as run: **Data Acquisition → Data Preprocessing
→ News Classification → Duplicate Detection → Key Information/Claim
Extraction → English Summarization → Translation & Urdu Refinement →
Bias & Sensational Detection → Fact Verification & Evidence Retrieval →
Final Urdu News Script Generation → Script Structuring for News →
Text-to-Speech → Audio Post-Processing**, each stage reading only the
prior stage's output table. Every stage tolerates a single bad article
without stopping the batch (each `*.run()` method catches per-item
exceptions); verified against missing title, missing body, very
short/long content, malformed text, non-English input, missing/empty
audio files, and BGM sample-rate mismatch in
`pipeline/tests/test_pipeline.py`, including a full end-to-end
`run_all()` test over a temporary raw+processed database pair.

The processed WAV output (`data/media/audio_processed/`) together with
`structured_scripts.full_script` is the hand-off point for the next
section of the architecture: **3.1 AI Avatar Generation** and **3.4
Video Composition**.
