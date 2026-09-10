# UrduNewsAI — Data Acquisition Layer

Implementation of **Section 1 (Data Collection Layer)** of the UrduNewsAI
system architecture: it collects raw news from six source types, runs
everything through one shared scraping engine, de-duplicates on the way in,
and stores the result in a local SQLite database (with JSON export) ready
for Section 2 (AI Processing & Generation Pipeline).

```
News Websites ─┐
RSS Feeds ──────┤
News APIs ──────┼──▶ Web Scraping Engine ──▶ Raw News Database ──▶ (Section 2)
Google Trends ──┤        (BeautifulSoup +          (SQLite / JSON)
Social Media ───┤          Selenium fallback)
Manual Files ───┘
```

## 1. Install

```bash
cd data_acquisition
python3 -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
cp .env.example .env                                   # fill in API keys you plan to use
```

`selenium` needs a matching browser driver (e.g. Chrome + chromedriver) only
if you enable `use_selenium: true` for a JS-heavy site. `snscrape` and
`tweepy` are only needed if you enable `social_media`.

## 2. Configure your sources

Everything lives in `data/sources_config.yaml` — no code changes needed to
add or remove a source.

| Source | Config section | Needs a key? |
|---|---|---|
| RSS Feeds | `rss_feeds` | No |
| News Websites | `websites` | No (see below) |
| News APIs | `news_api` | Yes — `NEWSAPI_KEY` in `.env` |
| Google Trends | `google_trends` | No |
| Social Media | `social_media` | No (snscrape) / optional `TWITTER_BEARER_TOKEN` |
| Manual Input | `manual_input` | No |

Two BBC RSS feeds are enabled out of the box so you can run the pipeline
immediately. Everything else ships **disabled** as a template — RSS URLs and
website HTML change per outlet, so nothing is turned on until you've
verified it against the live site.

### Adding a news website (2.1–2.5 in the diagram start here)

Website scraping is entirely selector-driven — add an entry under
`websites:` in `sources_config.yaml`:

1. Open the site's listing/homepage in a browser, right-click a headline
   link → **Inspect** → copy a CSS selector that matches every article link
   (e.g. `a.story-link`).
2. Open one article page, do the same for the title, body paragraphs, date,
   and category.
3. Fill those into the `selectors:` block and set `enabled: true`.
4. If the content doesn't appear in "View Page Source" (i.e. it's loaded by
   JavaScript), set `use_selenium: true` — the engine will render the page
   with headless Chrome instead of a plain HTTP GET.

No selectors are hard-coded in `scrapers/news_website_scraper.py` on
purpose: a wrong guess would silently scrape the wrong thing, so the code
only ever uses what you've explicitly configured and verified.

### Adding manual/local files

Drop `.json`, `.csv`, or `.txt` files into `data/manual_input/`. JSON/CSV
rows use these fields: `title, content, url, category, published_date,
language`. Processed files are renamed with a `.loaded` suffix so re-runs
don't reprocess them.

## 3. Run

```bash
python -m data_acquisition.main                       # run every enabled source
python -m data_acquisition.main --sources rss,manual   # only these sources
python -m data_acquisition.main --export json          # also write raw_news_export.json
python -m data_acquisition.main --stats                # print DB stats and exit
```

Each run prints a per-source JSON report (fetched / inserted / duplicates /
errors / duration) — one source failing (bad selector, expired key, feed
down) never stops the others from running.

## 4. Test (fully offline, no network required)

```bash
pytest tests -v
```

23 tests cover the database (insert/dedup/stats/export), text utilities
(cleaning, language detection, hashing), RSS parsing, website listing/article
parsing, and every manual-input format — all against local fixture files in
`tests/fixtures/`, so the suite runs the same with or without internet
access.

## 5. Project layout

```
data_acquisition/
├── config.py                    # paths, env vars, source-config loader
├── main.py                      # CLI entry point
├── database/
│   └── raw_news_db.py           # SQLite store + Article dataclass + JSON export
├── engine/
│   └── scraping_engine.py       # requests+BeautifulSoup engine: retries,
│                                 #   rate limiting, robots.txt, Selenium fallback
├── scrapers/
│   ├── base_scraper.py          # shared fetch() contract + Article builder
│   ├── news_website_scraper.py  # config-driven site scraping
│   ├── rss_scraper.py           # feedparser-based RSS collection
│   ├── news_api_client.py       # NewsAPI.org client
│   ├── trends_scraper.py        # Google Trends via trendspyg (RSS-backed)
│   ├── social_media_scraper.py  # Twitter/X via snscrape (+ optional official API)
│   └── manual_input_loader.py   # local JSON/CSV/TXT loader
├── orchestrator/
│   └── data_acquisition_orchestrator.py  # builds scrapers from config, runs
│                                          #   them all, stores results, reports
├── utils/
│   ├── logger.py                # shared rotating-file + console logger
│   └── text_utils.py            # cleaning, language detection, hashing
├── data/
│   ├── sources_config.yaml      # all source definitions
│   └── manual_input/            # drop local files here
└── tests/                       # offline pytest suite + fixtures
```

## 6. Design notes

- **De-duplication**: every article is hashed on insert. If it has a URL,
  the URL alone is the dedup key (so a re-scrape with a slightly reworded
  headline still collapses to one row). Only URL-less sources (manual notes,
  Trends terms) fall back to a title+content hash. This is intentionally
  separate from Section 2.3's semantic/near-duplicate detection later in the
  pipeline — this is raw-level, exact-match dedup only.
- **Resilience**: every scraper's `fetch()` is wrapped by the orchestrator —
  one source erroring out (network failure, bad API key, changed HTML) is
  logged and skipped; it never stops the rest of the sources from running.
- **Politeness**: the scraping engine enforces a minimum delay between
  requests to the same host, retries transient failures (429/5xx) with
  backoff, rotates User-Agent strings, and checks `robots.txt` before
  fetching (toggle via `RESPECT_ROBOTS_TXT` in `.env`).
- **Testability**: every scraper separates "get the HTML/XML" (network I/O)
  from "parse it into Articles" (pure logic), so parsing can be — and is —
  unit tested against local fixtures with zero network dependency.

## 6b. Fixed: Dawn.com and Google Trends (2026-08-25)

Both of these were failing. Root causes and fixes:

**Dawn.com Website Scraper.** There was no real Dawn entry in
`sources_config.yaml` -- only a disabled `Example Urdu News Site` template
with placeholder CSS selectors (`a.story-link`, `h1.headline`, ...) that
don't match any real site. Enabling it against Dawn's actual markup, or
against the template as-is, returns zero articles.

Rather than reverse-engineer Dawn's current CSS classes (which is exactly
what breaks again on their next redesign), Dawn is now pulled in via their
own official RSS feed, `https://www.dawn.com/feeds/home` -- verified
working and returning full article HTML in `<content:encoded>`, not just
a teaser. This reuses the same `rss_feeds` pipeline that was already
working for BBC. `scrapers/news_website_scraper.py` itself is unchanged
and still there for sites that genuinely have no RSS feed -- just fill in
real, verified selectors before enabling an entry under `websites:`.

**Google Trends.** The `pytrends` library this used was archived
(read-only) by its own maintainers on 2025-04-17 and no longer works
against Google's current endpoints -- it fails with a 404 or, as seen in
this project's logs, a raw connection error. `scrapers/trends_scraper.py`
now uses `trendspyg` instead, an actively maintained replacement whose
RSS-backed path reads Google's own public "Daily Search Trends" feed
(`https://trends.google.com/trending/rss?geo=<COUNTRY>`) rather than
Google's internal widget API. No Selenium/browser and no API key needed.
One config change goes with it: `google_trends.country` in
`sources_config.yaml` must now be an ISO-3166-1 alpha-2 code (`"PK"`), not
the old pytrends-style long-form name (`"pakistan"`).

Also hardened `rss_scraper.py`: it now fetches feeds with a normal browser
User-Agent before handing bytes to `feedparser`, instead of letting
`feedparser` make its own request with a generic `python-feedparser/x.y`
UA. Some publishers 403 that default UA, which `feedparser` then reports
as an opaque XML "syntax error" -- this fetches manually so a blocked
request surfaces as a clear HTTP error instead.

To pick up the fix: `pip install -r requirements.txt` (drops `pytrends`,
adds `trendspyg`), then re-run. All 23 existing offline tests still pass
unmodified.

## 7. Section 2 -- AI Processing Pipeline

Implemented in `../pipeline/` (plus `../preprocessing/`, `../classification/`,
`../duplicate_detection/`, `../extraction/`, `../summarization/`) --
see `../pipeline/README.md` for full documentation. It reads
`RawNewsDatabase.get_all(status="raw")`, advances each row's `status` via
the `update_status()` method below as it's processed, and runs Stages
2.1 Data Preprocessing -> 2.2 News Classification -> 2.3 Duplicate
Detection -> 2.4 Key Information/Claim Extraction -> 2.5 English
Summarization, writing every stage's output to new tables in this same
database.
