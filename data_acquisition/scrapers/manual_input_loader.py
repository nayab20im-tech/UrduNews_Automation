"""
scrapers/manual_input_loader.py
=================================
Implements the "Manual Input / Local Files" source: anyone on the team
(or another pipeline stage) can drop `.json`, `.csv`, or `.txt` files
into `data/manual_input/` and they'll be picked up on the next run.

Supported formats
------------------
- `.json`: either a single article object, or a list of article objects.
  Recognised keys: title, content, url, category, published_date,
  language, source (any missing key falls back to a sensible default).
- `.csv`: header row required; same column names as the JSON keys above.
- `.txt`: whole file becomes one article; first line is used as the
  title, the rest as content (mainly for quick manual notes/drafts).

Every successfully-loaded file is renamed with a `.loaded` suffix so
re-running the pipeline doesn't reprocess it, unless `mark_processed`
is disabled.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .base_scraper import BaseScraper
from ..database.raw_news_db import Article
from ..utils.text_utils import clean_text


class ManualInputLoader(BaseScraper):
    source_type = "manual"

    def __init__(self, folder: Path | str, mark_processed: bool = True):
        super().__init__(name="manual_input")
        self.folder = Path(folder)
        self.mark_processed = mark_processed
        self.folder.mkdir(parents=True, exist_ok=True)

    def fetch(self) -> List[Article]:
        articles: List[Article] = []
        files = sorted(
            p for p in self.folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".json", ".csv", ".txt"}
        )

        if not files:
            self.logger.info("No new manual input files found in %s", self.folder)
            return articles

        for path in files:
            try:
                if path.suffix.lower() == ".json":
                    file_articles = self._load_json(path)
                elif path.suffix.lower() == ".csv":
                    file_articles = self._load_csv(path)
                else:
                    file_articles = self._load_txt(path)

                self.logger.info("Loaded %d article(s) from %s", len(file_articles), path.name)
                articles.extend(file_articles)

                if self.mark_processed:
                    path.rename(path.with_suffix(path.suffix + ".loaded"))
            except Exception as exc:
                self.logger.error("Failed to load manual input file %s: %s", path, exc)
                continue

        return articles

    # -- format-specific loaders, each testable independently -------------
    def _load_json(self, path: Path) -> List[Article]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
        return [self._record_to_article(r, source_file=path.name) for r in records if isinstance(r, dict)]

    def _load_csv(self, path: Path) -> List[Article]:
        articles: List[Article] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                articles.append(self._record_to_article(row, source_file=path.name))
        return articles

    def _load_txt(self, path: Path) -> List[Article]:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        lines = text.split("\n", 1)
        title = clean_text(lines[0])
        content = clean_text(lines[1]) if len(lines) > 1 else ""
        return [
            self._build_article(
                title=title,
                content=content,
                url="",
                category="manual",
                extra={"source_file": path.name},
            )
        ]

    def _record_to_article(self, record: Dict[str, Any], source_file: str) -> Article:
        title = clean_text(str(record.get("title", "")).strip())
        content = clean_text(str(record.get("content", "")).strip())
        return self._build_article(
            title=title or "(untitled manual entry)",
            content=content,
            url=str(record.get("url", "") or ""),
            category=str(record.get("category", "manual") or "manual"),
            published_date=record.get("published_date"),
            language=record.get("language"),
            extra={"source_file": source_file},
        )
