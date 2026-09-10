"""
storage/storage_manager.py
==========================
Section 6 facade: one object exposing every store in the Data Storage
Layer so Section 5 (and dashboards, Section 7) never juggle five
connections themselves.

    Raw News DB        -> .raw          (Section 1, existing)
    Processed News DB  -> .processed    (Section 2, existing)
    Distribution DB    -> .distribution (Section 4, existing)
    Media Storage      -> .media        (this package)
    Logs DB            -> .logs         (this package)
    Config & Models    -> .config_models(this package)

Existing DB classes are imported lazily so ``import storage`` stays
cheap and never pulls in the heavy Section 2/3/4 dependency chain
unless a store is actually used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from data_acquisition.utils.logger import get_logger

from . import config as sc
from .config_model_storage import ConfigModelStorage
from .logs_db import LogsDatabase
from .media_storage import MediaStorage

logger = get_logger("storage.manager")


class StorageManager:
    """Lazy, closeable facade over all six Section 6 stores."""

    def __init__(
        self,
        db_path: Path | str = sc.STORAGE_DB_PATH,
        logs_db_path: Path | str = sc.LOGS_DB_PATH,
        media_root: Path | str = sc.MEDIA_ROOT,
    ):
        self.db_path = Path(db_path)
        self.logs_db_path = Path(logs_db_path)
        self.media_root = Path(media_root)
        self._raw = None
        self._processed = None
        self._distribution = None
        self._media: Optional[MediaStorage] = None
        self._logs: Optional[LogsDatabase] = None
        self._config_models: Optional[ConfigModelStorage] = None

    # -- stores (lazy) ------------------------------------------------------
    @property
    def raw(self):
        if self._raw is None:
            from data_acquisition.database.raw_news_db import RawNewsDatabase

            self._raw = RawNewsDatabase(db_path=self.db_path)
        return self._raw

    @property
    def processed(self):
        if self._processed is None:
            from pipeline.processed_news_db import ProcessedNewsDatabase

            self._processed = ProcessedNewsDatabase(db_path=self.db_path)
        return self._processed

    @property
    def distribution(self):
        if self._distribution is None:
            from distribution.distribution_db import DistributionDatabase

            self._distribution = DistributionDatabase(db_path=self.db_path)
        return self._distribution

    @property
    def media(self) -> MediaStorage:
        if self._media is None:
            self._media = MediaStorage(root=self.media_root, db_path=self.db_path)
        return self._media

    @property
    def logs(self) -> LogsDatabase:
        if self._logs is None:
            self._logs = LogsDatabase(db_path=self.logs_db_path)
        return self._logs

    @property
    def config_models(self) -> ConfigModelStorage:
        if self._config_models is None:
            self._config_models = ConfigModelStorage(db_path=self.db_path)
        return self._config_models

    # -- health ---------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """One status block per store; a failing store never hides the rest."""

        def _probe(label: str, fn) -> Dict[str, Any]:
            try:
                return {label: fn()}
            except Exception as exc:
                return {label: {"ok": False, "error": str(exc)}}

        report: Dict[str, Any] = {}
        report.update(_probe(
            "raw_news_db",
            lambda: {"ok": True, "path": str(self.db_path), "articles": self.raw.count()},
        ))
        report.update(_probe(
            "processed_news_db",
            lambda: {
                "ok": True,
                "path": str(self.db_path),
                "articles": len(self.processed.get_all_preprocessed()),
            },
        ))
        report.update(_probe(
            "distribution_db",
            lambda: {
                "ok": True,
                "path": str(self.db_path),
                "jobs": len(self.distribution.get_all_jobs()),
            },
        ))
        report.update(_probe(
            "media_storage",
            lambda: {"ok": True, **self.media.stats()},
        ))
        report.update(_probe(
            "logs_db",
            lambda: {"ok": True, "path": str(self.logs_db_path), **self.logs.stats()},
        ))
        report.update(_probe(
            "config_model_storage",
            lambda: {
                "ok": True,
                "configs": len(self.config_models.list_configs()),
                "models": len(self.config_models.list_models()),
            },
        ))
        return report

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        for store in (self._raw, self._processed, self._distribution,
                      self._media, self._logs, self._config_models):
            if store is not None:
                try:
                    store.close()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Error closing store %r: %s", store, exc)
        self._raw = self._processed = self._distribution = None
        self._media = self._logs = self._config_models = None

    def __enter__(self) -> "StorageManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
