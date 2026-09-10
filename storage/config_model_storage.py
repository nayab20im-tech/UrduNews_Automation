"""
storage/config_model_storage.py
===============================
Implements Section 6's "Config & Model Storage (Local Models)" box.

Two SQLite tables plus two on-disk locations:

- ``config_store``   -- named JSON config blobs (sources / models /
  categories snapshots produced by Section 5's Config Manager).
- ``model_registry`` -- local model artifacts (TTS voices, avatar /
  lip-sync checkpoints, fonts) registered by name so every stage
  resolves models the same way.
- ``MODELS_DIR``          -- where local model files live.
- ``CONFIG_SNAPSHOTS_DIR`` -- timestamped JSON snapshots for auditing.

No secrets are ever written here: snapshots contain paths, backends and
"configured" flags only (see ``orchestration/config_manager.py``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from . import config as sc

logger = get_logger("storage.config_model_storage")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS config_store (
    name         TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS model_registry (
    name          TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    path          TEXT NOT NULL,
    meta_json     TEXT,
    registered_at TEXT,
    updated_at    TEXT
);
"""


class ConfigModelStorage:
    """Thread-safe SQLite wrapper for config blobs and the local model registry."""

    def __init__(self, db_path: Path | str = sc.STORAGE_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir = sc.MODELS_DIR
        self.snapshots_dir = sc.CONFIG_SNAPSHOTS_DIR
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
        logger.info("Config & model storage connected at %s", self.db_path)

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    # -- config store -------------------------------------------------------
    def save_config(self, name: str, payload: Dict[str, Any]) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO config_store (name, payload_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET payload_json = excluded.payload_json, "
                "updated_at = excluded.updated_at",
                (name, json.dumps(payload, ensure_ascii=False, default=str), safe_iso_now()),
            )

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT payload_json FROM config_store WHERE name = ?", (name,))
            row = cur.fetchone()
            return json.loads(row["payload_json"]) if row else None

    def list_configs(self) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT name, updated_at FROM config_store ORDER BY name")
            return [dict(r) for r in cur.fetchall()]

    def delete_config(self, name: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM config_store WHERE name = ?", (name,))

    # -- model registry --------------------------------------------------------
    def register_model(
        self,
        name: str,
        path: Path | str,
        kind: str = "generic",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = safe_iso_now()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO model_registry (name, kind, path, meta_json, registered_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET kind = excluded.kind, path = excluded.path, "
                "meta_json = excluded.meta_json, updated_at = excluded.updated_at",
                (
                    name,
                    kind,
                    str(path),
                    json.dumps(meta or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM model_registry WHERE name = ?", (name,))
            row = cur.fetchone()
            if not row:
                return None
            out = dict(row)
            out["meta"] = json.loads(out.pop("meta_json") or "{}")
            return out

    def list_models(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM model_registry"
        params: List[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY name"
        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = []
            for r in cur.fetchall():
                out = dict(r)
                out["meta"] = json.loads(out.pop("meta_json") or "{}")
                rows.append(out)
            return rows

    # -- on-disk snapshots --------------------------------------------------------
    def snapshot_to_file(self, name: str, payload: Dict[str, Any]) -> Path:
        """Write a timestamped JSON snapshot under CONFIG_SNAPSHOTS_DIR."""
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        stamp = safe_iso_now().replace(":", "-")
        path = self.snapshots_dir / f"{name}_{stamp}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Config snapshot written to %s", path)
        return path

    def ensure_models_dir(self) -> Path:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self.models_dir

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("Config & model storage connection closed (%s)", self.db_path)

    def __enter__(self) -> "ConfigModelStorage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
