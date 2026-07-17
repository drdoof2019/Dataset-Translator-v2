"""
Checkpoint/resume state manager for translation jobs.
"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def _state_key(dataset_id: str, split: str) -> str:
    """Generate a deterministic filename from dataset_id + split."""
    raw = f"{dataset_id}::{split}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class ProgressManager:
    """Manages per-dataset JSON checkpoint files in output/state/."""

    def __init__(self, dataset_id: str, split: str, columns: list[str], total_rows: int):
        self.dataset_id = dataset_id
        self.split = split
        self.columns = columns
        self.total_rows = total_rows
        key = _state_key(dataset_id, split)
        self.state_path: Path = config.STATE_DIR / f"{key}.json"
        self._state: dict = self._load_or_init()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def completed_rows(self) -> int:
        return self._state.get("completed_rows", 0)

    @property
    def last_saved_row(self) -> int:
        return self._state.get("last_saved_row", -1)

    def has_pending(self) -> bool:
        """True if there is a previous incomplete job for this dataset."""
        return self.completed_rows > 0 and self.completed_rows < self.total_rows

    def save_progress(self, row_index: int):
        """Update checkpoint after completing row `row_index`."""
        self._state["completed_rows"] = row_index + 1
        self._state["last_saved_row"] = row_index
        self._state["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._flush()

    def get_resume_row(self) -> int:
        """Return the row index to resume from."""
        return self.completed_rows

    def clear(self):
        """Delete the state file."""
        if self.state_path.exists():
            self.state_path.unlink()
            logger.info("State file cleared: %s", self.state_path)
        self._state = self._default_state()

    def get_state(self) -> dict:
        """Return a copy of the current state (for UI display)."""
        return dict(self._state)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _default_state(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "columns": self.columns,
            "total_rows": self.total_rows,
            "completed_rows": 0,
            "last_saved_row": -1,
            "timestamp": None,
        }

    def _load_or_init(self) -> dict:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("dataset_id") == self.dataset_id:
                    logger.info("Loaded existing progress: %d/%d rows", data.get("completed_rows", 0), self.total_rows)
                    return data
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt state file, starting fresh.")
        return self._default_state()

    def _flush(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)


def list_pending_jobs() -> list[dict]:
    """Scan output/state/ for incomplete jobs."""
    jobs = []
    for f in config.STATE_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("completed_rows", 0) < data.get("total_rows", 0):
                data["_file"] = str(f)
                jobs.append(data)
        except Exception:
            pass
    return jobs
