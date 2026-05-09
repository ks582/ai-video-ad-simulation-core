"""Section 1.9.3 Behavior Normalizer — KuaiRand/KuaiRec → BehaviorRecord.

Reads KuaiRand-Pure and KuaiRec 2.0 interaction log CSV files and produces
canonical BehaviorRecord instances compatible with the simulator's behavior schema.

KuaiRand-Pure CSV schema (log_random_4_22_to_5_08_pure.csv):
    user_id, video_id, date, hourmin, time_ms,
    is_click, is_like, is_follow, is_comment, is_forward, is_hate,
    long_view, play_time_ms, duration_ms,
    profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab

KuaiRec 2.0 CSV schema (small_matrix.csv / big_matrix.csv):
    user_id, video_id, play_duration, video_duration, ...

Usage
-----
    from src.ingestion.behavior_normalizer import BehaviorNormalizer

    normalizer = BehaviorNormalizer()
    records = normalizer.normalize_csv("data/kuairand/interactions.csv")
    records = normalizer.normalize_kuairec_csv("data/KuaiRec 2.0/data/small_matrix.csv")
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.public_data_schemas import BehaviorRecord

# KuaiRand feedback signal columns (beyond click/like)
_FEEDBACK_COLS = [
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
    "long_view",
]


def _parse_int(val: str, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _parse_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class BehaviorNormalizer:
    """Normalizes KuaiRand/KuaiRec CSV interaction logs to BehaviorRecord.

    Args:
        source_dataset: Label stored in each output record's source_dataset field.
    """

    def __init__(self, source_dataset: str = "kuairand") -> None:
        self._source = source_dataset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_csv(
        self,
        csv_path: str | Path,
        limit: int | None = None,
    ) -> list[BehaviorRecord]:
        """Read a KuaiRand CSV and return normalized BehaviorRecord list.

        Args:
            csv_path: Path to the KuaiRand interaction log CSV.
            limit: If set, only the first *limit* rows are processed.
        """
        records: list[BehaviorRecord] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                records.append(self._row_to_record(row))
        return records

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_record(self, row: dict[str, str]) -> BehaviorRecord:
        play_time_ms = _parse_int(row.get("play_time_ms", "0"))
        duration_ms = _parse_int(row.get("duration_ms", "0"))

        watch_ratio = (
            play_time_ms / duration_ms if duration_ms > 0 else 0.0
        )

        timestamp = self._parse_timestamp(
            row.get("date", ""), row.get("hourmin", "")
        )

        feedback: dict[str, float] = {}
        for col in _FEEDBACK_COLS:
            raw = row.get(col, "")
            if raw:
                feedback[col] = _parse_float(raw)

        return BehaviorRecord(
            user_id=row.get("user_id", ""),
            item_id=row.get("video_id", ""),
            timestamp=timestamp,
            watch_time_ms=play_time_ms,
            watch_ratio=watch_ratio,
            clicked=bool(_parse_int(row.get("is_click", "0"))),
            liked=bool(_parse_int(row.get("is_like", "0"))),
            left_session=False,
            feedback_signals=feedback,
            source_dataset=self._source,
        )

    def normalize_kuairec_csv(
        self,
        csv_path: str | Path,
        limit: int | None = None,
    ) -> list[BehaviorRecord]:
        """Read a KuaiRec 2.0 CSV and return normalized BehaviorRecord list.

        KuaiRec 2.0 uses play_duration / video_duration instead of
        play_time_ms / duration_ms. watch_ratio is capped at 1.0.

        Args:
            csv_path: Path to the KuaiRec small_matrix.csv or big_matrix.csv.
            limit: If set, only the first *limit* rows are processed.
        """
        records: list[BehaviorRecord] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                records.append(self._kuairec_row_to_record(row))
        return records

    def _kuairec_row_to_record(self, row: dict[str, str]) -> BehaviorRecord:
        play_duration = _parse_float(row.get("play_duration", "0"))
        video_duration = _parse_float(row.get("video_duration", "0"))

        watch_ratio = (
            min(play_duration / video_duration, 1.0) if video_duration > 0 else 0.0
        )
        # Convert seconds to milliseconds for consistency with KuaiRand schema
        play_time_ms = int(play_duration * 1000)

        return BehaviorRecord(
            user_id=row.get("user_id", ""),
            item_id=row.get("video_id", ""),
            timestamp=None,
            watch_time_ms=play_time_ms,
            watch_ratio=watch_ratio,
            clicked=False,
            liked=False,
            left_session=False,
            feedback_signals={},
            source_dataset=self._source,
        )

    @staticmethod
    def _parse_timestamp(date_str: str, hourmin_str: str) -> datetime | None:
        """Parse KuaiRand date (YYYYMMDD) + hourmin (HHMM) into datetime."""
        try:
            date_str = date_str.strip()
            hourmin_str = hourmin_str.strip()
            if len(date_str) == 8 and len(hourmin_str) == 4:
                dt_str = f"{date_str}{hourmin_str}"
                return datetime.strptime(dt_str, "%Y%m%d%H%M").replace(
                    tzinfo=timezone.utc
                )
        except (ValueError, AttributeError):
            pass
        return None
