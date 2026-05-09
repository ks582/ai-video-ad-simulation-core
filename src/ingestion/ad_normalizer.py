"""Section 1.9.3 Ad Normalizer — LAMBDA / UltraLAMBDA → AdCreativeRecord.

Reads LAMBDA and UltraLAMBDA JSONL files and produces canonical
AdCreativeRecord instances compatible with the simulator's ad schema.

LAMBDA JSONL schema:
    {video_id, recall_score, youtube_id,
     ad_details: {Audio, Brand, Duration, Orientation, Pace, Scenes, Title},
     _split}

UltraLAMBDA JSONL schema:
    {id, memorability, _split}

Usage
-----
    from src.ingestion.ad_normalizer import AdNormalizer

    normalizer = AdNormalizer()
    records = normalizer.normalize_lambda("data/lambda/lambda_all.jsonl")
    records += normalizer.normalize_ultralambda("data/ultralambda/ultralambda_all.jsonl")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.ingestion.public_data_schemas import AdCreativeRecord

# UltraLAMBDA memorability label → numeric score mapping
_MEMORABILITY_MAP: dict[str, float] = {
    "high": 0.85,
    "medium": 0.50,
    "low": 0.20,
}


def _parse_duration_sec(duration_str: str) -> float:
    """Parse duration strings like '54 second', '1 minute', '1:30' → seconds."""
    if not duration_str:
        return 0.0
    s = duration_str.strip().lower()
    # "N second(s)"
    m = re.match(r"^(\d+(?:\.\d+)?)\s*seconds?$", s)
    if m:
        return float(m.group(1))
    # "N minute(s)"
    m = re.match(r"^(\d+(?:\.\d+)?)\s*minutes?$", s)
    if m:
        return float(m.group(1)) * 60
    # "M:SS"
    m = re.match(r"^(\d+):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    # plain number
    try:
        return float(s)
    except ValueError:
        return 0.0


class AdNormalizer:
    """Normalizes LAMBDA and UltraLAMBDA JSONL to AdCreativeRecord."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_lambda(
        self,
        jsonl_path: str | Path,
        limit: int | None = None,
    ) -> list[AdCreativeRecord]:
        """Read LAMBDA JSONL and return normalized AdCreativeRecord list.

        Args:
            jsonl_path: Path to the LAMBDA all.jsonl file.
            limit: If set, only the first *limit* entries are processed.
        """
        records: list[AdCreativeRecord] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(self._lambda_to_record(obj))
        return records

    def normalize_ultralambda(
        self,
        jsonl_path: str | Path,
        limit: int | None = None,
    ) -> list[AdCreativeRecord]:
        """Read UltraLAMBDA JSONL and return normalized AdCreativeRecord list.

        Args:
            jsonl_path: Path to the UltraLAMBDA all.jsonl file.
            limit: If set, only the first *limit* entries are processed.
        """
        records: list[AdCreativeRecord] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(self._ultralambda_to_record(obj))
        return records

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _lambda_to_record(obj: dict) -> AdCreativeRecord:
        details = obj.get("ad_details") or {}
        scenes = details.get("Scenes") or []

        # Normalise scene list to plain dicts with consistent keys
        normalised_scenes = [
            {
                "scene_id": s.get("Number", ""),
                "description": s.get("Description", ""),
                "tags": s.get("Tags", ""),
                "emotions": s.get("Emotions", ""),
                "tone": s.get("Tone", ""),
                "text_shown": s.get("Text Shown", ""),
                "visual_complexity": s.get("Visual Complexity", ""),
            }
            for s in scenes
            if isinstance(s, dict)
        ]

        ad_attrs = {}
        for key in ("Audio", "Orientation", "Pace", "Title"):
            val = details.get(key, "")
            if val:
                ad_attrs[key.lower()] = val

        recall = obj.get("recall_score", 0.0)
        try:
            recall = float(recall)
        except (TypeError, ValueError):
            recall = 0.0

        return AdCreativeRecord(
            ad_id=str(obj.get("youtube_id") or obj.get("video_id", "")),
            brand=details.get("Brand", ""),
            duration_sec=_parse_duration_sec(details.get("Duration", "")),
            memorability=0.0,  # LAMBDA has recall_score, not memorability
            recall_score=recall,
            scene_features=normalised_scenes,
            ad_attributes=ad_attrs,
            source_dataset="lambda",
        )

    @staticmethod
    def _ultralambda_to_record(obj: dict) -> AdCreativeRecord:
        mem_label = str(obj.get("memorability", "")).lower()
        mem_score = _MEMORABILITY_MAP.get(mem_label, 0.0)

        return AdCreativeRecord(
            ad_id=str(obj.get("id", "")),
            brand="",
            duration_sec=0.0,
            memorability=mem_score,
            recall_score=0.0,
            scene_features=[],
            ad_attributes={"memorability_label": mem_label} if mem_label else {},
            source_dataset="ultralambda",
        )
