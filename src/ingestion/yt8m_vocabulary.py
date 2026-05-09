"""Category Entity vocabulary loader.

Parses the curated vocabulary CSV (~700 entities across 24 Verticals)
and provides fast lookups for entity-to-vertical mapping. The schema
matches the original YouTube-8M release for backwards compatibility,
but the data itself is now hand-curated to reflect contemporary ad
targeting (see ``scripts/build_vocabulary_v2.py``).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CSV = Path(__file__).resolve().parent.parent.parent / "vocab" / "youtube8m" / "vocabulary.csv"

VERTICALS: list[str] = [
    "Arts & Entertainment",
    "Autos & Vehicles",
    "Beauty & Fitness",
    "Books & Literature",
    "Business & Industrial",
    "Computers & Electronics",
    "Finance",
    "Food & Drink",
    "Games",
    "Health",
    "Hobbies & Leisure",
    "Home & Garden",
    "Internet & Telecom",
    "Jobs & Education",
    "Law & Government",
    "News",
    "People & Society",
    "Pets & Animals",
    "Real Estate",
    "Reference",
    "Science",
    "Shopping",
    "Sports",
    "Travel",
]

VERTICAL_INDEX: dict[str, int] = {v: i for i, v in enumerate(VERTICALS)}
NUM_VERTICALS: int = len(VERTICALS)


@dataclass(frozen=True)
class Entity:
    index: int
    name: str
    kg_id: str
    verticals: tuple[str, ...]
    train_video_count: int = 0


@dataclass
class Yt8mVocabulary:
    entities: list[Entity] = field(default_factory=list)
    _name_to_entity: dict[str, Entity] = field(default_factory=dict, repr=False)
    _vertical_to_entities: dict[str, list[Entity]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_csv(cls, csv_path: Path) -> Yt8mVocabulary:
        entities: list[Entity] = []
        name_to_entity: dict[str, Entity] = {}
        vertical_to_entities: dict[str, list[Entity]] = {v: [] for v in VERTICALS}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["Name"].strip()
                kg_id = row.get("KnowledgeGraphId", "").strip()

                if not name:
                    continue

                verts: list[str] = []
                for col in ("Vertical1", "Vertical2", "Vertical3"):
                    v = row.get(col, "").strip()
                    if v and v != "(Unknown)" and v in VERTICAL_INDEX:
                        verts.append(v)

                entity = Entity(
                    index=int(row["Index"]),
                    name=name,
                    kg_id=kg_id,
                    verticals=tuple(verts),
                    train_video_count=int(row.get("TrainVideoCount") or 0),
                )
                entities.append(entity)
                name_to_entity[entity.name.lower()] = entity
                for v in entity.verticals:
                    vertical_to_entities[v].append(entity)

        vocab = cls(
            entities=entities,
            _name_to_entity=name_to_entity,
            _vertical_to_entities=vertical_to_entities,
        )
        logger.info(
            "Loaded YouTube-8M vocabulary: %d entities, %d verticals",
            len(entities),
            sum(1 for ents in vertical_to_entities.values() if ents),
        )
        return vocab

    def lookup(self, name: str) -> Entity | None:
        return self._name_to_entity.get(name.lower())

    def get_verticals_for(self, name: str) -> list[str]:
        entity = self.lookup(name)
        if entity is None:
            return []
        return list(entity.verticals)

    def get_entities_for_vertical(self, vertical: str) -> list[Entity]:
        return self._vertical_to_entities.get(vertical, [])

    def entity_names_for_vertical(self, vertical: str) -> list[str]:
        return sorted(e.name for e in self.get_entities_for_vertical(vertical))

    def all_entity_names(self) -> list[str]:
        return sorted(e.name for e in self.entities)

    def vertical_vector_for(self, name: str) -> list[float]:
        """Return a 24-dim binary vector indicating which verticals an entity belongs to."""
        verts = self.get_verticals_for(name)
        vec = [0.0] * NUM_VERTICALS
        for v in verts:
            idx = VERTICAL_INDEX.get(v)
            if idx is not None:
                vec[idx] = 1.0
        return vec


_singleton: Yt8mVocabulary | None = None


def get_vocabulary(csv_path: Path | None = None) -> Yt8mVocabulary:
    global _singleton
    if _singleton is not None:
        return _singleton

    path = csv_path or _DEFAULT_CSV
    if not path.exists():
        logger.warning("YouTube-8M vocabulary not found at %s; using empty vocabulary", path)
        _singleton = Yt8mVocabulary()
        return _singleton

    _singleton = Yt8mVocabulary.from_csv(path)
    return _singleton


def reset_vocabulary() -> None:
    """Reset singleton (for testing)."""
    global _singleton
    _singleton = None