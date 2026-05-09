"""Unit tests for YouTube-8M vocabulary loader."""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest

from src.ingestion.yt8m_vocabulary import (
    NUM_VERTICALS,
    VERTICAL_INDEX,
    VERTICALS,
    Entity,
    Yt8mVocabulary,
    get_vocabulary,
    reset_vocabulary,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_vocabulary()
    yield
    reset_vocabulary()


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "vocabulary.csv"
    fieldnames = [
        "Index", "TrainVideoCount", "KnowledgeGraphId", "Name",
        "WikiUrl", "Vertical1", "Vertical2", "Vertical3", "WikiDescription",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full = {k: "" for k in fieldnames}
            full.update(row)
            writer.writerow(full)
    return path


class TestVerticalConstants:
    def test_24_verticals(self) -> None:
        assert NUM_VERTICALS == 24

    def test_index_matches_list(self) -> None:
        for i, v in enumerate(VERTICALS):
            assert VERTICAL_INDEX[v] == i

    def test_no_duplicates(self) -> None:
        assert len(VERTICALS) == len(set(VERTICALS))

    def test_sorted_alphabetically(self) -> None:
        assert VERTICALS == sorted(VERTICALS)


class TestYt8mVocabularyFromCSV:
    def test_parses_entities(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [
            {"Index": "0", "TrainVideoCount": "100", "Name": "Game",
             "KnowledgeGraphId": "/m/03bt1gh", "Vertical1": "Games"},
            {"Index": "1", "TrainVideoCount": "50", "Name": "Car",
             "KnowledgeGraphId": "/m/0k4j", "Vertical1": "Autos & Vehicles"},
        ])
        vocab = Yt8mVocabulary.from_csv(csv_path)
        assert len(vocab.entities) == 2

    def test_lookup_case_insensitive(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [
            {"Index": "0", "Name": "Football", "Vertical1": "Sports"},
        ])
        vocab = Yt8mVocabulary.from_csv(csv_path)
        assert vocab.lookup("football") is not None
        assert vocab.lookup("Football") is not None
        assert vocab.lookup("FOOTBALL") is not None
        assert vocab.lookup("nonexistent") is None

    def test_get_verticals_for(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [
            {"Index": "0", "Name": "Dance",
             "Vertical1": "Arts & Entertainment", "Vertical2": "Beauty & Fitness"},
        ])
        vocab = Yt8mVocabulary.from_csv(csv_path)
        verts = vocab.get_verticals_for("Dance")
        assert "Arts & Entertainment" in verts
        assert "Beauty & Fitness" in verts
        assert len(verts) == 2

    def test_unknown_vertical_skipped(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [
            {"Index": "0", "Name": "Mystery", "Vertical1": "(Unknown)"},
        ])
        vocab = Yt8mVocabulary.from_csv(csv_path)
        entity = vocab.lookup("Mystery")
        assert entity is not None
        assert entity.verticals == ()

    def test_entity_names_for_vertical(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [
            {"Index": "0", "Name": "Basketball", "Vertical1": "Sports"},
            {"Index": "1", "Name": "Football", "Vertical1": "Sports"},
            {"Index": "2", "Name": "Car", "Vertical1": "Autos & Vehicles"},
        ])
        vocab = Yt8mVocabulary.from_csv(csv_path)
        sports = vocab.entity_names_for_vertical("Sports")
        assert sports == ["Basketball", "Football"]
        autos = vocab.entity_names_for_vertical("Autos & Vehicles")
        assert autos == ["Car"]

    def test_all_entity_names(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [
            {"Index": "0", "Name": "Zebra", "Vertical1": "Pets & Animals"},
            {"Index": "1", "Name": "Apple", "Vertical1": "Food & Drink"},
        ])
        vocab = Yt8mVocabulary.from_csv(csv_path)
        names = vocab.all_entity_names()
        assert names == ["Apple", "Zebra"]

    def test_vertical_vector_for(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [
            {"Index": "0", "Name": "Car", "Vertical1": "Autos & Vehicles"},
        ])
        vocab = Yt8mVocabulary.from_csv(csv_path)
        vec = vocab.vertical_vector_for("Car")
        assert len(vec) == 24
        assert vec[VERTICAL_INDEX["Autos & Vehicles"]] == 1.0
        assert sum(vec) == 1.0

    def test_vertical_vector_multi_vertical(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [
            {"Index": "0", "Name": "Dance",
             "Vertical1": "Arts & Entertainment", "Vertical2": "Beauty & Fitness"},
        ])
        vocab = Yt8mVocabulary.from_csv(csv_path)
        vec = vocab.vertical_vector_for("Dance")
        assert vec[VERTICAL_INDEX["Arts & Entertainment"]] == 1.0
        assert vec[VERTICAL_INDEX["Beauty & Fitness"]] == 1.0
        assert sum(vec) == 2.0

    def test_vertical_vector_unknown_entity(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [
            {"Index": "0", "Name": "Car", "Vertical1": "Autos & Vehicles"},
        ])
        vocab = Yt8mVocabulary.from_csv(csv_path)
        vec = vocab.vertical_vector_for("nonexistent")
        assert vec == [0.0] * 24


class TestGetVocabularySingleton:
    def test_returns_same_instance(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path, [
            {"Index": "0", "Name": "Game", "Vertical1": "Games"},
        ])
        v1 = get_vocabulary(csv_path)
        v2 = get_vocabulary(csv_path)
        assert v1 is v2

    def test_missing_csv_returns_empty(self, tmp_path: Path) -> None:
        vocab = get_vocabulary(tmp_path / "nonexistent.csv")
        assert len(vocab.entities) == 0
        assert vocab.lookup("anything") is None


class TestRealVocabulary:
    """Smoke test against the actual vocabulary CSV."""

    @pytest.fixture
    def real_vocab(self) -> Yt8mVocabulary:
        reset_vocabulary()
        return get_vocabulary()

    def test_loads_curated_vocabulary(self, real_vocab: Yt8mVocabulary) -> None:
        # The curated v2 vocabulary contains ~700 entries. Anchor with a
        # generous lower bound so that adding/removing a handful of entries
        # does not require an unrelated test edit.
        assert len(real_vocab.entities) >= 500

    def test_known_entity_lookup(self, real_vocab: Yt8mVocabulary) -> None:
        # Spot-check one entity per category type to catch breakage that
        # affects the whole vocabulary (case folding, vertical assignment,
        # CSV parsing). These cover the dimensions we actually care about:
        # legacy generic concepts, contemporary brand names, current
        # products, and entities that map to multiple verticals.
        cases = [
            # name,                    expected vertical
            ("BMW",                    "Autos & Vehicles"),       # brand
            ("iPhone 17",              "Computers & Electronics"), # current product
            ("ChatGPT",                "Internet & Telecom"),      # AI
            ("Fortnite",               "Games"),                   # game title
            ("Netflix",                "Arts & Entertainment"),    # streaming
            ("Yoga",                   "Beauty & Fitness"),        # generic concept
            ("Bitcoin",                "Finance"),                 # crypto
            ("Nike",                   "Sports"),                  # apparel/sports
            ("Airbnb",                 "Travel"),                  # service
        ]
        for name, expected_vertical in cases:
            entity = real_vocab.lookup(name)
            assert entity is not None, f"missing entity: {name}"
            assert expected_vertical in entity.verticals, (
                f"{name} expected in {expected_vertical}, got {entity.verticals}"
            )

    def test_all_24_verticals_populated(self, real_vocab: Yt8mVocabulary) -> None:
        for v in VERTICALS:
            entities = real_vocab.get_entities_for_vertical(v)
            assert len(entities) > 0, f"Vertical '{v}' has no entities"