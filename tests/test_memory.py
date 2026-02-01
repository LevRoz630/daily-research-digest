"""Tests for daily_research_digest.memory module."""

from pathlib import Path

import pytest

from daily_research_digest.memory import PaperMemory


class TestPaperMemory:
    """Tests for the PaperMemory class."""

    @pytest.fixture
    def memory(self, tmp_path: Path) -> PaperMemory:
        """Create a PaperMemory instance with temp file."""
        return PaperMemory(tmp_path / "memory.json")

    def test_init_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test parent directory is created on init."""
        path = tmp_path / "subdir" / "memory.json"
        PaperMemory(path)
        assert path.parent.exists()

    def test_record_and_is_seen(self, memory: PaperMemory) -> None:
        """Test recording and checking papers."""
        assert memory.is_seen("2401.00001") is False

        memory.record("2401.00001")
        assert memory.is_seen("2401.00001") is True

    def test_filter_unseen(self, memory: PaperMemory) -> None:
        """Test filtering to unseen papers."""
        memory.record("2401.00001")
        memory.record("2401.00002")

        ids = ["2401.00001", "2401.00002", "2401.00003", "2401.00004"]
        unseen = memory.filter_unseen(ids)

        assert unseen == {"2401.00003", "2401.00004"}

    def test_filter_unseen_empty_input(self, memory: PaperMemory) -> None:
        """Test filter_unseen with empty list."""
        unseen = memory.filter_unseen([])
        assert unseen == set()

    def test_filter_unseen_all_seen(self, memory: PaperMemory) -> None:
        """Test when all papers are already seen."""
        memory.record("2401.00001")
        memory.record("2401.00002")

        unseen = memory.filter_unseen(["2401.00001", "2401.00002"])
        assert unseen == set()

    def test_clear(self, memory: PaperMemory) -> None:
        """Test clearing all records."""
        memory.record("2401.00001")
        memory.record("2401.00002")

        memory.clear()

        assert memory.count() == 0
        assert memory.is_seen("2401.00001") is False

    def test_count(self, memory: PaperMemory) -> None:
        """Test counting seen papers."""
        assert memory.count() == 0

        memory.record("2401.00001")
        assert memory.count() == 1

        memory.record("2401.00002")
        assert memory.count() == 2

    def test_persistence(self, tmp_path: Path) -> None:
        """Test that data persists across instances."""
        path = tmp_path / "memory.json"

        # First instance
        memory1 = PaperMemory(path)
        memory1.record("2401.00001")
        memory1.record("2401.00002")

        # Second instance loads the same data
        memory2 = PaperMemory(path)
        assert memory2.is_seen("2401.00001") is True
        assert memory2.is_seen("2401.00002") is True
        assert memory2.count() == 2

    def test_duplicate_record(self, memory: PaperMemory) -> None:
        """Test that recording same ID twice doesn't duplicate."""
        memory.record("2401.00001")
        memory.record("2401.00001")

        assert memory.count() == 1
