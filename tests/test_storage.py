"""Tests for daily_research_digest.storage module."""

import json
from pathlib import Path

import pytest

from daily_research_digest.storage import DigestStorage


class TestDigestStorage:
    """Tests for the DigestStorage class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test storage directory is created on init."""
        storage_dir = tmp_path / "new_digests"
        assert not storage_dir.exists()

        DigestStorage(storage_dir)
        assert storage_dir.exists()

    def test_save_digest(self, temp_storage_dir: Path, sample_digest: dict) -> None:
        """Test saving a digest creates the correct file."""
        storage = DigestStorage(temp_storage_dir)
        filepath = storage.save_digest(sample_digest)

        assert filepath.exists()
        assert filepath.name == "2024-01-15.json"

        # Verify content
        with open(filepath) as f:
            saved = json.load(f)
        assert saved["date"] == sample_digest["date"]
        assert saved["interests"] == sample_digest["interests"]

    def test_save_digest_without_date_raises(self, temp_storage_dir: Path) -> None:
        """Test saving digest without date field raises ValueError."""
        storage = DigestStorage(temp_storage_dir)
        invalid_digest = {"papers": []}

        with pytest.raises(ValueError, match="must have a 'date' field"):
            storage.save_digest(invalid_digest)

    def test_get_digest_by_date(self, temp_storage_dir: Path, sample_digest: dict) -> None:
        """Test retrieving a digest by specific date."""
        storage = DigestStorage(temp_storage_dir)
        storage.save_digest(sample_digest)

        retrieved = storage.get_digest("2024-01-15")
        assert retrieved is not None
        assert retrieved["date"] == "2024-01-15"
        assert retrieved["interests"] == sample_digest["interests"]

    def test_get_digest_most_recent(self, temp_storage_dir: Path) -> None:
        """Test retrieving the most recent digest."""
        storage = DigestStorage(temp_storage_dir)

        # Save multiple digests
        storage.save_digest({"date": "2024-01-10", "papers": []})
        storage.save_digest({"date": "2024-01-15", "papers": []})
        storage.save_digest({"date": "2024-01-12", "papers": []})

        # Most recent by filename should be 2024-01-15
        retrieved = storage.get_digest()
        assert retrieved is not None
        assert retrieved["date"] == "2024-01-15"

    def test_get_digest_not_found(self, temp_storage_dir: Path) -> None:
        """Test retrieving non-existent digest returns None."""
        storage = DigestStorage(temp_storage_dir)
        result = storage.get_digest("2099-12-31")
        assert result is None

    def test_get_digest_empty_storage(self, temp_storage_dir: Path) -> None:
        """Test get_digest with no date on empty storage returns None."""
        storage = DigestStorage(temp_storage_dir)
        result = storage.get_digest()
        assert result is None

    def test_list_digests(self, temp_storage_dir: Path) -> None:
        """Test listing available digests."""
        storage = DigestStorage(temp_storage_dir)

        # Save some digests
        storage.save_digest({"date": "2024-01-10", "papers": []})
        storage.save_digest({"date": "2024-01-15", "papers": []})
        storage.save_digest({"date": "2024-01-12", "papers": []})

        dates = storage.list_digests()
        assert len(dates) == 3
        # Should be sorted newest first
        assert dates[0] == "2024-01-15"
        assert dates[1] == "2024-01-12"
        assert dates[2] == "2024-01-10"

    def test_list_digests_with_limit(self, temp_storage_dir: Path) -> None:
        """Test listing digests with a limit."""
        storage = DigestStorage(temp_storage_dir)

        for day in range(1, 11):
            storage.save_digest({"date": f"2024-01-{day:02d}", "papers": []})

        dates = storage.list_digests(limit=5)
        assert len(dates) == 5
        assert dates[0] == "2024-01-10"

    def test_list_digests_empty(self, temp_storage_dir: Path) -> None:
        """Test listing digests on empty storage."""
        storage = DigestStorage(temp_storage_dir)
        dates = storage.list_digests()
        assert dates == []

    def test_delete_digest(self, temp_storage_dir: Path, sample_digest: dict) -> None:
        """Test deleting a digest."""
        storage = DigestStorage(temp_storage_dir)
        storage.save_digest(sample_digest)

        # Verify it exists
        assert storage.get_digest("2024-01-15") is not None

        # Delete it
        result = storage.delete_digest("2024-01-15")
        assert result is True

        # Verify it's gone
        assert storage.get_digest("2024-01-15") is None

    def test_delete_digest_not_found(self, temp_storage_dir: Path) -> None:
        """Test deleting non-existent digest returns False."""
        storage = DigestStorage(temp_storage_dir)
        result = storage.delete_digest("2099-12-31")
        assert result is False
