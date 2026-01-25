"""Storage and retrieval of arxiv digests."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DigestStorage:
    """Handles storage and retrieval of arxiv digests."""

    def __init__(self, storage_dir: str | Path):
        """Initialize storage.

        Args:
            storage_dir: Directory to store digest JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_digest(self, digest: dict) -> Path:
        """Save a digest to disk.

        Args:
            digest: Digest dictionary with 'date' key

        Returns:
            Path to saved digest file
        """
        date = digest.get("date")
        if not date:
            raise ValueError("Digest must have a 'date' field")

        digest_file = self.storage_dir / f"{date}.json"
        with open(digest_file, "w", encoding="utf-8") as f:
            json.dump(digest, f, indent=2)

        logger.info(f"Digest saved to {digest_file}")
        return digest_file

    def get_digest(self, date: str | None = None) -> dict | None:
        """Get a digest by date.

        Args:
            date: Date in YYYY-MM-DD format. If None, returns most recent digest.

        Returns:
            Digest dictionary or None if not found
        """
        if date is None:
            # Get the most recent digest
            digests = sorted(self.storage_dir.glob("*.json"), reverse=True)
            if not digests:
                return None
            digest_file = digests[0]
        else:
            digest_file = self.storage_dir / f"{date}.json"

        if not digest_file.exists():
            return None

        with open(digest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_digests(self, limit: int = 30) -> list[str]:
        """List available digest dates.

        Args:
            limit: Maximum number of dates to return

        Returns:
            List of date strings (YYYY-MM-DD) sorted newest first
        """
        digests = sorted(self.storage_dir.glob("*.json"), reverse=True)[:limit]
        return [d.stem for d in digests]

    def delete_digest(self, date: str) -> bool:
        """Delete a digest by date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            True if deleted, False if not found
        """
        digest_file = self.storage_dir / f"{date}.json"
        if digest_file.exists():
            digest_file.unlink()
            logger.info(f"Deleted digest {date}")
            return True
        return False
