"""Simple paper memory to track seen papers."""

import json
from pathlib import Path


class PaperMemory:
    """Tracks which papers have been shown to avoid duplicates."""

    def __init__(self, path: str | Path):
        """Initialize memory.

        Args:
            path: Path to JSON file for storing seen paper IDs
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = self._load()

    def _load(self) -> set[str]:
        """Load seen IDs from file."""
        if self.path.exists():
            data = json.loads(self.path.read_text())
            return set(data.get("seen", []))
        return set()

    def _save(self) -> None:
        """Save seen IDs to file."""
        self.path.write_text(json.dumps({"seen": sorted(self._seen)}))

    def record(self, arxiv_id: str) -> None:
        """Record a paper as seen."""
        self._seen.add(arxiv_id)
        self._save()

    def is_seen(self, arxiv_id: str) -> bool:
        """Check if a paper has been seen."""
        return arxiv_id in self._seen

    def filter_unseen(self, arxiv_ids: list[str]) -> set[str]:
        """Return only the IDs that haven't been seen."""
        return set(arxiv_ids) - self._seen

    def clear(self) -> None:
        """Clear all records."""
        self._seen.clear()
        self._save()

    def count(self) -> int:
        """Return number of seen papers."""
        return len(self._seen)
