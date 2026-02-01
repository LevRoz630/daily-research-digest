"""Idempotency state tracking for digest email sending."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Protocol


def compute_digest_id(
    window_start: datetime,
    window_end: datetime,
    recipients: list[str],
    subject_template: str,
) -> str:
    """Compute a stable digest ID from parameters.

    The digest ID is a SHA256 hash of the normalized inputs, ensuring
    the same parameters always produce the same ID.

    Args:
        window_start: Start of the digest time window
        window_end: End of the digest time window
        recipients: List of recipient email addresses
        subject_template: Email subject template

    Returns:
        Hexadecimal digest ID (first 16 chars of SHA256)
    """
    # Normalize inputs for consistent hashing
    normalized = {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "recipients": sorted(r.lower().strip() for r in recipients),
        "subject_template": subject_template.strip(),
    }

    # Create deterministic JSON string
    content = json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    # Hash and return first 16 chars
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class DigestStateBackend(Protocol):
    """Protocol for digest state persistence backends."""

    def already_sent(self, digest_id: str) -> bool:
        """Check if a digest has already been sent.

        Args:
            digest_id: The digest ID to check

        Returns:
            True if already sent, False otherwise
        """
        ...

    def mark_sent(self, digest_id: str) -> None:
        """Mark a digest as sent.

        Args:
            digest_id: The digest ID to mark as sent
        """
        ...


class LocalFileStateBackend:
    """Store sent markers in a local JSON file.

    State is stored in .digest_state/sent.json by default.
    """

    def __init__(self, state_dir: Path | str = ".digest_state") -> None:
        """Initialize local file state backend.

        Args:
            state_dir: Directory to store state file
        """
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "sent.json"
        self._ensure_state_dir()

    def _ensure_state_dir(self) -> None:
        """Create state directory if it doesn't exist."""
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> dict[str, list[str]]:
        """Load state from file."""
        if not self.state_file.exists():
            return {"sent": []}

        try:
            data: dict[str, list[str]] = json.loads(self.state_file.read_text())
            return data
        except (json.JSONDecodeError, OSError):
            return {"sent": []}

    def _save_state(self, state: dict[str, list[str]]) -> None:
        """Save state to file."""
        self.state_file.write_text(json.dumps(state, indent=2))

    def already_sent(self, digest_id: str) -> bool:
        """Check if a digest has already been sent.

        Args:
            digest_id: The digest ID to check

        Returns:
            True if already sent, False otherwise
        """
        state = self._load_state()
        return digest_id in state.get("sent", [])

    def mark_sent(self, digest_id: str) -> None:
        """Mark a digest as sent.

        Args:
            digest_id: The digest ID to mark as sent
        """
        state = self._load_state()
        sent_list = state.get("sent", [])

        if digest_id not in sent_list:
            sent_list.append(digest_id)
            state["sent"] = sent_list
            self._save_state(state)

    def clear(self) -> None:
        """Clear all sent markers (useful for testing)."""
        self._save_state({"sent": []})

    def list_sent(self) -> list[str]:
        """List all sent digest IDs.

        Returns:
            List of digest IDs that have been marked as sent
        """
        state = self._load_state()
        return list(state.get("sent", []))


class S3StateBackend:
    """Stub for S3-based state persistence.

    This is a placeholder for future S3 support.
    """

    def __init__(self, s3_uri: str) -> None:
        """Initialize S3 state backend.

        Args:
            s3_uri: S3 URI like s3://bucket/path/to/state.json
        """
        self.s3_uri = s3_uri

    def already_sent(self, digest_id: str) -> bool:
        """Check if a digest has already been sent.

        Raises:
            NotImplementedError: S3 backend not yet implemented
        """
        raise NotImplementedError(
            "S3 state backend not yet implemented. "
            "Use local file backend or implement S3 support."
        )

    def mark_sent(self, digest_id: str) -> None:
        """Mark a digest as sent.

        Raises:
            NotImplementedError: S3 backend not yet implemented
        """
        raise NotImplementedError(
            "S3 state backend not yet implemented. "
            "Use local file backend or implement S3 support."
        )


def get_state_backend(s3_uri: str | None = None) -> DigestStateBackend:
    """Get the appropriate state backend based on configuration.

    Args:
        s3_uri: Optional S3 URI. If provided, returns S3 backend (stub).
                If None, returns local file backend.

    Returns:
        DigestStateBackend instance
    """
    if s3_uri:
        return S3StateBackend(s3_uri)
    return LocalFileStateBackend()
