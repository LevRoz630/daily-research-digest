"""Tests for daily_research_digest.digest_state module."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from daily_research_digest.digest_state import (
    LocalFileStateBackend,
    S3StateBackend,
    compute_digest_id,
    get_state_backend,
)


class TestComputeDigestId:
    """Tests for compute_digest_id function."""

    def test_stable_id(self) -> None:
        """Test same inputs produce same ID."""
        window_start = datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
        window_end = datetime(2024, 1, 16, 6, 0, 0, tzinfo=timezone.utc)
        recipients = ["a@b.com", "c@d.com"]
        subject = "Test Subject"

        id1 = compute_digest_id(window_start, window_end, recipients, subject)
        id2 = compute_digest_id(window_start, window_end, recipients, subject)

        assert id1 == id2
        assert len(id1) == 16  # First 16 chars of SHA256

    def test_different_window_start(self) -> None:
        """Test different window_start produces different ID."""
        window_end = datetime(2024, 1, 16, 6, 0, 0, tzinfo=timezone.utc)
        recipients = ["a@b.com"]
        subject = "Test"

        id1 = compute_digest_id(
            datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc),
            window_end, recipients, subject
        )
        id2 = compute_digest_id(
            datetime(2024, 1, 14, 6, 0, 0, tzinfo=timezone.utc),
            window_end, recipients, subject
        )

        assert id1 != id2

    def test_different_window_end(self) -> None:
        """Test different window_end produces different ID."""
        window_start = datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
        recipients = ["a@b.com"]
        subject = "Test"

        id1 = compute_digest_id(
            window_start,
            datetime(2024, 1, 16, 6, 0, 0, tzinfo=timezone.utc),
            recipients, subject
        )
        id2 = compute_digest_id(
            window_start,
            datetime(2024, 1, 17, 6, 0, 0, tzinfo=timezone.utc),
            recipients, subject
        )

        assert id1 != id2

    def test_different_recipients(self) -> None:
        """Test different recipients produces different ID."""
        window_start = datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
        window_end = datetime(2024, 1, 16, 6, 0, 0, tzinfo=timezone.utc)
        subject = "Test"

        id1 = compute_digest_id(window_start, window_end, ["a@b.com"], subject)
        id2 = compute_digest_id(window_start, window_end, ["x@y.com"], subject)

        assert id1 != id2

    def test_different_subject(self) -> None:
        """Test different subject produces different ID."""
        window_start = datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
        window_end = datetime(2024, 1, 16, 6, 0, 0, tzinfo=timezone.utc)
        recipients = ["a@b.com"]

        id1 = compute_digest_id(window_start, window_end, recipients, "Subject A")
        id2 = compute_digest_id(window_start, window_end, recipients, "Subject B")

        assert id1 != id2

    def test_recipient_order_insensitive(self) -> None:
        """Test recipient order doesn't affect ID (sorted internally)."""
        window_start = datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
        window_end = datetime(2024, 1, 16, 6, 0, 0, tzinfo=timezone.utc)
        subject = "Test"

        id1 = compute_digest_id(window_start, window_end, ["a@b.com", "c@d.com"], subject)
        id2 = compute_digest_id(window_start, window_end, ["c@d.com", "a@b.com"], subject)

        assert id1 == id2

    def test_recipient_case_insensitive(self) -> None:
        """Test recipient case doesn't affect ID (lowercased internally)."""
        window_start = datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
        window_end = datetime(2024, 1, 16, 6, 0, 0, tzinfo=timezone.utc)
        subject = "Test"

        id1 = compute_digest_id(window_start, window_end, ["A@B.com"], subject)
        id2 = compute_digest_id(window_start, window_end, ["a@b.com"], subject)

        assert id1 == id2


class TestLocalFileStateBackend:
    """Tests for LocalFileStateBackend class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test initialization creates state directory."""
        state_dir = tmp_path / "state"
        assert not state_dir.exists()

        LocalFileStateBackend(state_dir)
        assert state_dir.exists()

    def test_initially_not_sent(self, tmp_path: Path) -> None:
        """Test new digest ID is not marked as sent."""
        backend = LocalFileStateBackend(tmp_path / "state")
        assert backend.already_sent("test-id") is False

    def test_mark_sent(self, tmp_path: Path) -> None:
        """Test marking digest as sent."""
        backend = LocalFileStateBackend(tmp_path / "state")

        assert backend.already_sent("test-id") is False
        backend.mark_sent("test-id")
        assert backend.already_sent("test-id") is True

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        """Test state persists across backend instances."""
        state_dir = tmp_path / "state"

        # First instance marks as sent
        backend1 = LocalFileStateBackend(state_dir)
        backend1.mark_sent("persistent-id")

        # New instance should see the state
        backend2 = LocalFileStateBackend(state_dir)
        assert backend2.already_sent("persistent-id") is True

    def test_multiple_ids(self, tmp_path: Path) -> None:
        """Test tracking multiple digest IDs."""
        backend = LocalFileStateBackend(tmp_path / "state")

        backend.mark_sent("id-1")
        backend.mark_sent("id-2")

        assert backend.already_sent("id-1") is True
        assert backend.already_sent("id-2") is True
        assert backend.already_sent("id-3") is False

    def test_mark_sent_idempotent(self, tmp_path: Path) -> None:
        """Test marking same ID multiple times doesn't duplicate."""
        backend = LocalFileStateBackend(tmp_path / "state")

        backend.mark_sent("test-id")
        backend.mark_sent("test-id")
        backend.mark_sent("test-id")

        assert backend.list_sent().count("test-id") == 1

    def test_clear(self, tmp_path: Path) -> None:
        """Test clearing all sent markers."""
        backend = LocalFileStateBackend(tmp_path / "state")

        backend.mark_sent("id-1")
        backend.mark_sent("id-2")
        assert backend.already_sent("id-1") is True

        backend.clear()
        assert backend.already_sent("id-1") is False
        assert backend.already_sent("id-2") is False

    def test_list_sent(self, tmp_path: Path) -> None:
        """Test listing all sent IDs."""
        backend = LocalFileStateBackend(tmp_path / "state")

        backend.mark_sent("id-1")
        backend.mark_sent("id-2")
        backend.mark_sent("id-3")

        sent = backend.list_sent()
        assert set(sent) == {"id-1", "id-2", "id-3"}

    def test_handles_corrupted_file(self, tmp_path: Path) -> None:
        """Test handles corrupted state file gracefully."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        state_file = state_dir / "sent.json"
        state_file.write_text("not valid json {{{")

        backend = LocalFileStateBackend(state_dir)
        # Should not raise, treats as empty state
        assert backend.already_sent("test") is False


class TestS3StateBackend:
    """Tests for S3StateBackend class."""

    def test_already_sent_raises_not_implemented(self) -> None:
        """Test already_sent raises NotImplementedError."""
        backend = S3StateBackend("s3://bucket/state.json")
        with pytest.raises(NotImplementedError, match="S3 state backend"):
            backend.already_sent("test-id")

    def test_mark_sent_raises_not_implemented(self) -> None:
        """Test mark_sent raises NotImplementedError."""
        backend = S3StateBackend("s3://bucket/state.json")
        with pytest.raises(NotImplementedError, match="S3 state backend"):
            backend.mark_sent("test-id")


class TestGetStateBackend:
    """Tests for get_state_backend function."""

    def test_returns_local_by_default(self) -> None:
        """Test returns LocalFileStateBackend when no S3 URI."""
        backend = get_state_backend()
        assert isinstance(backend, LocalFileStateBackend)

    def test_returns_local_when_none(self) -> None:
        """Test returns LocalFileStateBackend when S3 URI is None."""
        backend = get_state_backend(s3_uri=None)
        assert isinstance(backend, LocalFileStateBackend)

    def test_returns_s3_when_uri_provided(self) -> None:
        """Test returns S3StateBackend when URI provided."""
        backend = get_state_backend(s3_uri="s3://bucket/state.json")
        assert isinstance(backend, S3StateBackend)
