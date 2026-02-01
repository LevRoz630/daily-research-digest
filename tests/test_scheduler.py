"""Tests for daily_research_digest.scheduler module."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from daily_research_digest.digest import DigestGenerator
from daily_research_digest.models import DigestConfig
from daily_research_digest.scheduler import ArxivScheduler
from daily_research_digest.storage import DigestStorage


class TestArxivScheduler:
    """Tests for the ArxivScheduler class."""

    @pytest.fixture
    def generator(self, temp_storage_dir: Path) -> DigestGenerator:
        """Create a DigestGenerator for testing."""
        storage = DigestStorage(temp_storage_dir)
        return DigestGenerator(storage)

    def test_init_default_hour(self, generator: DigestGenerator) -> None:
        """Test scheduler initializes with default hour."""
        scheduler = ArxivScheduler(generator)
        assert scheduler.schedule_hour == 6
        assert scheduler.is_running is False

    def test_init_custom_hour(self, generator: DigestGenerator) -> None:
        """Test scheduler initializes with custom hour."""
        scheduler = ArxivScheduler(generator, schedule_hour=12)
        assert scheduler.schedule_hour == 12

    @pytest.mark.asyncio
    async def test_start_stop(
        self, generator: DigestGenerator, sample_config: DigestConfig
    ) -> None:
        """Test starting and stopping the scheduler."""
        scheduler = ArxivScheduler(generator)

        assert scheduler.is_running is False

        scheduler.start(sample_config)
        assert scheduler.is_running is True
        assert scheduler._config == sample_config

        scheduler.stop()
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_start_with_schedule_hour_override(
        self, generator: DigestGenerator, sample_config: DigestConfig
    ) -> None:
        """Test starting with schedule_hour override."""
        scheduler = ArxivScheduler(generator, schedule_hour=6)

        scheduler.start(sample_config, schedule_hour=18)
        assert scheduler.schedule_hour == 18

        scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(
        self, generator: DigestGenerator, sample_config: DigestConfig
    ) -> None:
        """Test that calling start multiple times is safe."""
        scheduler = ArxivScheduler(generator)

        scheduler.start(sample_config)
        task1 = scheduler._task

        scheduler.start(sample_config)
        task2 = scheduler._task

        # Should be the same task (not restarted)
        assert task1 is task2

        scheduler.stop()

    def test_seconds_until_next_run_future(self, generator: DigestGenerator) -> None:
        """Test calculation when scheduled time is in the future."""
        scheduler = ArxivScheduler(generator, schedule_hour=23)

        # Mock current time to be early in the day
        with patch("daily_research_digest.scheduler.datetime") as mock_dt:
            mock_now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            seconds = scheduler._seconds_until_next_run()

        # Should be about 13 hours (from 10:00 to 23:00)
        expected_hours = 13
        assert abs(seconds - expected_hours * 3600) < 60  # Allow 1 minute tolerance

    def test_seconds_until_next_run_past(self, generator: DigestGenerator) -> None:
        """Test calculation when scheduled time has passed today."""
        scheduler = ArxivScheduler(generator, schedule_hour=6)

        # Mock current time to be after schedule hour
        with patch("daily_research_digest.scheduler.datetime") as mock_dt:
            mock_now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            seconds = scheduler._seconds_until_next_run()

        # Should be about 20 hours (to next day 6:00)
        expected_hours = 20
        assert abs(seconds - expected_hours * 3600) < 60

    def test_is_running_property(
        self, generator: DigestGenerator, sample_config: DigestConfig
    ) -> None:
        """Test is_running property reflects internal state."""
        scheduler = ArxivScheduler(generator)

        assert scheduler.is_running is False

        scheduler._running = True
        assert scheduler.is_running is True

        scheduler._running = False
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_run_loop_cancellation(
        self, generator: DigestGenerator, sample_config: DigestConfig
    ) -> None:
        """Test that the run loop handles cancellation gracefully."""
        scheduler = ArxivScheduler(generator)

        # Start the scheduler
        scheduler.start(sample_config)

        # Give it a moment to start
        await asyncio.sleep(0.01)

        # Stop should cancel cleanly
        scheduler.stop()

        # Verify it stopped
        assert scheduler.is_running is False
        assert scheduler._task is None
