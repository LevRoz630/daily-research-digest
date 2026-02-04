"""Background scheduler for digest generation."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from .digest import DigestGenerator
from .models import DigestConfig

logger = logging.getLogger(__name__)


class DigestScheduler:
    """Background scheduler for daily digest generation."""

    def __init__(self, generator: DigestGenerator, schedule_hour: int = 6):
        """Initialize scheduler.

        Args:
            generator: DigestGenerator instance
            schedule_hour: UTC hour (0-23) to run daily digest
        """
        self.generator = generator
        self.schedule_hour = schedule_hour
        self._task: asyncio.Task | None = None
        self._running = False
        self._config: DigestConfig | None = None

    def start(self, config: DigestConfig, schedule_hour: int | None = None) -> None:
        """Start the background digest scheduler.

        Args:
            config: DigestConfig to use for scheduled runs
            schedule_hour: Optional UTC hour override
        """
        if self._running:
            return

        if schedule_hour is not None:
            self.schedule_hour = schedule_hour

        self._config = config
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Digest scheduler started (daily at {self.schedule_hour}:00 UTC)")

    def stop(self) -> None:
        """Stop the background scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Digest scheduler stopped")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def _seconds_until_next_run(self) -> float:
        """Calculate seconds until next scheduled run."""
        now = datetime.now(timezone.utc)
        target = now.replace(hour=self.schedule_hour, minute=0, second=0, microsecond=0)

        if now >= target:
            # Already past today's schedule, run tomorrow
            target += timedelta(days=1)

        return (target - now).total_seconds()

    async def _run_loop(self) -> None:
        """Background loop for scheduled digest generation."""
        while self._running:
            try:
                # Wait until next scheduled time
                wait_seconds = self._seconds_until_next_run()
                logger.info(f"Digest scheduler: next run in {wait_seconds / 3600:.1f} hours")
                await asyncio.sleep(wait_seconds)

                if self._running and self._config:
                    logger.info("Running scheduled digest generation...")
                    result = await self.generator.generate(self._config)
                    logger.info(f"Scheduled digest completed: {result.get('status')}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Digest scheduler error: {e}")
                # Sleep a bit before retrying on error
                await asyncio.sleep(3600)
