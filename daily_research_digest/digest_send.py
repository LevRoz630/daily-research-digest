"""CLI module for sending digest emails.

Run with: python -m daily_research_digest.digest_send
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from pathlib import Path

from .config_env import ConfigError, DigestEmailConfig, load_config_from_env, parse_window
from .digest import DigestGenerator
from .digest_renderer import Digest, render_digest
from .digest_state import LocalFileStateBackend, S3StateBackend, compute_digest_id
from .email_provider import EmailSendError, SMTPProvider
from .models import DateFilter, DigestConfig
from .storage import DigestStorage

# Structured logging setup
logger = logging.getLogger("daily_research_digest.send")


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "digest_id"):
            log_data["digest_id"] = record.digest_id
        if hasattr(record, "window_start"):
            log_data["window_start"] = record.window_start
        if hasattr(record, "window_end"):
            log_data["window_end"] = record.window_end
        if hasattr(record, "recipient_count"):
            log_data["recipient_count"] = record.recipient_count
        if hasattr(record, "paper_count"):
            log_data["paper_count"] = record.paper_count
        if hasattr(record, "provider"):
            log_data["provider"] = record.provider
        if hasattr(record, "error"):
            log_data["error"] = record.error

        return json.dumps(log_data)


def setup_logging() -> None:
    """Set up structured JSON logging to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log_info(message: str, **kwargs: Any) -> None:
    """Log info with extra structured fields."""
    record = logger.makeRecord(logger.name, logging.INFO, "", 0, message, (), None)
    for key, value in kwargs.items():
        setattr(record, key, value)
    logger.handle(record)


def log_error(message: str, **kwargs: Any) -> None:
    """Log error with extra structured fields."""
    record = logger.makeRecord(logger.name, logging.ERROR, "", 0, message, (), None)
    for key, value in kwargs.items():
        setattr(record, key, value)
    logger.handle(record)


async def generate_digest_content(
    window_start: datetime,
    window_end: datetime,
    config: DigestEmailConfig,
    save_dir: Path | None = None,
) -> Digest:
    """Generate digest content for the time window.

    Args:
        window_start: Start of time window
        window_end: End of time window
        config: Email configuration with digest settings
        save_dir: Optional directory to save digest JSON

    Returns:
        Digest object with papers and metadata

    Raises:
        Exception: If digest generation fails
    """
    # Build DigestConfig from email config
    digest_config = DigestConfig(
        categories=config.categories,
        interests=config.interests,
        max_papers=config.max_papers,
        top_n=config.top_n,
        llm_provider=config.llm_provider,
        anthropic_api_key=config.anthropic_api_key,
        openai_api_key=config.openai_api_key,
        google_api_key=config.google_api_key,
        date_filter=DateFilter(
            published_after=window_start.strftime("%Y-%m-%d"),
            published_before=window_end.strftime("%Y-%m-%d"),
        ),
    )

    # Use temporary storage for this run
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = DigestStorage(Path(tmpdir))
        generator = DigestGenerator(storage)

        result = await generator.generate(digest_config)

        if result["status"] == "error":
            errors = result.get("errors", ["Unknown error"])
            raise RuntimeError(f"Digest generation failed: {'; '.join(errors)}")

        if result["status"] != "completed":
            raise RuntimeError(f"Unexpected digest status: {result['status']}")

        digest_data = result["digest"]
        papers = digest_data.get("papers", [])

        # Save digest JSON if directory specified
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            date_str = window_end.strftime("%Y-%m-%d")
            digest_file = save_dir / f"{date_str}.json"
            with open(digest_file, "w") as f:
                json.dump(digest_data, f, indent=2)
            log_info(f"Saved digest to {digest_file}")

        # Convert paper dicts back to Paper objects
        from .models import Paper

        paper_objects = [
            Paper(
                arxiv_id=p["arxiv_id"],
                title=p["title"],
                abstract=p["abstract"],
                authors=p["authors"],
                categories=p["categories"],
                published=p["published"],
                updated=p["updated"],
                link=p["link"],
                relevance_score=p.get("relevance_score", 0),
                relevance_reason=p.get("relevance_reason", ""),
            )
            for p in papers
        ]

        return Digest(
            items=paper_objects,
            window_start=window_start,
            window_end=window_end,
            categories=config.categories,
            interests=config.interests,
            total_fetched=digest_data.get("total_papers_fetched", 0),
        )


async def async_main() -> int:
    """Async main entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    setup_logging()

    # Load configuration
    try:
        config = load_config_from_env()
    except ConfigError as e:
        log_error("Configuration error", error=str(e))
        return 1

    # Compute time window
    try:
        tz = ZoneInfo(config.timezone)
    except KeyError:
        log_error("Invalid timezone", error=f"Unknown timezone: {config.timezone}")
        return 1

    now = datetime.now(tz)
    window_delta = parse_window(config.window)
    window_end = now
    window_start = now - window_delta

    # Compute digest ID for idempotency
    digest_id = compute_digest_id(window_start, window_end, config.recipients, config.subject)

    log_info(
        "Starting digest generation",
        digest_id=digest_id,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
        recipient_count=len(config.recipients),
    )

    # Get state backend
    state_backend: LocalFileStateBackend | S3StateBackend
    if config.state_s3_uri:
        state_backend = S3StateBackend(config.state_s3_uri)
    else:
        state_backend = LocalFileStateBackend()

    # Check idempotency
    try:
        if state_backend.already_sent(digest_id):
            log_info("Digest already sent, skipping", digest_id=digest_id)
            return 0
    except NotImplementedError as e:
        log_error("State backend error", error=str(e))
        return 1

    # Check if we should save digest to a directory
    save_dir = None
    save_dir_env = os.environ.get("DIGEST_SAVE_DIR")
    if save_dir_env:
        save_dir = Path(save_dir_env)

    # Generate digest
    try:
        digest = await generate_digest_content(window_start, window_end, config, save_dir)
    except Exception as e:
        log_error("Failed to generate digest", error=str(e))
        return 1

    if not digest.items:
        log_info("No papers in digest, skipping send", digest_id=digest_id)
        # Still mark as "sent" to avoid re-running
        state_backend.mark_sent(digest_id)
        return 0

    # Render email bodies
    text_body, html_body = render_digest(digest)

    # Format subject with date
    subject = config.subject.format(date=now.strftime("%Y-%m-%d"))

    # Send email
    if config.smtp_config is None:
        log_error("SMTP configuration missing")
        return 1

    try:
        provider = SMTPProvider(config.smtp_config)
        provider.send(
            subject=subject,
            from_addr=config.from_addr,
            to_addrs=config.recipients,
            text_body=text_body,
            html_body=html_body,
        )
    except EmailSendError as e:
        log_error("Failed to send email", error=str(e))
        return 1

    # Mark as sent
    state_backend.mark_sent(digest_id)

    log_info(
        "Digest sent successfully",
        digest_id=digest_id,
        recipient_count=len(config.recipients),
        paper_count=len(digest.items),
        provider="smtp",
    )

    return 0


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    import asyncio

    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
