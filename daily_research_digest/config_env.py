"""Environment variable configuration for digest email."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


@dataclass
class SMTPConfig:
    """SMTP server configuration."""

    host: str
    port: int = 587
    user: str | None = None
    password: str | None = None
    use_tls: bool = True


@dataclass
class DigestEmailConfig:
    """Configuration for digest email sending."""

    # Email settings
    recipients: list[str]
    subject: str = "Daily Research Digest - {date}"
    from_addr: str = "noreply@example.com"

    # Time window settings
    timezone: str = "UTC"
    window: str = "24h"

    # Digest content settings
    categories: list[str] = field(default_factory=list)
    interests: str = ""
    max_papers: int = 50
    top_n: int = 10

    # LLM settings
    llm_provider: str = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None

    # SMTP settings
    smtp_config: SMTPConfig | None = None

    # State backend
    state_s3_uri: str | None = None


def parse_window(window_str: str) -> timedelta:
    """Parse time window string to timedelta.

    Args:
        window_str: Time window like "24h", "1d", "48h", "7d"

    Returns:
        Corresponding timedelta

    Raises:
        ValueError: If format is invalid
    """
    window_str = window_str.strip().lower()

    if window_str.endswith("h"):
        try:
            hours = int(window_str[:-1])
            return timedelta(hours=hours)
        except ValueError:
            pass
    elif window_str.endswith("d"):
        try:
            days = int(window_str[:-1])
            return timedelta(days=days)
        except ValueError:
            pass

    raise ValueError(f"Invalid window format: '{window_str}'. Use format like '24h' or '1d'")


def parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower() in ("true", "1", "yes", "on")


def parse_list(value: str) -> list[str]:
    """Parse comma-separated list, stripping whitespace."""
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config_from_env() -> DigestEmailConfig:
    """Load digest email configuration from environment variables.

    Required environment variables:
        DIGEST_RECIPIENTS: Comma-separated email addresses
        DIGEST_CATEGORIES: Comma-separated arXiv categories
        DIGEST_INTERESTS: Research interests description

    Optional environment variables:
        DIGEST_SUBJECT: Email subject template (default: "Daily Research Digest - {date}")
        DIGEST_FROM: Sender email address
        DIGEST_TZ: Timezone (default: "UTC")
        DIGEST_WINDOW: Time window (default: "24h")
        DIGEST_MAX_PAPERS: Max papers to fetch (default: 50)
        DIGEST_TOP_N: Top papers in digest (default: 10)
        LLM_PROVIDER: LLM provider (default: "anthropic")
        ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY: API keys
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_TLS: SMTP config
        DIGEST_STATE_S3_URI: S3 URI for state backend (optional)

    Returns:
        DigestEmailConfig instance

    Raises:
        ConfigError: If required variables are missing or invalid
    """
    errors: list[str] = []

    # Required: recipients
    recipients_str = os.environ.get("DIGEST_RECIPIENTS", "")
    recipients = parse_list(recipients_str)
    if not recipients:
        errors.append("DIGEST_RECIPIENTS is required (comma-separated emails)")

    # Required: categories
    categories_str = os.environ.get("DIGEST_CATEGORIES", "")
    categories = parse_list(categories_str)
    if not categories:
        errors.append("DIGEST_CATEGORIES is required (comma-separated arXiv categories)")

    # Required: interests
    interests = os.environ.get("DIGEST_INTERESTS", "").strip()
    if not interests:
        errors.append("DIGEST_INTERESTS is required (research interests description)")

    # Optional settings with defaults
    subject = os.environ.get("DIGEST_SUBJECT", "Daily Research Digest - {date}")
    from_addr = os.environ.get("DIGEST_FROM", "noreply@example.com")
    timezone = os.environ.get("DIGEST_TZ", "UTC")
    window = os.environ.get("DIGEST_WINDOW", "24h")

    # Validate window format
    try:
        parse_window(window)
    except ValueError as e:
        errors.append(str(e))

    # Numeric settings
    try:
        max_papers = int(os.environ.get("DIGEST_MAX_PAPERS", "50"))
    except ValueError:
        errors.append("DIGEST_MAX_PAPERS must be an integer")
        max_papers = 50

    try:
        top_n = int(os.environ.get("DIGEST_TOP_N", "10"))
    except ValueError:
        errors.append("DIGEST_TOP_N must be an integer")
        top_n = 10

    # LLM settings
    llm_provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY")

    # Validate API key for chosen provider
    if llm_provider == "anthropic" and not anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
    elif llm_provider == "openai" and not openai_api_key:
        errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
    elif llm_provider == "google" and not google_api_key:
        errors.append("GOOGLE_API_KEY is required when LLM_PROVIDER=google")

    # SMTP settings
    smtp_config = None
    smtp_host = os.environ.get("SMTP_HOST")
    if smtp_host:
        try:
            smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        except ValueError:
            errors.append("SMTP_PORT must be an integer")
            smtp_port = 587

        smtp_config = SMTPConfig(
            host=smtp_host,
            port=smtp_port,
            user=os.environ.get("SMTP_USER"),
            password=os.environ.get("SMTP_PASS"),
            use_tls=parse_bool(os.environ.get("SMTP_TLS", "true")),
        )
    else:
        errors.append("SMTP_HOST is required for sending email")

    # Optional S3 state backend
    state_s3_uri = os.environ.get("DIGEST_STATE_S3_URI")

    if errors:
        raise ConfigError("Configuration errors:\n- " + "\n- ".join(errors))

    return DigestEmailConfig(
        recipients=recipients,
        subject=subject,
        from_addr=from_addr,
        timezone=timezone,
        window=window,
        categories=categories,
        interests=interests,
        max_papers=max_papers,
        top_n=top_n,
        llm_provider=llm_provider,
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
        google_api_key=google_api_key,
        smtp_config=smtp_config,
        state_s3_uri=state_s3_uri,
    )
