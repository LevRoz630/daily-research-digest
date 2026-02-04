"""Data models for daily-research-digest."""

from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class DateFilter:
    """Date filtering options for papers.

    Attributes:
        days_back: Only include papers published within the last N days.
        published_after: Only include papers published after this date (YYYY-MM-DD).
        published_before: Only include papers published before this date (YYYY-MM-DD).
    """

    days_back: int | None = None
    published_after: str | None = None
    published_before: str | None = None


@dataclass
class Paper:
    """Represents a research paper."""

    arxiv_id: str  # Paper identifier (kept for compatibility)
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    link: str
    relevance_score: float = 0.0
    relevance_reason: str = ""
    author_h_indices: list[int] | None = None
    quality_score: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DigestState:
    """Tracks digest generation state."""

    last_digest: datetime | None = None
    is_generating: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class DigestConfig:
    """Configuration for digest generation."""

    categories: list[str]
    interests: str
    max_papers: int = 50
    top_n: int = 10
    date_filter: DateFilter | None = None
    exclude_seen: bool = True
    priority_authors: list[str] | None = None
    author_boost: float = 1.5
    sources: list[str] | None = None  # Currently only ["semantic_scholar"] is supported
    semantic_scholar_api_key: str | None = None
    llm_provider: str = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
