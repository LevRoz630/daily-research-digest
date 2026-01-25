"""Data models for arxiv-digest."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    """Represents an arXiv paper."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    link: str
    relevance_score: float = 0.0
    relevance_reason: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published,
            "updated": self.updated,
            "link": self.link,
            "relevance_score": self.relevance_score,
            "relevance_reason": self.relevance_reason,
        }


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
    llm_provider: str = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
