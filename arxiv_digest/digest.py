"""Digest generation logic."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .client import ArxivClient
from .models import DigestConfig, DigestState
from .ranker import PaperRanker, get_llm_for_provider
from .storage import DigestStorage

if TYPE_CHECKING:
    from .memory import PaperMemory

logger = logging.getLogger(__name__)


class DigestGenerator:
    """Generates arXiv paper digests."""

    def __init__(
        self,
        storage: DigestStorage,
        client: ArxivClient | None = None,
        memory: PaperMemory | None = None,
    ):
        """Initialize generator.

        Args:
            storage: DigestStorage instance for saving/loading digests
            client: ArxivClient instance (creates default if None)
            memory: PaperMemory instance for tracking seen papers (optional)
        """
        self.storage = storage
        self.client = client or ArxivClient()
        self.memory = memory
        self.state = DigestState()

    async def generate(self, config: DigestConfig) -> dict:
        """Generate a digest.

        Args:
            config: DigestConfig with categories, interests, etc.

        Returns:
            Dictionary with status and digest data
        """
        if self.state.is_generating:
            return {"status": "already_generating"}

        self.state.is_generating = True
        self.state.errors = []

        try:
            # Fetch papers
            logger.info(f"Fetching papers from arXiv for categories: {config.categories}")
            papers = await self.client.fetch_papers(
                config.categories, config.max_papers, config.date_filter
            )
            logger.info(f"Fetched {len(papers)} papers")

            if not papers:
                self.state.errors.append("No papers fetched from arXiv")
                return {"status": "error", "errors": self.state.errors}

            # Filter out previously seen papers if memory is enabled
            if self.memory and config.exclude_seen:
                arxiv_ids = [p.arxiv_id for p in papers]
                unseen_ids = self.memory.filter_unseen(arxiv_ids)
                original_count = len(papers)
                papers = [p for p in papers if p.arxiv_id in unseen_ids]
                excluded = original_count - len(papers)
                logger.info(f"Filtered to {len(papers)} unseen papers (excluded {excluded} seen)")

                if not papers:
                    self.state.errors.append("No unseen papers after filtering")
                    return {"status": "error", "errors": self.state.errors}

            # Rank papers
            logger.info(f"Ranking papers against interests: {config.interests[:50]}...")
            llm = get_llm_for_provider(
                config.llm_provider,
                anthropic_api_key=config.anthropic_api_key,
                openai_api_key=config.openai_api_key,
                google_api_key=config.google_api_key,
            )
            ranker = PaperRanker(llm)
            ranked_papers = await ranker.rank_papers(papers, config.interests)

            # Take top N papers
            top_papers = ranked_papers[: config.top_n]

            # Build digest
            now = datetime.now(timezone.utc)
            digest = {
                "date": now.strftime("%Y-%m-%d"),
                "generated_at": now.isoformat(),
                "categories": config.categories,
                "interests": config.interests,
                "total_papers_fetched": len(papers),
                "papers": [p.to_dict() for p in top_papers],
            }

            # Record papers in memory
            if self.memory:
                for paper in top_papers:
                    self.memory.record(paper.arxiv_id)
                logger.info(f"Recorded {len(top_papers)} papers in memory")

            # Save digest
            self.storage.save_digest(digest)
            self.state.last_digest = now

            return {"status": "completed", "digest": digest}

        except Exception as e:
            logger.error(f"Digest generation error: {e}")
            self.state.errors.append(str(e))
            return {"status": "error", "errors": self.state.errors}
        finally:
            self.state.is_generating = False

    def get_state(self) -> DigestState:
        """Get current digest generation state."""
        return self.state
