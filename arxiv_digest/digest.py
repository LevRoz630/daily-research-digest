"""Digest generation logic."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .client import ArxivClient
from .models import DigestConfig, DigestState
from .ranker import PaperRanker, get_llm_for_provider
from .sources.huggingface import HuggingFaceClient
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
            # Determine which sources to use
            sources = config.sources or ["arxiv"]
            papers: list = []
            seen_ids: set[str] = set()

            # Fetch from arXiv
            if "arxiv" in sources:
                logger.info(f"Fetching papers from arXiv for categories: {config.categories}")
                arxiv_papers = await self.client.fetch_papers(
                    config.categories, config.max_papers, config.date_filter
                )
                for p in arxiv_papers:
                    if p.arxiv_id not in seen_ids:
                        papers.append(p)
                        seen_ids.add(p.arxiv_id)
                logger.info(f"Fetched {len(arxiv_papers)} papers from arXiv")

            # Fetch from HuggingFace
            if "huggingface" in sources:
                logger.info("Fetching papers from HuggingFace Daily Papers")
                hf_client = HuggingFaceClient()
                hf_papers = await hf_client.fetch_papers(limit=config.max_papers)
                added = 0
                for p in hf_papers:
                    if p.arxiv_id not in seen_ids:
                        papers.append(p)
                        seen_ids.add(p.arxiv_id)
                        added += 1
                logger.info(f"Added {added} unique papers from HuggingFace")

            logger.info(f"Total papers from all sources: {len(papers)}")

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

            # Boost papers from priority authors
            if config.priority_authors:
                priority_lower = [a.lower() for a in config.priority_authors]
                for paper in ranked_papers:
                    paper_authors_lower = [a.lower() for a in paper.authors]
                    if any(
                        priority in author
                        for priority in priority_lower
                        for author in paper_authors_lower
                    ):
                        paper.relevance_score *= config.author_boost
                # Re-sort after boosting
                ranked_papers.sort(key=lambda p: p.relevance_score, reverse=True)

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
