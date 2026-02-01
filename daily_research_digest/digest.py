"""Digest generation logic."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .models import DigestConfig, DigestState, Paper
from .quality import compute_quality_scores
from .ranker import PaperRanker, get_llm_for_provider
from .sources.semantic_scholar import SemanticScholarClient
from .storage import DigestStorage

if TYPE_CHECKING:
    from .memory import PaperMemory

logger = logging.getLogger(__name__)


class DigestGenerator:
    """Generates research paper digests."""

    def __init__(
        self,
        storage: DigestStorage,
        memory: PaperMemory | None = None,
    ):
        """Initialize generator.

        Args:
            storage: DigestStorage instance for saving/loading digests
            memory: PaperMemory instance for tracking seen papers (optional)
        """
        self.storage = storage
        self.memory = memory
        self.state = DigestState()

    def _add_unique_papers(
        self, papers: list[Paper], new_papers: list[Paper], seen_ids: set[str]
    ) -> int:
        """Add papers not already in seen_ids, merging quality data. Returns count added."""
        added = 0
        for p in new_papers:
            if p.arxiv_id not in seen_ids:
                papers.append(p)
                seen_ids.add(p.arxiv_id)
                added += 1
            else:
                # Merge quality data into existing paper
                existing = next((ep for ep in papers if ep.arxiv_id == p.arxiv_id), None)
                if existing:
                    if p.author_h_indices and not existing.author_h_indices:
                        existing.author_h_indices = p.author_h_indices
                    if p.huggingface_upvotes and not existing.huggingface_upvotes:
                        existing.huggingface_upvotes = p.huggingface_upvotes
        return added

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
            # New pipeline: Semantic Scholar + HuggingFace only (no arXiv)
            papers: list[Paper] = []
            seen_ids: set[str] = set()

            # Fetch from Semantic Scholar (with h-index)
            logger.info("Fetching papers from Semantic Scholar")
            ss_client = SemanticScholarClient(api_key=config.semantic_scholar_api_key)
            ss_papers = await ss_client.fetch_papers(
                query=config.interests,
                limit=config.max_papers,
                fields_of_study=["Computer Science"],
            )
            self._add_unique_papers(papers, ss_papers, seen_ids)
            logger.info(f"Fetched {len(ss_papers)} papers from Semantic Scholar")

            logger.info(f"Total papers: {len(papers)}")

            if not papers:
                self.state.errors.append("No papers fetched from any source")
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

            # Compute quality scores (incorporates h-index and upvotes)
            logger.info("Computing quality scores")
            compute_quality_scores(ranked_papers)

            # Sort by quality score (final ranking)
            ranked_papers.sort(key=lambda p: p.quality_score or 0, reverse=True)

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
                self.memory.record_many([p.arxiv_id for p in top_papers])
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
