"""Tests for arxiv_digest.ranker module."""

from unittest.mock import MagicMock

import pytest

from arxiv_digest.models import Paper
from arxiv_digest.ranker import PaperRanker, get_llm_for_provider


class TestPaperRanker:
    """Tests for the PaperRanker class."""

    def test_init_defaults(self, mock_llm: MagicMock) -> None:
        """Test ranker initializes with default values."""
        ranker = PaperRanker(mock_llm)
        assert ranker.batch_size == 5
        assert ranker.batch_delay == 1.0

    def test_init_custom_values(self, mock_llm: MagicMock) -> None:
        """Test ranker initializes with custom values."""
        ranker = PaperRanker(mock_llm, batch_size=10, batch_delay=0.5)
        assert ranker.batch_size == 10
        assert ranker.batch_delay == 0.5

    @pytest.mark.asyncio
    async def test_rank_paper_valid_response(
        self, mock_llm: MagicMock, sample_paper: Paper
    ) -> None:
        """Test ranking a paper with valid JSON response."""
        ranker = PaperRanker(mock_llm)

        ranked = await ranker.rank_paper(sample_paper, "machine learning")

        assert ranked.relevance_score == 8.0
        assert "ML" in ranked.relevance_reason or "relevant" in ranked.relevance_reason.lower()

    @pytest.mark.asyncio
    async def test_rank_paper_markdown_response(
        self, mock_llm_markdown_response: MagicMock, sample_paper: Paper
    ) -> None:
        """Test ranking a paper with markdown-wrapped JSON response."""
        ranker = PaperRanker(mock_llm_markdown_response)

        ranked = await ranker.rank_paper(sample_paper, "test interests")

        assert ranked.relevance_score == 7.0
        assert ranked.relevance_reason == "Test reason"

    @pytest.mark.asyncio
    async def test_rank_paper_invalid_json(
        self, mock_llm_invalid_response: MagicMock, sample_paper: Paper
    ) -> None:
        """Test ranking a paper with invalid JSON falls back to default score."""
        ranker = PaperRanker(mock_llm_invalid_response)

        ranked = await ranker.rank_paper(sample_paper, "test interests")

        assert ranked.relevance_score == 5.0
        assert ranked.relevance_reason == "Unable to rank"

    @pytest.mark.asyncio
    async def test_rank_papers_batch(self, mock_llm: MagicMock, sample_papers: list[Paper]) -> None:
        """Test ranking multiple papers in batches."""
        ranker = PaperRanker(mock_llm, batch_size=2, batch_delay=0.01)

        ranked = await ranker.rank_papers(sample_papers, "machine learning")

        assert len(ranked) == len(sample_papers)
        # All papers should have been scored
        assert all(p.relevance_score > 0 for p in ranked)

    @pytest.mark.asyncio
    async def test_rank_papers_sorting(
        self, mock_llm: MagicMock, sample_papers: list[Paper]
    ) -> None:
        """Test that ranked papers are sorted by score descending."""
        ranker = PaperRanker(mock_llm, batch_size=10, batch_delay=0.01)

        ranked = await ranker.rank_papers(sample_papers, "machine learning")

        scores = [p.relevance_score for p in ranked]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rank_papers_empty_list(self, mock_llm: MagicMock) -> None:
        """Test ranking empty list returns empty list."""
        ranker = PaperRanker(mock_llm)

        ranked = await ranker.rank_papers([], "test interests")

        assert ranked == []


class TestGetLLMForProvider:
    """Tests for the get_llm_for_provider function."""

    def test_unknown_provider_raises(self) -> None:
        """Test unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_for_provider("unknown_provider")

    def test_anthropic_missing_key_raises(self) -> None:
        """Test anthropic provider without key raises ValueError."""
        with pytest.raises(ValueError, match="anthropic_api_key is required"):
            get_llm_for_provider("anthropic")

    def test_openai_missing_key_raises(self) -> None:
        """Test openai provider without key raises ValueError."""
        with pytest.raises(ValueError, match="openai_api_key is required"):
            get_llm_for_provider("openai")

    def test_google_missing_key_raises(self) -> None:
        """Test google provider without key raises ValueError."""
        with pytest.raises(ValueError, match="google_api_key is required"):
            get_llm_for_provider("google")
