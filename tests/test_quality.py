"""Tests for daily_research_digest.quality module."""

import pytest

from daily_research_digest.models import Paper
from daily_research_digest.quality import compute_quality_score, compute_quality_scores


@pytest.fixture
def sample_paper() -> Paper:
    """Create a sample paper for testing."""
    return Paper(
        arxiv_id="2401.00001",
        title="Test Paper",
        abstract="This is a test abstract.",
        authors=["Author One", "Author Two"],
        categories=["cs.AI"],
        published="2024-01-01",
        updated="2024-01-01",
        link="https://arxiv.org/abs/2401.00001",
        relevance_score=7.0,
    )


class TestComputeQualityScore:
    """Tests for compute_quality_score function."""

    def test_no_quality_signals(self, sample_paper: Paper) -> None:
        """Test score with no h-index or upvotes."""
        score = compute_quality_score(sample_paper)
        # Should equal base relevance score (no boost)
        assert score == 7.0

    def test_with_h_index(self, sample_paper: Paper) -> None:
        """Test score with h-index."""
        sample_paper.author_h_indices = [50, 50]  # avg 50, normalized to 0.5
        score = compute_quality_score(sample_paper, max_h_index=100.0)
        # boost = 0.15 * 0.5 = 0.075
        # final = 7 * (1 + 0.075) = 7.525
        assert score == pytest.approx(7.525)

    def test_with_max_h_index(self, sample_paper: Paper) -> None:
        """Test score with max h-index."""
        sample_paper.author_h_indices = [100]  # normalized to 1.0
        score = compute_quality_score(sample_paper, max_h_index=100.0)
        # boost = 0.15 * 1.0 = 0.15
        # final = 7 * (1 + 0.15) = 8.05
        assert score == pytest.approx(8.05)

    def test_max_boost_is_15_percent(self, sample_paper: Paper) -> None:
        """Test that max boost is 15%."""
        sample_paper.relevance_score = 10.0
        sample_paper.author_h_indices = [200]  # > max, capped to 1.0
        score = compute_quality_score(sample_paper, max_h_index=100.0)
        # boost capped at 0.15
        # final = 10 * 1.15 = 11.5
        assert score == pytest.approx(11.5)


class TestComputeQualityScores:
    """Tests for compute_quality_scores function."""

    def test_empty_list(self) -> None:
        """Test with empty list."""
        result = compute_quality_scores([])
        assert result == []

    def test_populates_quality_score(self, sample_paper: Paper) -> None:
        """Test that quality_score is populated on each paper."""
        sample_paper.author_h_indices = [50]
        papers = compute_quality_scores([sample_paper])
        assert papers[0].quality_score is not None
        assert papers[0].quality_score > sample_paper.relevance_score

    def test_auto_detects_max_values(self) -> None:
        """Test that max values are auto-detected from papers."""
        papers = [
            Paper(
                arxiv_id="1",
                title="Paper 1",
                abstract="",
                authors=["A"],
                categories=[],
                published="",
                updated="",
                link="",
                relevance_score=5.0,
                author_h_indices=[100],
            ),
            Paper(
                arxiv_id="2",
                title="Paper 2",
                abstract="",
                authors=["B"],
                categories=[],
                published="",
                updated="",
                link="",
                relevance_score=5.0,
                author_h_indices=[50],
            ),
        ]
        compute_quality_scores(papers)
        # Paper 1 has h-index 100 (max), normalized to 1.0
        # Paper 2 has h-index 50, normalized to 0.5
        # Paper 1 boost = 0.15 * 1.0 = 0.15, final = 5 * 1.15 = 5.75
        # Paper 2 boost = 0.15 * 0.5 = 0.075, final = 5 * 1.075 = 5.375
        assert papers[0].quality_score == pytest.approx(5.75)
        assert papers[1].quality_score == pytest.approx(5.375)

    def test_handles_missing_signals(self) -> None:
        """Test that papers with missing signals still work."""
        papers = [
            Paper(
                arxiv_id="1",
                title="Paper 1",
                abstract="",
                authors=["A"],
                categories=[],
                published="",
                updated="",
                link="",
                relevance_score=5.0,
                # No h-index or upvotes
            ),
        ]
        compute_quality_scores(papers)
        # No boost, score should equal relevance_score
        assert papers[0].quality_score == pytest.approx(5.0)
