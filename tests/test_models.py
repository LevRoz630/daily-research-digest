"""Tests for daily_research_digest.models module."""

from daily_research_digest.models import DateFilter, DigestConfig, DigestState, Paper


class TestPaper:
    """Tests for the Paper dataclass."""

    def test_paper_creation(self, sample_paper: Paper) -> None:
        """Test Paper instantiation with all required fields."""
        assert sample_paper.arxiv_id == "2401.00001v1"
        assert sample_paper.title == "Test Paper: A Study in Machine Learning"
        assert "machine learning" in sample_paper.abstract.lower()
        assert len(sample_paper.authors) == 2
        assert "cs.AI" in sample_paper.categories
        assert sample_paper.link.startswith("https://arxiv.org/abs/")

    def test_paper_default_scores(self) -> None:
        """Test Paper has default relevance score and reason."""
        paper = Paper(
            arxiv_id="test",
            title="Test",
            abstract="Abstract",
            authors=["Author"],
            categories=["cs.AI"],
            published="2024-01-01T00:00:00Z",
            updated="2024-01-01T00:00:00Z",
            link="https://arxiv.org/abs/test",
        )
        assert paper.relevance_score == 0.0
        assert paper.relevance_reason == ""

    def test_paper_to_dict(self, sample_paper: Paper) -> None:
        """Test Paper serialization to dictionary."""
        data = sample_paper.to_dict()

        assert isinstance(data, dict)
        assert data["arxiv_id"] == sample_paper.arxiv_id
        assert data["title"] == sample_paper.title
        assert data["abstract"] == sample_paper.abstract
        assert data["authors"] == sample_paper.authors
        assert data["categories"] == sample_paper.categories
        assert data["published"] == sample_paper.published
        assert data["updated"] == sample_paper.updated
        assert data["link"] == sample_paper.link
        assert data["relevance_score"] == sample_paper.relevance_score
        assert data["relevance_reason"] == sample_paper.relevance_reason


class TestDateFilter:
    """Tests for the DateFilter dataclass."""

    def test_date_filter_defaults(self) -> None:
        """Test DateFilter has all None defaults."""
        df = DateFilter()
        assert df.days_back is None
        assert df.published_after is None
        assert df.published_before is None

    def test_date_filter_with_days_back(self) -> None:
        """Test DateFilter with days_back set."""
        df = DateFilter(days_back=7)
        assert df.days_back == 7
        assert df.published_after is None
        assert df.published_before is None

    def test_date_filter_with_date_range(self) -> None:
        """Test DateFilter with published_after and published_before."""
        df = DateFilter(
            published_after="2024-01-01",
            published_before="2024-01-31",
        )
        assert df.days_back is None
        assert df.published_after == "2024-01-01"
        assert df.published_before == "2024-01-31"


class TestDigestState:
    """Tests for the DigestState dataclass."""

    def test_digest_state_defaults(self) -> None:
        """Test DigestState has correct default values."""
        state = DigestState()
        assert state.last_digest is None
        assert state.is_generating is False
        assert state.errors == []

    def test_digest_state_errors_list(self) -> None:
        """Test DigestState errors is a mutable list."""
        state = DigestState()
        state.errors.append("error1")
        assert len(state.errors) == 1

        # Each instance should have its own list
        state2 = DigestState()
        assert len(state2.errors) == 0


class TestDigestConfig:
    """Tests for the DigestConfig dataclass."""

    def test_digest_config_required_fields(self) -> None:
        """Test DigestConfig requires categories and interests."""
        config = DigestConfig(
            categories=["cs.AI"],
            interests="machine learning",
        )
        assert config.categories == ["cs.AI"]
        assert config.interests == "machine learning"

    def test_digest_config_defaults(self, sample_config: DigestConfig) -> None:
        """Test DigestConfig has correct default values."""
        config = DigestConfig(
            categories=["cs.AI"],
            interests="test interests",
        )
        assert config.max_papers == 50
        assert config.top_n == 10
        assert config.batch_size == 25
        assert config.batch_delay == 0.2
        assert config.date_filter is None
        assert config.llm_provider == "anthropic"
        assert config.anthropic_api_key is None
        assert config.openai_api_key is None
        assert config.google_api_key is None

    def test_digest_config_with_date_filter(self) -> None:
        """Test DigestConfig with date_filter."""
        config = DigestConfig(
            categories=["cs.AI"],
            interests="test",
            date_filter=DateFilter(days_back=3),
        )
        assert config.date_filter is not None
        assert config.date_filter.days_back == 3

    def test_digest_config_exclude_seen_default(self) -> None:
        """Test DigestConfig.exclude_seen defaults to True."""
        config = DigestConfig(
            categories=["cs.AI"],
            interests="test",
        )
        assert config.exclude_seen is True

    def test_digest_config_exclude_seen_false(self) -> None:
        """Test DigestConfig.exclude_seen can be set to False."""
        config = DigestConfig(
            categories=["cs.AI"],
            interests="test",
            exclude_seen=False,
        )
        assert config.exclude_seen is False

    def test_digest_config_priority_authors_default(self) -> None:
        """Test DigestConfig.priority_authors defaults to None."""
        config = DigestConfig(
            categories=["cs.AI"],
            interests="test",
        )
        assert config.priority_authors is None
        assert config.author_boost == 1.5

    def test_digest_config_priority_authors(self) -> None:
        """Test DigestConfig with priority authors."""
        config = DigestConfig(
            categories=["cs.AI"],
            interests="test",
            priority_authors=["Alice Smith", "Bob Johnson"],
            author_boost=2.0,
        )
        assert config.priority_authors == ["Alice Smith", "Bob Johnson"]
        assert config.author_boost == 2.0

    def test_digest_config_sources_default(self) -> None:
        """Test DigestConfig.sources defaults to None (arxiv only)."""
        config = DigestConfig(
            categories=["cs.AI"],
            interests="test",
        )
        assert config.sources is None

    def test_digest_config_sources_multiple(self) -> None:
        """Test DigestConfig with multiple sources."""
        config = DigestConfig(
            categories=["cs.AI"],
            interests="test",
            sources=["arxiv", "huggingface"],
        )
        assert config.sources == ["arxiv", "huggingface"]
