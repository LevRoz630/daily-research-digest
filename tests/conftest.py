"""Shared test fixtures for daily-research-digest tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from daily_research_digest.models import DateFilter, DigestConfig, Paper


@pytest.fixture
def sample_paper() -> Paper:
    """Return a single test Paper instance."""
    return Paper(
        arxiv_id="2401.00001v1",
        title="Test Paper: A Study in Machine Learning",
        abstract="This is a test abstract about machine learning and AI research.",
        authors=["Alice Smith", "Bob Johnson"],
        categories=["cs.AI", "cs.LG"],
        published="2024-01-15T00:00:00Z",
        updated="2024-01-15T00:00:00Z",
        link="https://arxiv.org/abs/2401.00001v1",
    )


@pytest.fixture
def sample_papers() -> list[Paper]:
    """Return a list of test Paper instances."""
    now = datetime.now(timezone.utc)
    return [
        Paper(
            arxiv_id="2401.00001v1",
            title="Paper One: Machine Learning Basics",
            abstract="Abstract about machine learning fundamentals.",
            authors=["Alice Smith"],
            categories=["cs.AI", "cs.LG"],
            published=(now - timedelta(days=1)).isoformat(),
            updated=(now - timedelta(days=1)).isoformat(),
            link="https://arxiv.org/abs/2401.00001v1",
        ),
        Paper(
            arxiv_id="2401.00002v1",
            title="Paper Two: Neural Networks",
            abstract="Abstract about neural network architectures.",
            authors=["Bob Johnson", "Carol Williams"],
            categories=["cs.CL", "cs.AI"],
            published=(now - timedelta(days=2)).isoformat(),
            updated=(now - timedelta(days=2)).isoformat(),
            link="https://arxiv.org/abs/2401.00002v1",
        ),
        Paper(
            arxiv_id="2401.00003v1",
            title="Paper Three: Reinforcement Learning",
            abstract="Abstract about reinforcement learning algorithms.",
            authors=["David Brown"],
            categories=["cs.LG"],
            published=(now - timedelta(days=5)).isoformat(),
            updated=(now - timedelta(days=5)).isoformat(),
            link="https://arxiv.org/abs/2401.00003v1",
        ),
        Paper(
            arxiv_id="2401.00004v1",
            title="Paper Four: Computer Vision",
            abstract="Abstract about computer vision techniques.",
            authors=["Eve Davis", "Frank Miller"],
            categories=["cs.CV"],
            published=(now - timedelta(days=10)).isoformat(),
            updated=(now - timedelta(days=10)).isoformat(),
            link="https://arxiv.org/abs/2401.00004v1",
        ),
        Paper(
            arxiv_id="2401.00005v1",
            title="Paper Five: NLP Transformers",
            abstract="Abstract about transformer models for NLP.",
            authors=["Grace Lee"],
            categories=["cs.CL"],
            published=(now - timedelta(days=15)).isoformat(),
            updated=(now - timedelta(days=15)).isoformat(),
            link="https://arxiv.org/abs/2401.00005v1",
        ),
    ]


@pytest.fixture
def sample_config() -> DigestConfig:
    """Return a test DigestConfig instance."""
    return DigestConfig(
        categories=["cs.AI", "cs.LG"],
        interests="machine learning, neural networks, deep learning",
        max_papers=50,
        top_n=10,
        llm_provider="anthropic",
        anthropic_api_key="test-api-key",
    )


@pytest.fixture
def sample_config_with_date_filter() -> DigestConfig:
    """Return a test DigestConfig with date filter."""
    return DigestConfig(
        categories=["cs.AI"],
        interests="machine learning",
        date_filter=DateFilter(days_back=7),
    )


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for storage tests."""
    storage_dir = tmp_path / "digests"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def mock_arxiv_response() -> str:
    """Return sample arXiv API XML response."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    return (fixtures_dir / "arxiv_response.xml").read_text()


@pytest.fixture
def mock_llm() -> MagicMock:
    """Return a mock LLM that returns predictable ranking responses."""
    mock = MagicMock()

    async def mock_ainvoke(prompt: str) -> MagicMock:
        response = MagicMock()
        if "machine learning" in prompt.lower():
            response.content = '{"score": 8, "reason": "Highly relevant to ML interests"}'
        elif "neural" in prompt.lower():
            response.content = '{"score": 7, "reason": "Relevant neural network research"}'
        elif "reinforcement" in prompt.lower():
            response.content = '{"score": 6, "reason": "Somewhat relevant RL paper"}'
        else:
            response.content = '{"score": 5, "reason": "Moderate relevance"}'
        return response

    mock.ainvoke = mock_ainvoke
    return mock


@pytest.fixture
def mock_llm_markdown_response() -> MagicMock:
    """Return a mock LLM that returns markdown-wrapped JSON."""
    mock = MagicMock()

    async def mock_ainvoke(prompt: str) -> MagicMock:
        response = MagicMock()
        response.content = '```json\n{"score": 7, "reason": "Test reason"}\n```'
        return response

    mock.ainvoke = mock_ainvoke
    return mock


@pytest.fixture
def mock_llm_invalid_response() -> MagicMock:
    """Return a mock LLM that returns invalid JSON."""
    mock = MagicMock()

    async def mock_ainvoke(prompt: str) -> MagicMock:
        response = MagicMock()
        response.content = "This is not valid JSON at all"
        return response

    mock.ainvoke = mock_ainvoke
    return mock


@pytest.fixture
def sample_digest() -> dict:
    """Return a sample digest dictionary."""
    return {
        "date": "2024-01-15",
        "generated_at": "2024-01-15T10:00:00+00:00",
        "categories": ["cs.AI", "cs.LG"],
        "interests": "machine learning, neural networks",
        "total_papers_fetched": 50,
        "papers": [
            {
                "arxiv_id": "2401.00001v1",
                "title": "Test Paper One",
                "abstract": "Test abstract one.",
                "authors": ["Alice Smith"],
                "categories": ["cs.AI"],
                "published": "2024-01-15T00:00:00Z",
                "updated": "2024-01-15T00:00:00Z",
                "link": "https://arxiv.org/abs/2401.00001v1",
                "relevance_score": 9.0,
                "relevance_reason": "Highly relevant",
            },
            {
                "arxiv_id": "2401.00002v1",
                "title": "Test Paper Two",
                "abstract": "Test abstract two.",
                "authors": ["Bob Johnson"],
                "categories": ["cs.LG"],
                "published": "2024-01-14T00:00:00Z",
                "updated": "2024-01-14T00:00:00Z",
                "link": "https://arxiv.org/abs/2401.00002v1",
                "relevance_score": 8.0,
                "relevance_reason": "Very relevant",
            },
        ],
    }
