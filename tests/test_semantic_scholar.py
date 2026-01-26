"""Tests for Semantic Scholar client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arxiv_digest.sources.semantic_scholar import SemanticScholarClient


class TestSemanticScholarClient:
    """Tests for the SemanticScholarClient class."""

    def test_init_default(self) -> None:
        """Test client initializes with defaults."""
        client = SemanticScholarClient()
        assert client.api_key is None
        assert client.timeout == 30.0

    def test_init_with_api_key(self) -> None:
        """Test client initializes with API key."""
        client = SemanticScholarClient(api_key="test-key", timeout=60.0)
        assert client.api_key == "test-key"
        assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_fetch_papers_success(self) -> None:
        """Test successful paper fetching."""
        mock_response_data = {
            "data": [
                {
                    "paperId": "abc123",
                    "externalIds": {"ArXiv": "2401.00001"},
                    "title": "Test Paper Title",
                    "abstract": "This is a test abstract.",
                    "authors": [
                        {"name": "Alice Smith"},
                        {"name": "Bob Johnson"},
                    ],
                    "year": 2024,
                    "fieldsOfStudy": ["Computer Science"],
                }
            ]
        }

        client = SemanticScholarClient(api_key="test-key")

        with patch(
            "arxiv_digest.sources.semantic_scholar.httpx.AsyncClient"
        ) as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            papers = await client.fetch_papers(query="machine learning", limit=10)

        assert len(papers) == 1
        assert papers[0].arxiv_id == "2401.00001"
        assert papers[0].title == "Test Paper Title"
        assert papers[0].abstract == "This is a test abstract."
        assert papers[0].authors == ["Alice Smith", "Bob Johnson"]
        assert papers[0].categories == ["Computer Science"]

    @pytest.mark.asyncio
    async def test_fetch_papers_no_arxiv_id(self) -> None:
        """Test handling papers without arXiv ID."""
        mock_response_data = {
            "data": [
                {
                    "paperId": "abc123",
                    "externalIds": {},
                    "title": "Test Paper",
                    "abstract": "Abstract",
                    "authors": [{"name": "Author"}],
                    "year": 2024,
                    "fieldsOfStudy": None,
                }
            ]
        }

        client = SemanticScholarClient()

        with patch(
            "arxiv_digest.sources.semantic_scholar.httpx.AsyncClient"
        ) as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            papers = await client.fetch_papers(query="test", limit=10)

        assert len(papers) == 1
        assert papers[0].arxiv_id == "s2:abc123"  # Fallback to paperId

    @pytest.mark.asyncio
    async def test_fetch_papers_empty_response(self) -> None:
        """Test handling empty response."""
        client = SemanticScholarClient()

        with patch(
            "arxiv_digest.sources.semantic_scholar.httpx.AsyncClient"
        ) as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": []}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            papers = await client.fetch_papers(query="test", limit=10)

        assert papers == []

    @pytest.mark.asyncio
    async def test_fetch_papers_http_error(self) -> None:
        """Test handling HTTP errors."""
        import httpx

        client = SemanticScholarClient()

        with patch(
            "arxiv_digest.sources.semantic_scholar.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.HTTPError("Connection failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            papers = await client.fetch_papers(query="test", limit=10)

        assert papers == []
