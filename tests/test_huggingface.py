"""Tests for HuggingFace client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daily_research_digest.sources.huggingface import HuggingFaceClient


class TestHuggingFaceClient:
    """Tests for the HuggingFaceClient class."""

    def test_init_default_timeout(self) -> None:
        """Test client initializes with default timeout."""
        client = HuggingFaceClient()
        assert client.timeout == 30.0

    def test_init_custom_timeout(self) -> None:
        """Test client initializes with custom timeout."""
        client = HuggingFaceClient(timeout=60.0)
        assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_fetch_papers_success(self) -> None:
        """Test successful paper fetching."""
        mock_response_data = [
            {
                "paper": {
                    "id": "2401.00001",
                    "authors": [
                        {"name": "Alice Smith"},
                        {"name": "Bob Johnson"},
                    ],
                },
                "title": "Test Paper Title",
                "summary": "This is a test abstract.",
                "publishedAt": "2024-01-15T00:00:00Z",
            }
        ]

        client = HuggingFaceClient()

        with patch(
            "daily_research_digest.sources.huggingface.httpx.AsyncClient"
        ) as mock_client_class:
            # httpx response.json() is synchronous, so use MagicMock not AsyncMock
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            papers = await client.fetch_papers(limit=10)

        assert len(papers) == 1
        assert papers[0].arxiv_id == "2401.00001"
        assert papers[0].title == "Test Paper Title"
        assert papers[0].abstract == "This is a test abstract."
        assert papers[0].authors == ["Alice Smith", "Bob Johnson"]
        assert papers[0].link == "https://huggingface.co/papers/2401.00001"

    @pytest.mark.asyncio
    async def test_fetch_papers_empty_response(self) -> None:
        """Test handling empty response."""
        client = HuggingFaceClient()

        with patch(
            "daily_research_digest.sources.huggingface.httpx.AsyncClient"
        ) as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = []
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            papers = await client.fetch_papers(limit=10)

        assert papers == []

    @pytest.mark.asyncio
    async def test_fetch_papers_http_error(self) -> None:
        """Test handling HTTP errors."""
        import httpx

        client = HuggingFaceClient()

        with patch(
            "daily_research_digest.sources.huggingface.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.HTTPError("Connection failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            papers = await client.fetch_papers(limit=10)

        assert papers == []
