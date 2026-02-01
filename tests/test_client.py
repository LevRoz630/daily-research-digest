"""Tests for daily_research_digest.client module."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from daily_research_digest.client import ArxivClient
from daily_research_digest.models import DateFilter, Paper


class TestArxivClient:
    """Tests for the ArxivClient class."""

    def test_init_default_timeout(self) -> None:
        """Test client initializes with default timeout."""
        client = ArxivClient()
        assert client.timeout == 30.0

    def test_init_custom_timeout(self) -> None:
        """Test client initializes with custom timeout."""
        client = ArxivClient(timeout=60.0)
        assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_fetch_papers_success(self, mock_arxiv_response: str) -> None:
        """Test successful paper fetching and parsing."""
        client = ArxivClient()

        mock_response = MagicMock()
        mock_response.text = mock_arxiv_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_async_client

            papers = await client.fetch_papers(["cs.AI"], max_results=10)

        assert len(papers) == 3
        assert papers[0].arxiv_id == "2401.00001v1"
        assert papers[0].title == "Test Paper One: A Study in Machine Learning"
        assert "Alice Smith" in papers[0].authors
        assert "Bob Johnson" in papers[0].authors
        assert "cs.AI" in papers[0].categories

    @pytest.mark.asyncio
    async def test_fetch_papers_http_error(self) -> None:
        """Test handling of HTTP errors."""
        client = ArxivClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get.side_effect = httpx.HTTPError("Connection failed")
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_async_client

            papers = await client.fetch_papers(["cs.AI"])

        assert papers == []

    @pytest.mark.asyncio
    async def test_fetch_papers_xml_error(self) -> None:
        """Test handling of malformed XML."""
        client = ArxivClient()

        mock_response = MagicMock()
        mock_response.text = "This is not valid XML"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_async_client

            papers = await client.fetch_papers(["cs.AI"])

        assert papers == []

    @pytest.mark.asyncio
    async def test_fetch_papers_builds_correct_url(self) -> None:
        """Test that the correct URL is built for categories."""
        client = ArxivClient()

        mock_response = MagicMock()
        mock_response.text = (
            '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        )
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_async_client

            await client.fetch_papers(["cs.AI", "cs.LG"], max_results=25)

            # Verify URL construction
            call_args = mock_async_client.get.call_args
            url = call_args[0][0]
            assert "cat:cs.AI+OR+cat:cs.LG" in url
            assert "max_results=25" in url
            assert "sortBy=submittedDate" in url


class TestDateFiltering:
    """Tests for date filtering functionality."""

    def test_filter_by_days_back(self, sample_papers: list[Paper]) -> None:
        """Test filtering papers by days_back."""
        client = ArxivClient()
        date_filter = DateFilter(days_back=3)

        filtered = client._filter_by_date(sample_papers, date_filter)

        # Should only include papers from last 3 days (indices 0 and 1)
        assert len(filtered) == 2
        assert all("00001" in p.arxiv_id or "00002" in p.arxiv_id for p in filtered)

    def test_filter_by_days_back_week(self, sample_papers: list[Paper]) -> None:
        """Test filtering papers by 7 days."""
        client = ArxivClient()
        date_filter = DateFilter(days_back=7)

        filtered = client._filter_by_date(sample_papers, date_filter)

        # Should include papers from last 7 days (indices 0, 1, 2)
        assert len(filtered) == 3

    def test_filter_by_published_after(self) -> None:
        """Test filtering papers published after a specific date."""
        client = ArxivClient()
        papers = [
            Paper(
                arxiv_id="1",
                title="Paper 1",
                abstract="Abstract",
                authors=["Author"],
                categories=["cs.AI"],
                published="2024-01-15T00:00:00Z",
                updated="2024-01-15T00:00:00Z",
                link="https://arxiv.org/abs/1",
            ),
            Paper(
                arxiv_id="2",
                title="Paper 2",
                abstract="Abstract",
                authors=["Author"],
                categories=["cs.AI"],
                published="2024-01-10T00:00:00Z",
                updated="2024-01-10T00:00:00Z",
                link="https://arxiv.org/abs/2",
            ),
            Paper(
                arxiv_id="3",
                title="Paper 3",
                abstract="Abstract",
                authors=["Author"],
                categories=["cs.AI"],
                published="2024-01-05T00:00:00Z",
                updated="2024-01-05T00:00:00Z",
                link="https://arxiv.org/abs/3",
            ),
        ]

        date_filter = DateFilter(published_after="2024-01-08")
        filtered = client._filter_by_date(papers, date_filter)

        assert len(filtered) == 2
        assert all(p.arxiv_id in ["1", "2"] for p in filtered)

    def test_filter_by_published_before(self) -> None:
        """Test filtering papers published before a specific date."""
        client = ArxivClient()
        papers = [
            Paper(
                arxiv_id="1",
                title="Paper 1",
                abstract="Abstract",
                authors=["Author"],
                categories=["cs.AI"],
                published="2024-01-15T00:00:00Z",
                updated="2024-01-15T00:00:00Z",
                link="https://arxiv.org/abs/1",
            ),
            Paper(
                arxiv_id="2",
                title="Paper 2",
                abstract="Abstract",
                authors=["Author"],
                categories=["cs.AI"],
                published="2024-01-10T00:00:00Z",
                updated="2024-01-10T00:00:00Z",
                link="https://arxiv.org/abs/2",
            ),
        ]

        date_filter = DateFilter(published_before="2024-01-12")
        filtered = client._filter_by_date(papers, date_filter)

        assert len(filtered) == 1
        assert filtered[0].arxiv_id == "2"

    def test_filter_combined(self) -> None:
        """Test combining multiple date filters."""
        client = ArxivClient()
        papers = [
            Paper(
                arxiv_id="1",
                title="Paper 1",
                abstract="Abstract",
                authors=["Author"],
                categories=["cs.AI"],
                published="2024-01-20T00:00:00Z",
                updated="2024-01-20T00:00:00Z",
                link="https://arxiv.org/abs/1",
            ),
            Paper(
                arxiv_id="2",
                title="Paper 2",
                abstract="Abstract",
                authors=["Author"],
                categories=["cs.AI"],
                published="2024-01-15T00:00:00Z",
                updated="2024-01-15T00:00:00Z",
                link="https://arxiv.org/abs/2",
            ),
            Paper(
                arxiv_id="3",
                title="Paper 3",
                abstract="Abstract",
                authors=["Author"],
                categories=["cs.AI"],
                published="2024-01-05T00:00:00Z",
                updated="2024-01-05T00:00:00Z",
                link="https://arxiv.org/abs/3",
            ),
        ]

        date_filter = DateFilter(
            published_after="2024-01-10",
            published_before="2024-01-18",
        )
        filtered = client._filter_by_date(papers, date_filter)

        assert len(filtered) == 1
        assert filtered[0].arxiv_id == "2"

    def test_filter_no_filter(self, sample_papers: list[Paper]) -> None:
        """Test that no filtering returns all papers."""
        client = ArxivClient()
        date_filter = DateFilter()  # All None

        filtered = client._filter_by_date(sample_papers, date_filter)
        assert len(filtered) == len(sample_papers)

    def test_filter_invalid_date_includes_paper(self) -> None:
        """Test that papers with unparseable dates are included."""
        client = ArxivClient()
        papers = [
            Paper(
                arxiv_id="1",
                title="Paper 1",
                abstract="Abstract",
                authors=["Author"],
                categories=["cs.AI"],
                published="invalid-date",
                updated="invalid-date",
                link="https://arxiv.org/abs/1",
            ),
        ]

        date_filter = DateFilter(days_back=7)
        filtered = client._filter_by_date(papers, date_filter)

        # Invalid date should still be included
        assert len(filtered) == 1
