"""Tests for arxiv_digest.digest module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arxiv_digest.client import ArxivClient
from arxiv_digest.digest import DigestGenerator
from arxiv_digest.models import DigestConfig, Paper
from arxiv_digest.storage import DigestStorage


class TestDigestGenerator:
    """Tests for the DigestGenerator class."""

    def test_init_with_storage(self, temp_storage_dir: Path) -> None:
        """Test generator initializes with storage."""
        storage = DigestStorage(temp_storage_dir)
        generator = DigestGenerator(storage)

        assert generator.storage == storage
        assert generator.client is not None
        assert generator.state.is_generating is False

    def test_init_with_custom_client(self, temp_storage_dir: Path) -> None:
        """Test generator initializes with custom client."""
        storage = DigestStorage(temp_storage_dir)
        client = ArxivClient(timeout=60.0)
        generator = DigestGenerator(storage, client=client)

        assert generator.client == client
        assert generator.client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_generate_success(
        self,
        temp_storage_dir: Path,
        sample_config: DigestConfig,
        sample_papers: list[Paper],
    ) -> None:
        """Test successful digest generation."""
        storage = DigestStorage(temp_storage_dir)
        generator = DigestGenerator(storage)

        # Mock the client and ranker
        with patch.object(generator.client, "fetch_papers", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_papers

            with patch("arxiv_digest.digest.get_llm_for_provider") as mock_get_llm:
                mock_llm = MagicMock()

                async def mock_ainvoke(prompt: str) -> MagicMock:
                    response = MagicMock()
                    response.content = '{"score": 7, "reason": "Relevant"}'
                    return response

                mock_llm.ainvoke = mock_ainvoke
                mock_get_llm.return_value = mock_llm

                result = await generator.generate(sample_config)

        assert result["status"] == "completed"
        assert "digest" in result
        assert result["digest"]["total_papers_fetched"] == len(sample_papers)
        assert len(result["digest"]["papers"]) <= sample_config.top_n

    @pytest.mark.asyncio
    async def test_generate_no_papers(
        self, temp_storage_dir: Path, sample_config: DigestConfig
    ) -> None:
        """Test generation with no papers returns error."""
        storage = DigestStorage(temp_storage_dir)
        generator = DigestGenerator(storage)

        with patch.object(generator.client, "fetch_papers", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []

            result = await generator.generate(sample_config)

        assert result["status"] == "error"
        assert "No papers fetched" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_generate_already_running(
        self,
        temp_storage_dir: Path,
        sample_config: DigestConfig,
        sample_papers: list[Paper],
    ) -> None:
        """Test concurrent generation is prevented."""
        storage = DigestStorage(temp_storage_dir)
        generator = DigestGenerator(storage)

        # Manually set generating state
        generator.state.is_generating = True

        result = await generator.generate(sample_config)

        assert result["status"] == "already_generating"

    @pytest.mark.asyncio
    async def test_generate_resets_state_on_error(
        self, temp_storage_dir: Path, sample_config: DigestConfig
    ) -> None:
        """Test that is_generating is reset even on error."""
        storage = DigestStorage(temp_storage_dir)
        generator = DigestGenerator(storage)

        with patch.object(generator.client, "fetch_papers", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("Test error")

            result = await generator.generate(sample_config)

        assert result["status"] == "error"
        assert generator.state.is_generating is False

    def test_get_state(self, temp_storage_dir: Path) -> None:
        """Test getting current generator state."""
        storage = DigestStorage(temp_storage_dir)
        generator = DigestGenerator(storage)

        state = generator.get_state()

        assert state.last_digest is None
        assert state.is_generating is False
        assert state.errors == []

    @pytest.mark.asyncio
    async def test_generate_passes_date_filter(
        self,
        temp_storage_dir: Path,
        sample_config_with_date_filter: DigestConfig,
        sample_papers: list[Paper],
    ) -> None:
        """Test that date_filter is passed to client."""
        storage = DigestStorage(temp_storage_dir)
        generator = DigestGenerator(storage)

        with patch.object(generator.client, "fetch_papers", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_papers

            with patch("arxiv_digest.digest.get_llm_for_provider") as mock_get_llm:
                mock_llm = MagicMock()

                async def mock_ainvoke(prompt: str) -> MagicMock:
                    response = MagicMock()
                    response.content = '{"score": 7, "reason": "Relevant"}'
                    return response

                mock_llm.ainvoke = mock_ainvoke
                mock_get_llm.return_value = mock_llm

                await generator.generate(sample_config_with_date_filter)

            # Verify fetch_papers was called with date_filter
            mock_fetch.assert_called_once()
            call_args = mock_fetch.call_args
            assert call_args[0][2] == sample_config_with_date_filter.date_filter
