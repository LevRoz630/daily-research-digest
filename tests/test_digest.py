"""Tests for daily_research_digest.digest module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daily_research_digest.digest import DigestGenerator
from daily_research_digest.memory import PaperMemory
from daily_research_digest.models import DigestConfig, Paper
from daily_research_digest.storage import DigestStorage


class TestDigestGenerator:
    """Tests for the DigestGenerator class."""

    def test_init_with_storage(self, temp_storage_dir: Path) -> None:
        """Test generator initializes with storage."""
        storage = DigestStorage(temp_storage_dir)
        generator = DigestGenerator(storage)

        assert generator.storage == storage
        assert generator.state.is_generating is False

    def test_init_with_memory(self, temp_storage_dir: Path, tmp_path: Path) -> None:
        """Test generator initializes with paper memory."""
        storage = DigestStorage(temp_storage_dir)
        memory = PaperMemory(tmp_path / "memory.json")
        generator = DigestGenerator(storage, memory=memory)

        assert generator.memory == memory

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

        # Mock the Semantic Scholar and HuggingFace clients
        with patch(
            "daily_research_digest.digest.SemanticScholarClient"
        ) as mock_ss_cls, patch(
            "daily_research_digest.digest.HuggingFaceClient"
        ) as mock_hf_cls, patch(
            "daily_research_digest.digest.get_llm_for_provider"
        ) as mock_get_llm:
            mock_ss = MagicMock()
            mock_ss.fetch_papers = AsyncMock(return_value=sample_papers)
            mock_ss_cls.return_value = mock_ss

            mock_hf = MagicMock()
            mock_hf.fetch_papers = AsyncMock(return_value=[])
            mock_hf_cls.return_value = mock_hf

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

        with patch(
            "daily_research_digest.digest.SemanticScholarClient"
        ) as mock_ss_cls, patch(
            "daily_research_digest.digest.HuggingFaceClient"
        ) as mock_hf_cls:
            mock_ss = MagicMock()
            mock_ss.fetch_papers = AsyncMock(return_value=[])
            mock_ss_cls.return_value = mock_ss

            mock_hf = MagicMock()
            mock_hf.fetch_papers = AsyncMock(return_value=[])
            mock_hf_cls.return_value = mock_hf

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

        with patch(
            "daily_research_digest.digest.SemanticScholarClient"
        ) as mock_ss_cls:
            mock_ss = MagicMock()
            mock_ss.fetch_papers = AsyncMock(side_effect=Exception("Test error"))
            mock_ss_cls.return_value = mock_ss

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
    async def test_generate_filters_seen_papers(
        self,
        temp_storage_dir: Path,
        tmp_path: Path,
        sample_config: DigestConfig,
        sample_papers: list[Paper],
    ) -> None:
        """Test that seen papers are filtered out when memory is enabled."""
        storage = DigestStorage(temp_storage_dir)
        memory = PaperMemory(tmp_path / "memory.json")

        # Pre-record some papers as seen
        memory.record(sample_papers[0].arxiv_id)
        memory.record(sample_papers[1].arxiv_id)

        generator = DigestGenerator(storage, memory=memory)

        with patch(
            "daily_research_digest.digest.SemanticScholarClient"
        ) as mock_ss_cls, patch(
            "daily_research_digest.digest.HuggingFaceClient"
        ) as mock_hf_cls, patch(
            "daily_research_digest.digest.get_llm_for_provider"
        ) as mock_get_llm:
            mock_ss = MagicMock()
            mock_ss.fetch_papers = AsyncMock(return_value=sample_papers)
            mock_ss_cls.return_value = mock_ss

            mock_hf = MagicMock()
            mock_hf.fetch_papers = AsyncMock(return_value=[])
            mock_hf_cls.return_value = mock_hf

            mock_llm = MagicMock()

            async def mock_ainvoke(prompt: str) -> MagicMock:
                response = MagicMock()
                response.content = '{"score": 7, "reason": "OK"}'
                return response

            mock_llm.ainvoke = mock_ainvoke
            mock_get_llm.return_value = mock_llm

            result = await generator.generate(sample_config)

        assert result["status"] == "completed"
        # Should have filtered out 2 seen papers, so only 3 remain
        assert result["digest"]["total_papers_fetched"] == 3

    @pytest.mark.asyncio
    async def test_generate_records_papers_in_memory(
        self,
        temp_storage_dir: Path,
        tmp_path: Path,
        sample_config: DigestConfig,
        sample_papers: list[Paper],
    ) -> None:
        """Test that generated papers are recorded in memory."""
        storage = DigestStorage(temp_storage_dir)
        memory = PaperMemory(tmp_path / "memory.json")
        generator = DigestGenerator(storage, memory=memory)

        with patch(
            "daily_research_digest.digest.SemanticScholarClient"
        ) as mock_ss_cls, patch(
            "daily_research_digest.digest.HuggingFaceClient"
        ) as mock_hf_cls, patch(
            "daily_research_digest.digest.get_llm_for_provider"
        ) as mock_get_llm:
            mock_ss = MagicMock()
            mock_ss.fetch_papers = AsyncMock(return_value=sample_papers)
            mock_ss_cls.return_value = mock_ss

            mock_hf = MagicMock()
            mock_hf.fetch_papers = AsyncMock(return_value=[])
            mock_hf_cls.return_value = mock_hf

            mock_llm = MagicMock()

            async def mock_ainvoke(prompt: str) -> MagicMock:
                response = MagicMock()
                response.content = '{"score": 7, "reason": "OK"}'
                return response

            mock_llm.ainvoke = mock_ainvoke
            mock_get_llm.return_value = mock_llm

            await generator.generate(sample_config)

        # Check that papers were recorded
        assert memory.count() > 0

    @pytest.mark.asyncio
    async def test_generate_without_memory(
        self,
        temp_storage_dir: Path,
        sample_config: DigestConfig,
        sample_papers: list[Paper],
    ) -> None:
        """Test generation works without memory (backward compatibility)."""
        storage = DigestStorage(temp_storage_dir)
        generator = DigestGenerator(storage)  # No memory

        assert generator.memory is None

        with patch(
            "daily_research_digest.digest.SemanticScholarClient"
        ) as mock_ss_cls, patch(
            "daily_research_digest.digest.HuggingFaceClient"
        ) as mock_hf_cls, patch(
            "daily_research_digest.digest.get_llm_for_provider"
        ) as mock_get_llm:
            mock_ss = MagicMock()
            mock_ss.fetch_papers = AsyncMock(return_value=sample_papers)
            mock_ss_cls.return_value = mock_ss

            mock_hf = MagicMock()
            mock_hf.fetch_papers = AsyncMock(return_value=[])
            mock_hf_cls.return_value = mock_hf

            mock_llm = MagicMock()

            async def mock_ainvoke(prompt: str) -> MagicMock:
                response = MagicMock()
                response.content = '{"score": 7, "reason": "OK"}'
                return response

            mock_llm.ainvoke = mock_ainvoke
            mock_get_llm.return_value = mock_llm

            result = await generator.generate(sample_config)

        assert result["status"] == "completed"
        assert result["digest"]["total_papers_fetched"] == len(sample_papers)

    @pytest.mark.asyncio
    async def test_generate_exclude_seen_false(
        self,
        temp_storage_dir: Path,
        tmp_path: Path,
        sample_papers: list[Paper],
    ) -> None:
        """Test that exclude_seen=False disables filtering."""
        storage = DigestStorage(temp_storage_dir)
        memory = PaperMemory(tmp_path / "memory.json")

        # Pre-record some papers
        memory.record(sample_papers[0].arxiv_id)
        memory.record(sample_papers[1].arxiv_id)

        generator = DigestGenerator(storage, memory=memory)

        config = DigestConfig(
            categories=["cs.AI"],
            interests="machine learning",
            exclude_seen=False,  # Disable filtering
            anthropic_api_key="test-key",
        )

        with patch(
            "daily_research_digest.digest.SemanticScholarClient"
        ) as mock_ss_cls, patch(
            "daily_research_digest.digest.HuggingFaceClient"
        ) as mock_hf_cls, patch(
            "daily_research_digest.digest.get_llm_for_provider"
        ) as mock_get_llm:
            mock_ss = MagicMock()
            mock_ss.fetch_papers = AsyncMock(return_value=sample_papers)
            mock_ss_cls.return_value = mock_ss

            mock_hf = MagicMock()
            mock_hf.fetch_papers = AsyncMock(return_value=[])
            mock_hf_cls.return_value = mock_hf

            mock_llm = MagicMock()

            async def mock_ainvoke(prompt: str) -> MagicMock:
                response = MagicMock()
                response.content = '{"score": 7, "reason": "OK"}'
                return response

            mock_llm.ainvoke = mock_ainvoke
            mock_get_llm.return_value = mock_llm

            result = await generator.generate(config)

        assert result["status"] == "completed"
        # Should NOT have filtered, all 5 papers should remain
        assert result["digest"]["total_papers_fetched"] == 5

    @pytest.mark.asyncio
    async def test_generate_all_papers_seen(
        self,
        temp_storage_dir: Path,
        tmp_path: Path,
        sample_config: DigestConfig,
        sample_papers: list[Paper],
    ) -> None:
        """Test generation when all papers are already seen."""
        storage = DigestStorage(temp_storage_dir)
        memory = PaperMemory(tmp_path / "memory.json")

        # Pre-record all papers
        for paper in sample_papers:
            memory.record(paper.arxiv_id)

        generator = DigestGenerator(storage, memory=memory)

        with patch(
            "daily_research_digest.digest.SemanticScholarClient"
        ) as mock_ss_cls, patch(
            "daily_research_digest.digest.HuggingFaceClient"
        ) as mock_hf_cls:
            mock_ss = MagicMock()
            mock_ss.fetch_papers = AsyncMock(return_value=sample_papers)
            mock_ss_cls.return_value = mock_ss

            mock_hf = MagicMock()
            mock_hf.fetch_papers = AsyncMock(return_value=[])
            mock_hf_cls.return_value = mock_hf

            result = await generator.generate(sample_config)

        assert result["status"] == "error"
        assert "No unseen papers" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_generate_with_priority_authors(
        self,
        temp_storage_dir: Path,
        sample_papers: list[Paper],
    ) -> None:
        """Test that priority authors boost paper scores."""
        storage = DigestStorage(temp_storage_dir)
        generator = DigestGenerator(storage)

        # Alice Smith is an author on sample_papers[0]
        config = DigestConfig(
            categories=["cs.AI"],
            interests="machine learning",
            priority_authors=["Alice Smith"],
            author_boost=2.0,
            anthropic_api_key="test-key",
        )

        with patch(
            "daily_research_digest.digest.SemanticScholarClient"
        ) as mock_ss_cls, patch(
            "daily_research_digest.digest.HuggingFaceClient"
        ) as mock_hf_cls, patch(
            "daily_research_digest.digest.get_llm_for_provider"
        ) as mock_get_llm:
            mock_ss = MagicMock()
            mock_ss.fetch_papers = AsyncMock(return_value=sample_papers)
            mock_ss_cls.return_value = mock_ss

            mock_hf = MagicMock()
            mock_hf.fetch_papers = AsyncMock(return_value=[])
            mock_hf_cls.return_value = mock_hf

            mock_llm = MagicMock()

            async def mock_ainvoke(prompt: str) -> MagicMock:
                response = MagicMock()
                # Give all papers same base score
                response.content = '{"score": 5, "reason": "OK"}'
                return response

            mock_llm.ainvoke = mock_ainvoke
            mock_get_llm.return_value = mock_llm

            result = await generator.generate(config)

        assert result["status"] == "completed"
        papers = result["digest"]["papers"]
        # Paper by Alice Smith should be first (boosted from 5 to 10)
        assert papers[0]["authors"] == ["Alice Smith"]
        assert papers[0]["relevance_score"] == 10.0
