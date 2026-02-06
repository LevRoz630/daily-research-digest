"""Daily Research Digest - AI-powered paper digest with LLM-based ranking."""

from .digest import DigestGenerator
from .digest_renderer import Digest, render_digest
from .memory import PaperMemory
from .models import DateFilter, DigestConfig, DigestState, Paper
from .quality import compute_quality_score, compute_quality_scores
from .ranker import PaperRanker, get_llm_for_provider
from .sources.semantic_scholar import SemanticScholarClient
from .storage import DigestStorage

__version__ = "0.3.0"

__all__ = [
    "DateFilter",
    "Digest",
    "DigestConfig",
    "DigestGenerator",
    "DigestState",
    "DigestStorage",
    "Paper",
    "PaperMemory",
    "PaperRanker",
    "SemanticScholarClient",
    "compute_quality_score",
    "compute_quality_scores",
    "get_llm_for_provider",
    "render_digest",
]
