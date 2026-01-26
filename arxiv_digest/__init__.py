"""ArXiv Digest - AI-powered paper digest with LLM-based ranking."""

from .client import ArxivClient
from .digest import DigestGenerator
from .memory import PaperMemory
from .models import DateFilter, DigestConfig, DigestState, Paper
from .ranker import PaperRanker, get_llm_for_provider
from .scheduler import ArxivScheduler
from .sources.huggingface import HuggingFaceClient
from .storage import DigestStorage

__version__ = "0.1.0"

__all__ = [
    "ArxivClient",
    "ArxivScheduler",
    "DateFilter",
    "DigestConfig",
    "DigestGenerator",
    "DigestState",
    "DigestStorage",
    "HuggingFaceClient",
    "Paper",
    "PaperMemory",
    "PaperRanker",
    "get_llm_for_provider",
]
