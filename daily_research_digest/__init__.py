"""Daily Research Digest - AI-powered paper digest with LLM-based ranking."""

from .client import ArxivClient
from .config_env import ConfigError, DigestEmailConfig, SMTPConfig, load_config_from_env
from .digest import DigestGenerator
from .digest_renderer import Digest, render_digest
from .digest_send import main as send_digest
from .digest_state import LocalFileStateBackend, compute_digest_id
from .email_provider import DryRunProvider, EmailSendError, SMTPProvider
from .memory import PaperMemory
from .models import DateFilter, DigestConfig, DigestState, Paper
from .ranker import PaperRanker, get_llm_for_provider
from .scheduler import ArxivScheduler
from .sources.huggingface import HuggingFaceClient
from .sources.semantic_scholar import SemanticScholarClient
from .storage import DigestStorage

__version__ = "0.1.0"

__all__ = [
    # Core
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
    "SemanticScholarClient",
    "get_llm_for_provider",
    # Email digest
    "ConfigError",
    "Digest",
    "DigestEmailConfig",
    "DryRunProvider",
    "EmailSendError",
    "LocalFileStateBackend",
    "SMTPConfig",
    "SMTPProvider",
    "compute_digest_id",
    "load_config_from_env",
    "render_digest",
    "send_digest",
]
