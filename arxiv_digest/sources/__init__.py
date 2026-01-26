"""Paper sources for arxiv-digest."""

from .huggingface import HuggingFaceClient
from .semantic_scholar import SemanticScholarClient

__all__ = ["HuggingFaceClient", "SemanticScholarClient"]
