"""Quality scoring for papers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Paper


def compute_quality_score(
    paper: Paper,
    max_h_index: float = 100.0,
) -> float:
    """Compute final quality score for a paper.

    Formula: final = llm_relevance × (1 + quality_boost)
    where quality_boost = 0.35 × h_factor

    Args:
        paper: Paper with relevance_score, author_h_indices
        max_h_index: Maximum h-index for normalization (default 100)

    Returns:
        Final quality score (max 35% boost from h-index)
    """
    llm_relevance = paper.relevance_score

    # Calculate h_factor: normalized average h-index (0-1)
    h_factor = 0.0
    if paper.author_h_indices:
        avg_h_index = sum(paper.author_h_indices) / len(paper.author_h_indices)
        h_factor = min(avg_h_index / max_h_index, 1.0)

    # Quality boost: max 35% from h-index
    quality_boost = 0.35 * h_factor

    return llm_relevance * (1 + quality_boost)


def compute_quality_scores(
    papers: list[Paper],
    max_h_index: float | None = None,
) -> list[Paper]:
    """Compute quality scores for a list of papers.

    Uses actual max values from the papers if not provided.

    Args:
        papers: List of papers with relevance scores
        max_h_index: Maximum h-index for normalization (auto-detected if None)

    Returns:
        Same list of papers with quality_score field populated
    """
    if not papers:
        return papers

    # Auto-detect max values from papers if not provided
    if max_h_index is None:
        all_h_indices = [
            h for p in papers if p.author_h_indices for h in p.author_h_indices
        ]
        max_h_index = max(all_h_indices) if all_h_indices else 100.0

    # Ensure we don't divide by zero
    max_h_index = max(max_h_index, 1.0)

    for paper in papers:
        paper.quality_score = compute_quality_score(
            paper, max_h_index=max_h_index
        )

    return papers
