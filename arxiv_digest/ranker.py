"""LLM-based paper ranking."""

import asyncio
import json
import logging
import re

from .models import Paper

logger = logging.getLogger(__name__)


class PaperRanker:
    """Ranks papers by relevance using LLMs."""

    def __init__(self, llm, batch_size: int = 5, batch_delay: float = 1.0):
        """Initialize ranker.

        Args:
            llm: LangChain LLM instance
            batch_size: Number of papers to rank concurrently
            batch_delay: Delay in seconds between batches
        """
        self.llm = llm
        self.batch_size = batch_size
        self.batch_delay = batch_delay

    async def rank_paper(self, paper: Paper, interests: str) -> Paper:
        """Rank a single paper's relevance to interests.

        Args:
            paper: Paper to rank
            interests: Research interests description

        Returns:
            Paper with relevance_score and relevance_reason set
        """
        prompt = f"""Rate this paper's relevance to the following research interests on a scale of 1-10.
Be strict - only give 8+ for papers directly relevant to the interests.

Research interests: {interests}

Paper title: {paper.title}
Abstract: {paper.abstract[:1000]}

Respond with ONLY a JSON object in this exact format (no other text):
{{"score": <number 1-10>, "reason": "<brief 1-sentence explanation>"}}"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```" in content:
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
                content = match.group(1) if match else "{}"

            result = json.loads(content)
            paper.relevance_score = float(result.get("score", 0))
            paper.relevance_reason = result.get("reason", "")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse ranking for {paper.arxiv_id}: {e}")
            paper.relevance_score = 5.0
            paper.relevance_reason = "Unable to rank"

        return paper

    async def rank_papers(self, papers: list[Paper], interests: str) -> list[Paper]:
        """Rank all papers by relevance to interests.

        Args:
            papers: List of papers to rank
            interests: Research interests description

        Returns:
            List of ranked papers sorted by relevance_score descending
        """
        ranked_papers = []

        # Rank papers in batches to avoid rate limits
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i : i + self.batch_size]
            tasks = [self.rank_paper(paper, interests) for paper in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Paper):
                    ranked_papers.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Ranking error: {result}")

            # Small delay between batches
            if i + self.batch_size < len(papers):
                await asyncio.sleep(self.batch_delay)

        # Sort by relevance score descending
        ranked_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return ranked_papers


def get_llm_for_provider(
    provider: str,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
    google_api_key: str | None = None,
):
    """Get LLM instance for the specified provider.

    Args:
        provider: One of "anthropic", "openai", or "google"
        anthropic_api_key: Anthropic API key (required for anthropic provider)
        openai_api_key: OpenAI API key (required for openai provider)
        google_api_key: Google API key (required for google provider)

    Returns:
        LangChain LLM instance

    Raises:
        ValueError: If provider is unknown or API key is missing
        ImportError: If required langchain package is not installed
    """
    if provider == "anthropic":
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key is required for anthropic provider")
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model="claude-3-haiku-20240307",
                api_key=anthropic_api_key,
                max_tokens=256,
            )
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic not installed. Install with: pip install arxiv-digest[anthropic]"
            ) from e

    elif provider == "openai":
        if not openai_api_key:
            raise ValueError("openai_api_key is required for openai provider")
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=openai_api_key,
                max_tokens=256,
            )
        except ImportError as e:
            raise ImportError(
                "langchain-openai not installed. Install with: pip install arxiv-digest[openai]"
            ) from e

    elif provider == "google":
        if not google_api_key:
            raise ValueError("google_api_key is required for google provider")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_api_key,
                max_output_tokens=256,
            )
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai not installed. Install with: pip install arxiv-digest[google]"
            ) from e

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic', 'openai', or 'google'")
