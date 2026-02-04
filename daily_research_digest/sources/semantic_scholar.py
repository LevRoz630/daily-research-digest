"""Semantic Scholar client."""

import asyncio
import logging

import httpx

from ..models import Paper

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarClient:
    """Client for fetching papers from Semantic Scholar."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize client.

        Args:
            api_key: Semantic Scholar API key (optional but recommended)
            timeout: Request timeout in seconds
            max_retries: Max retries on rate limit (429) errors
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    async def fetch_papers(
        self,
        query: str,
        limit: int = 50,
        fields_of_study: list[str] | None = None,
        year: str | None = None,
    ) -> list[Paper]:
        """Fetch papers from Semantic Scholar.

        Args:
            query: Search query (keywords)
            limit: Maximum number of papers to fetch
            fields_of_study: Filter by fields (e.g., ["Computer Science"])
            year: Filter by year (e.g., "2024" or "2023-2024")

        Returns:
            List of Paper objects
        """
        papers: list[Paper] = []

        params = {
            "query": query,
            "limit": min(limit, 100),  # API max is 100
            "fields": "paperId,externalIds,title,abstract,authors,authors.hIndex,year,fieldsOfStudy",  # noqa: E501
        }

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        if year:
            params["year"] = year

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            print(f"Querying Semantic Scholar with: {params['query']}")
            data = await self._fetch_with_retry(params, headers)
            print(f"Semantic Scholar returned {len(data.get('data', []))} results")

            for item in data.get("data", []):
                # Get arxiv ID if available
                external_ids = item.get("externalIds") or {}
                arxiv_id = external_ids.get("ArXiv", "")

                # Skip if no arxiv ID (can't dedupe reliably)
                if not arxiv_id:
                    # Use paperId as fallback
                    arxiv_id = f"s2:{item.get('paperId', '')}"

                # Extract author names and h-indices
                authors = [
                    author.get("name", "")
                    for author in item.get("authors", [])
                    if author.get("name")
                ]
                author_h_indices = [
                    author.get("hIndex")
                    for author in item.get("authors", [])
                    if author.get("hIndex") is not None
                ]

                # Get categories from fieldsOfStudy
                categories = item.get("fieldsOfStudy") or ["semantic_scholar"]

                year_val = item.get("year")
                published = f"{year_val}-01-01" if year_val else ""

                paper = Paper(
                    arxiv_id=arxiv_id,
                    title=item.get("title", ""),
                    abstract=item.get("abstract") or "",
                    authors=authors,
                    categories=categories,
                    published=published,
                    updated=published,
                    link=f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                    author_h_indices=author_h_indices if author_h_indices else None,
                )
                papers.append(paper)

            logger.info(f"Fetched {len(papers)} papers from Semantic Scholar")

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar API error: {e}")
        except Exception as e:
            logger.error(f"Error fetching from Semantic Scholar: {e}")

        return papers

    async def _fetch_with_retry(self, params: dict, headers: dict) -> dict:
        """Fetch with retry on rate limit errors.

        Args:
            params: Request parameters
            headers: Request headers

        Returns:
            JSON response data

        Raises:
            httpx.HTTPError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    SEMANTIC_SCHOLAR_API_URL,
                    params=params,
                    headers=headers,
                )

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2**attempt  # 1, 2, 4 seconds
                    logger.warning(
                        f"Rate limited by Semantic Scholar, waiting {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    last_error = httpx.HTTPStatusError(
                        "Rate limited", request=response.request, response=response
                    )
                    continue

                response.raise_for_status()
                result: dict = response.json()
                return result

        # All retries failed
        if last_error:
            raise last_error
        return {"data": []}
