"""Semantic Scholar client."""

import logging

import httpx

from ..models import Paper

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarClient:
    """Client for fetching papers from Semantic Scholar."""

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        """Initialize client.

        Args:
            api_key: Semantic Scholar API key (optional but recommended)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout

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
            "fields": "paperId,externalIds,title,abstract,authors,year,fieldsOfStudy",
        }

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        if year:
            params["year"] = year

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    SEMANTIC_SCHOLAR_API_URL,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

            for item in data.get("data", []):
                # Get arxiv ID if available
                external_ids = item.get("externalIds") or {}
                arxiv_id = external_ids.get("ArXiv", "")

                # Skip if no arxiv ID (can't dedupe reliably)
                if not arxiv_id:
                    # Use paperId as fallback
                    arxiv_id = f"s2:{item.get('paperId', '')}"

                # Extract author names
                authors = [
                    author.get("name", "")
                    for author in item.get("authors", [])
                    if author.get("name")
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
                )
                papers.append(paper)

            logger.info(f"Fetched {len(papers)} papers from Semantic Scholar")

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar API error: {e}")
        except Exception as e:
            logger.error(f"Error fetching from Semantic Scholar: {e}")

        return papers
