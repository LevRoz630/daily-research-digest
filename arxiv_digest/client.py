"""ArXiv API client for fetching papers."""

import logging
import xml.etree.ElementTree as ET

import httpx

from .models import Paper

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for fetching papers from arXiv API."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, timeout: float = 30.0):
        """Initialize client.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout

    async def fetch_papers(
        self, categories: list[str], max_results: int = 50
    ) -> list[Paper]:
        """Fetch recent papers from arXiv API for given categories.

        Args:
            categories: List of arXiv category codes (e.g. ["cs.AI", "cs.LG"])
            max_results: Maximum number of papers to fetch

        Returns:
            List of Paper objects
        """
        papers = []

        # Build query for multiple categories (OR them together)
        cat_query = "+OR+".join([f"cat:{cat}" for cat in categories])
        url = (
            f"{self.BASE_URL}?"
            f"search_query={cat_query}&"
            f"sortBy=submittedDate&"
            f"sortOrder=descending&"
            f"max_results={max_results}"
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch arXiv papers: {e}")
                return papers

        # Parse XML response
        try:
            root = ET.fromstring(response.text)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            for entry in root.findall("atom:entry", ns):
                # Extract arxiv ID from the id URL
                id_url = entry.find("atom:id", ns).text
                arxiv_id = id_url.split("/abs/")[-1]

                # Get categories
                cats = [
                    cat.get("term")
                    for cat in entry.findall("atom:category", ns)
                    if cat.get("term")
                ]

                paper = Paper(
                    arxiv_id=arxiv_id,
                    title=entry.find("atom:title", ns).text.strip().replace("\n", " "),
                    abstract=entry.find("atom:summary", ns)
                    .text.strip()
                    .replace("\n", " "),
                    authors=[
                        a.find("atom:name", ns).text
                        for a in entry.findall("atom:author", ns)
                    ],
                    categories=cats,
                    published=entry.find("atom:published", ns).text,
                    updated=entry.find("atom:updated", ns).text,
                    link=f"https://arxiv.org/abs/{arxiv_id}",
                )
                papers.append(paper)

        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML: {e}")

        return papers
