"""HuggingFace Daily Papers client."""

import logging
from datetime import datetime

import httpx

from ..models import Paper

logger = logging.getLogger(__name__)

HUGGINGFACE_API_URL = "https://huggingface.co/api/daily_papers"


class HuggingFaceClient:
    """Client for fetching papers from HuggingFace Daily Papers."""

    def __init__(self, timeout: float = 30.0):
        """Initialize client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    async def fetch_papers(self, limit: int = 50) -> list[Paper]:
        """Fetch papers from HuggingFace Daily Papers.

        Args:
            limit: Maximum number of papers to fetch (capped at 50 by API)

        Returns:
            List of Paper objects
        """
        papers: list[Paper] = []
        # HuggingFace API doesn't accept limit > 50
        limit = min(limit, 50)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    HUGGINGFACE_API_URL,
                    params={"limit": limit},
                )
                response.raise_for_status()
                data = response.json()

            for item in data:
                paper_data = item.get("paper", {})
                paper_id = paper_data.get("id", "")

                # Extract author names
                authors = [
                    author.get("name", "")
                    for author in paper_data.get("authors", [])
                    if author.get("name")
                ]

                # Parse dates
                published = item.get("publishedAt", "")
                if published:
                    try:
                        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                        published = dt.isoformat()
                    except ValueError:
                        pass

                # Get upvotes from paper data
                upvotes = paper_data.get("upvotes")

                paper = Paper(
                    arxiv_id=paper_id,
                    title=item.get("title", ""),
                    abstract=item.get("summary", ""),
                    authors=authors,
                    categories=["huggingface"],  # HF doesn't provide categories
                    published=published,
                    updated=published,
                    link=f"https://huggingface.co/papers/{paper_id}",
                    huggingface_upvotes=upvotes,
                )
                papers.append(paper)

            logger.info(f"Fetched {len(papers)} papers from HuggingFace")

        except httpx.HTTPError as e:
            logger.error(f"HuggingFace API error: {e}")
        except Exception as e:
            logger.error(f"Error fetching from HuggingFace: {e}")

        return papers
