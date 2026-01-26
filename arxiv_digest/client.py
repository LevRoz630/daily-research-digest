"""ArXiv API client for fetching papers."""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx

from .models import DateFilter, Paper

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

    def _filter_by_date(self, papers: list[Paper], date_filter: DateFilter) -> list[Paper]:
        """Filter papers by date criteria.

        Args:
            papers: List of papers to filter
            date_filter: Date filtering options

        Returns:
            Filtered list of papers
        """
        if not date_filter:
            return papers

        filtered = []
        now = datetime.now(timezone.utc)

        for paper in papers:
            # Parse the published date
            try:
                published = datetime.fromisoformat(paper.published.replace("Z", "+00:00"))
            except ValueError:
                # If we can't parse the date, include the paper
                filtered.append(paper)
                continue

            # Check days_back filter
            if date_filter.days_back is not None:
                cutoff = now - timedelta(days=date_filter.days_back)
                if published < cutoff:
                    continue

            # Check published_after filter
            if date_filter.published_after is not None:
                after_date = datetime.fromisoformat(date_filter.published_after).replace(
                    tzinfo=timezone.utc
                )
                if published < after_date:
                    continue

            # Check published_before filter
            if date_filter.published_before is not None:
                before_date = datetime.fromisoformat(date_filter.published_before).replace(
                    tzinfo=timezone.utc
                )
                if published > before_date:
                    continue

            filtered.append(paper)

        return filtered

    async def fetch_papers(
        self,
        categories: list[str],
        max_results: int = 50,
        date_filter: DateFilter | None = None,
    ) -> list[Paper]:
        """Fetch recent papers from arXiv API for given categories.

        Args:
            categories: List of arXiv category codes (e.g. ["cs.AI", "cs.LG"])
            max_results: Maximum number of papers to fetch
            date_filter: Optional date filtering options

        Returns:
            List of Paper objects
        """
        papers: list[Paper] = []

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
                id_elem = entry.find("atom:id", ns)
                if id_elem is None or id_elem.text is None:
                    continue
                arxiv_id = id_elem.text.split("/abs/")[-1]

                # Get title
                title_elem = entry.find("atom:title", ns)
                if title_elem is None or title_elem.text is None:
                    continue
                title = title_elem.text.strip().replace("\n", " ")

                # Get abstract
                abstract_elem = entry.find("atom:summary", ns)
                if abstract_elem is None or abstract_elem.text is None:
                    continue
                abstract = abstract_elem.text.strip().replace("\n", " ")

                # Get authors
                authors: list[str] = []
                for a in entry.findall("atom:author", ns):
                    name_elem = a.find("atom:name", ns)
                    if name_elem is not None and name_elem.text is not None:
                        authors.append(name_elem.text)

                # Get categories
                cats: list[str] = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term")
                    if term:
                        cats.append(term)

                # Get dates
                published_elem = entry.find("atom:published", ns)
                updated_elem = entry.find("atom:updated", ns)
                if published_elem is None or published_elem.text is None:
                    continue
                if updated_elem is None or updated_elem.text is None:
                    continue

                paper = Paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    categories=cats,
                    published=published_elem.text,
                    updated=updated_elem.text,
                    link=f"https://arxiv.org/abs/{arxiv_id}",
                )
                papers.append(paper)

        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML: {e}")

        # Apply date filtering if specified
        if date_filter:
            papers = self._filter_by_date(papers, date_filter)
            logger.info(f"After date filtering: {len(papers)} papers")

        return papers
