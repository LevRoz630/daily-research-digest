# ArXiv Digest

AI-powered arXiv paper digest with LLM-based ranking and automatic scheduling.

## Features

- Fetch recent papers from arXiv by category
- Rank papers by relevance using LLMs (Anthropic, OpenAI, or Google)
- Generate daily digests with top relevant papers
- Background scheduler for automated digest generation
- Store digests as JSON files with date-based organization

## Installation

```bash
# Basic installation
pip install arxiv-digest

# With specific LLM provider
pip install arxiv-digest[anthropic]  # For Claude
pip install arxiv-digest[openai]     # For GPT
pip install arxiv-digest[google]     # For Gemini

# With all providers
pip install arxiv-digest[all]

# Development installation
pip install arxiv-digest[dev]
```

## Quick Start

```python
import asyncio
from pathlib import Path
from arxiv_digest import (
    ArxivClient,
    DigestConfig,
    DigestGenerator,
    DigestStorage,
    ArxivScheduler,
)

# Configure digest
config = DigestConfig(
    categories=["cs.AI", "cs.CL", "cs.LG"],
    interests="AI agents, large language models, natural language processing",
    max_papers=50,
    top_n=10,
    llm_provider="anthropic",
    anthropic_api_key="your-api-key-here",
)

# Set up storage
storage = DigestStorage(Path("./digests"))

# Create generator
generator = DigestGenerator(storage)

# Generate digest
async def main():
    result = await generator.generate(config)
    print(f"Status: {result['status']}")

    if result['status'] == 'completed':
        digest = result['digest']
        print(f"Generated digest with {len(digest['papers'])} papers")

        for paper in digest['papers']:
            print(f"\n{paper['relevance_score']:.1f} - {paper['title']}")
            print(f"  {paper['link']}")
            print(f"  Reason: {paper['relevance_reason']}")

asyncio.run(main())
```

## Scheduled Digests

```python
import asyncio
from arxiv_digest import ArxivScheduler, DigestGenerator, DigestStorage, DigestConfig
from pathlib import Path

config = DigestConfig(
    categories=["cs.AI", "cs.LG"],
    interests="machine learning research",
    llm_provider="anthropic",
    anthropic_api_key="your-key",
)

storage = DigestStorage(Path("./digests"))
generator = DigestGenerator(storage)
scheduler = ArxivScheduler(generator, schedule_hour=6)  # 6 AM UTC

async def run_scheduler():
    # Start scheduler (runs daily at 6 AM UTC)
    scheduler.start(config)

    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        scheduler.stop()

asyncio.run(run_scheduler())
```

## Configuration

### DigestConfig

- `categories`: List of arXiv category codes (e.g., `["cs.AI", "cs.LG"]`)
- `interests`: Research interests description for ranking
- `max_papers`: Maximum papers to fetch (default: 50)
- `top_n`: Number of top papers to include in digest (default: 10)
- `llm_provider`: One of "anthropic", "openai", or "google"
- API keys for your chosen provider

### Common arXiv Categories

| Category | Description |
|----------|-------------|
| cs.AI | Artificial Intelligence |
| cs.CL | Computation and Language (NLP) |
| cs.LG | Machine Learning |
| cs.CV | Computer Vision |
| cs.NE | Neural and Evolutionary Computing |
| stat.ML | Machine Learning (Statistics) |
| q-fin.ST | Statistical Finance |
| q-fin.PM | Portfolio Management |

Full taxonomy: https://arxiv.org/category_taxonomy

## LLM Providers

The package supports multiple LLM providers for paper ranking:

| Provider | Model | Package Required |
|----------|-------|------------------|
| anthropic | claude-3-haiku-20240307 | langchain-anthropic |
| openai | gpt-3.5-turbo | langchain-openai |
| google | gemini-1.5-flash | langchain-google-genai |

Each uses fast, cost-effective models optimized for ranking tasks.

## Digest Format

Digests are saved as JSON files with the following structure:

```json
{
  "date": "2024-01-15",
  "generated_at": "2024-01-15T06:00:00Z",
  "categories": ["cs.AI", "cs.CL"],
  "interests": "AI agents, LLMs",
  "total_papers_fetched": 50,
  "papers": [
    {
      "arxiv_id": "2401.12345",
      "title": "Paper Title",
      "abstract": "Abstract text...",
      "authors": ["Author One", "Author Two"],
      "categories": ["cs.AI", "cs.CL"],
      "published": "2024-01-14T00:00:00Z",
      "updated": "2024-01-14T00:00:00Z",
      "link": "https://arxiv.org/abs/2401.12345",
      "relevance_score": 9.0,
      "relevance_reason": "Directly addresses AI agent architectures"
    }
  ]
}
```

## API Reference

### ArxivClient

Fetches papers from arXiv API.

```python
client = ArxivClient(timeout=30.0)
papers = await client.fetch_papers(["cs.AI"], max_results=50)
```

### DigestGenerator

Generates paper digests.

```python
storage = DigestStorage(Path("./digests"))
generator = DigestGenerator(storage)
result = await generator.generate(config)
```

### DigestStorage

Manages digest persistence.

```python
storage = DigestStorage(Path("./digests"))
storage.save_digest(digest)
digest = storage.get_digest("2024-01-15")
dates = storage.list_digests(limit=30)
```

### ArxivScheduler

Schedules automated digest generation.

```python
scheduler = ArxivScheduler(generator, schedule_hour=6)
scheduler.start(config)
scheduler.stop()
```

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/arxiv-digest.git
cd arxiv-digest

# Install with dev dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Format code
black arxiv_digest tests

# Lint
ruff arxiv_digest tests

# Type check
mypy arxiv_digest
```

## License

MIT License - see LICENSE file for details.
