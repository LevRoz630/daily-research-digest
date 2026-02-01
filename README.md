# Daily Research Digest

AI-powered research paper digest with LLM-based ranking and automatic scheduling. Fetches papers from arXiv, HuggingFace Daily Papers, and Semantic Scholar.

## Features

- Fetch recent papers from multiple sources (arXiv, HuggingFace, Semantic Scholar)
- Rank papers by relevance using LLMs (Anthropic, OpenAI, or Google)
- Generate daily digests with top relevant papers
- Background scheduler for automated digest generation
- Email digest delivery via SMTP (GitHub Actions compatible)
- Store digests as JSON files with date-based organization

## Installation

```bash
# Basic installation
pip install daily-research-digest

# With specific LLM provider
pip install daily-research-digest[anthropic]  # For Claude
pip install daily-research-digest[openai]     # For GPT
pip install daily-research-digest[google]     # For Gemini

# With all providers
pip install daily-research-digest[all]

# Development installation
pip install daily-research-digest[dev]
```

## Quick Start

```python
import asyncio
from pathlib import Path
from daily_research_digest import (
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
from daily_research_digest import ArxivScheduler, DigestGenerator, DigestStorage, DigestConfig
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

## GitHub Actions Cron Usage

Send daily digest emails using GitHub Actions. The digest runner supports:

- Configurable time windows (24h, 48h, 7d)
- Idempotent execution (won't re-send on workflow reruns)
- Multiple LLM providers
- SMTP email delivery
- Structured JSON logging

### Quick Setup

1. **Add repository secrets** (Settings > Secrets and variables > Actions):

   | Secret | Required | Description |
   |--------|----------|-------------|
   | `DIGEST_RECIPIENTS` | Yes | Comma-separated email addresses |
   | `SMTP_HOST` | Yes | SMTP server hostname |
   | `SMTP_USER` | No | SMTP username |
   | `SMTP_PASS` | No | SMTP password |
   | `ANTHROPIC_API_KEY` | Yes* | Anthropic API key |
   | `OPENAI_API_KEY` | Alt | OpenAI API key |
   | `GOOGLE_API_KEY` | Alt | Google API key |

   *Required if using Anthropic (default). Use OpenAI or Google key with corresponding `LLM_PROVIDER`.

2. **Add repository variables** (optional, for customization):

   | Variable | Default | Description |
   |----------|---------|-------------|
   | `DIGEST_CATEGORIES` | `cs.AI,cs.LG,cs.CL` | arXiv categories |
   | `DIGEST_INTERESTS` | `machine learning...` | Research interests |
   | `DIGEST_SUBJECT` | `Daily Research Digest - {date}` | Email subject |
   | `DIGEST_TZ` | `UTC` | Timezone |
   | `DIGEST_WINDOW` | `24h` | Time window |
   | `LLM_PROVIDER` | `anthropic` | LLM provider |

3. **Enable the workflow**: The `.github/workflows/digest.yml` file runs daily at 6 AM UTC.

### Manual Trigger

You can manually trigger the digest from the Actions tab using "Run workflow".

### CLI Usage

Run the digest sender locally:

```bash
# Set required environment variables
export DIGEST_RECIPIENTS="you@example.com"
export DIGEST_CATEGORIES="cs.AI,cs.LG"
export DIGEST_INTERESTS="machine learning, AI agents"
export SMTP_HOST="smtp.gmail.com"
export SMTP_USER="your-email@gmail.com"
export SMTP_PASS="your-app-password"
export ANTHROPIC_API_KEY="your-api-key"

# Run the digest sender
python -m daily_research_digest.digest_send
```

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DIGEST_RECIPIENTS` | Yes | - | Comma-separated email addresses |
| `DIGEST_CATEGORIES` | Yes | - | Comma-separated arXiv categories |
| `DIGEST_INTERESTS` | Yes | - | Research interests for ranking |
| `DIGEST_SUBJECT` | No | `Daily Research Digest - {date}` | Email subject (supports `{date}`) |
| `DIGEST_FROM` | No | `noreply@example.com` | Sender address |
| `DIGEST_TZ` | No | `UTC` | Timezone for window calculation |
| `DIGEST_WINDOW` | No | `24h` | Time window (`24h`, `1d`, `48h`, `7d`) |
| `DIGEST_MAX_PAPERS` | No | `50` | Max papers to fetch |
| `DIGEST_TOP_N` | No | `10` | Top papers in digest |
| `SMTP_HOST` | Yes | - | SMTP server hostname |
| `SMTP_PORT` | No | `587` | SMTP server port |
| `SMTP_USER` | No | - | SMTP username |
| `SMTP_PASS` | No | - | SMTP password |
| `SMTP_TLS` | No | `true` | Use TLS (`true`/`false`) |
| `LLM_PROVIDER` | No | `anthropic` | `anthropic`, `openai`, or `google` |
| `ANTHROPIC_API_KEY` | * | - | Required for anthropic provider |
| `OPENAI_API_KEY` | * | - | Required for openai provider |
| `GOOGLE_API_KEY` | * | - | Required for google provider |

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
git clone https://github.com/LevRoz630/daily-research-digest.git
cd daily-research-digest

# Install with dev dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Format code
black daily_research_digest tests

# Lint
ruff daily_research_digest tests

# Type check
mypy daily_research_digest
```

## License

MIT License - see LICENSE file for details.
