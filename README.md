# Daily Research Digest

AI-powered research paper digest that fetches papers from Semantic Scholar, ranks them by relevance using LLMs, and delivers daily email digests.

## Installation

```bash
pip install daily-research-digest[anthropic]  # or [openai], [google], [all]
```

## Quick Start

### Python API

```python
import asyncio
from pathlib import Path
from daily_research_digest import DigestConfig, DigestGenerator, DigestStorage

config = DigestConfig(
    categories=[],
    interests="AI agents, large language models",
    sources=["semantic_scholar"],
    llm_provider="anthropic",
    anthropic_api_key="your-api-key",
)

async def main():
    generator = DigestGenerator(DigestStorage(Path("./digests")))
    result = await generator.generate(config)
    for paper in result['digest']['papers']:
        print(f"{paper['relevance_score']:.1f} - {paper['title']}")

asyncio.run(main())
```

### CLI (Email Digest)

```bash
export DIGEST_RECIPIENTS="you@example.com"
export DIGEST_INTERESTS="machine learning, AI agents"
export SMTP_HOST="smtp.gmail.com"
export SMTP_USER="your-email@gmail.com"
export SMTP_PASS="your-app-password"
export ANTHROPIC_API_KEY="your-api-key"

python -m daily_research_digest.digest_send
```

## GitHub Actions

The workflow at `.github/workflows/digest.yml` sends daily emails at 6 AM UTC.

**Required secrets:** `DIGEST_RECIPIENTS`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, `ANTHROPIC_API_KEY`

**Optional variables:** `DIGEST_INTERESTS`, `DIGEST_WINDOW` (24h/48h/7d), `LLM_PROVIDER`

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `interests` | - | Research interests for search and ranking |
| `max_papers` | 50 | Max papers to fetch |
| `top_n` | 10 | Papers in final digest |
| `llm_provider` | anthropic | `anthropic`, `openai`, or `google` |

## LLM Providers

| Provider | Model |
|----------|-------|
| anthropic | claude-3-haiku |
| openai | gpt-3.5-turbo |
| google | gemini-1.5-flash |

## Development

```bash
git clone https://github.com/LevRoz630/daily-research-digest.git
cd daily-research-digest
pip install -e ".[dev,all]"
pytest
```

## License

MIT
