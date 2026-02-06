# Daily Research Digest

AI-powered research paper digest that fetches papers from Semantic Scholar and ranks them by relevance using LLMs.

## Installation

```bash
pip install daily-research-digest[anthropic]  # or [openai], [google], [all]
```

## Quick Start

```python
import asyncio
from pathlib import Path
from daily_research_digest import DigestConfig, DigestGenerator, DigestStorage

config = DigestConfig(
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

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `interests` | - | Research interests for search and ranking |
| `max_papers` | 50 | Max papers to fetch |
| `top_n` | 10 | Papers in final digest |
| `batch_size` | 25 | Concurrent LLM ranking calls per batch |
| `batch_delay` | 0.2 | Delay in seconds between ranking batches |
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
