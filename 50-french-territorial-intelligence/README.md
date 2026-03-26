# French Territorial Intelligence Agent

Cross-reference French open data to produce actionable territorial intelligence.
This agent combines three zero-auth public APIs to analyze any French city:

- **Recherche Entreprises** — Business registry (SIRENE): enterprise density, sector breakdown
- **DVF** — Real estate transactions: price per sqm, transaction volume, property types
- **Geo API** — Administrative geography: population, surface, density

The key differentiator is **cross-referencing**: the agent doesn't just query APIs —
it computes derived indicators (affordability index, economic intensity, enterprise
density) by combining data across sources, producing insights no single API can provide.

> Built on [Tawiza](https://github.com/tawiza/tawiza)'s territorial intelligence engine.

## Requirements

- Python 3.10+
- Poetry
- `OPENAI_API_KEY` environment variable

## Setup

```bash
# From the repository root
poetry install --no-root

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

## Run

```bash
poetry run uvicorn french_territorial_intelligence.main:app --port 7777 --reload
```

## Test

```bash
python -m pytest 50-french-territorial-intelligence/tests/ -v
```

## API Documentation

http://localhost:7777/docs

## Example Queries

- "Analyze the business dynamics in Marseille"
- "What's the real estate market like in Lyon?"
- "Compare Bordeaux and Nantes for a tech startup"
- "How affordable is Toulouse for new businesses?"
- "Search for biotech companies in France"

## Cross-Referenced Indicators

| Indicator | Formula | Interpretation |
|-----------|---------|----------------|
| Enterprise density | enterprises / population * 1000 | Economic maturity of the territory |
| Affordability index | avg price per sqm / enterprise density | Cost vs. economic activity balance |
| Economic intensity | (enterprises * density) / population | Concentration of economic activity |

## Architecture

```
sources/         Data source adapters (extensible via Protocol pattern)
  base.py        DataSource protocol — implement this to add a new API
  registry.py    Auto-discovery registry
  geo.py         Geo API (communes, population, geography)
  entreprises.py Recherche Entreprises (SIRENE business data)
  dvf.py         DVF (real estate transactions)
crossref.py      Cross-referencing engine (the core differentiator)
agent.py         LLM orchestration + OpenAI function calling
main.py          FastAPI application
```

## Adding a New Data Source

1. Create a file in `sources/` implementing the `DataSource` protocol
2. Call `registry.register(YourSource())` at module level
3. The crossref engine and agent automatically pick it up

```python
# sources/education.py
from french_territorial_intelligence.sources.registry import registry


class EducationSource:
    name = "education"
    description = "French education data (schools, universities)"

    async def fetch_territory(self, code_commune: str) -> dict:
        # Call your API here
        ...

    async def search(self, query: str) -> list[dict]:
        ...

    def available_metrics(self) -> list[str]:
        return ["schools", "universities", "student_population"]


registry.register(EducationSource())
```

## Data Sources

All APIs are **zero-auth** — no API keys or registration required.

| Source | API | Coverage | Update Frequency |
|--------|-----|----------|-----------------|
| Recherche Entreprises | recherche-entreprises.api.gouv.fr | All French businesses | Daily |
| DVF | apidf-preprod.cerema.fr | Real estate since 2014 | Biannual |
| Geo API | geo.api.gouv.fr | All French communes | Annual (census) |
