from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DataSource(Protocol):
    """Contract for territorial data sources.

    Each source provides data about a French territory (commune).
    Implement this protocol and call registry.register() to add a new source.
    """

    name: str
    description: str

    async def fetch_territory(self, code_commune: str) -> dict:
        """Fetch structured data for a commune by its INSEE code."""
        ...

    async def search(self, query: str) -> list[dict]:
        """Free-text search. Returns a list of matching results."""
        ...

    def available_metrics(self) -> list[str]:
        """List of metric keys this source contributes."""
        ...
