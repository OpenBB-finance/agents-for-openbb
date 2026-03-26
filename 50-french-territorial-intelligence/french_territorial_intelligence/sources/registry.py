from __future__ import annotations

from french_territorial_intelligence.sources.base import DataSource


class SourceRegistry:
    """Central registry for data sources. Extensible by design."""

    def __init__(self) -> None:
        self._sources: dict[str, DataSource] = {}

    def register(self, source: DataSource) -> None:
        self._sources[source.name] = source

    def get(self, name: str) -> DataSource | None:
        return self._sources.get(name)

    def get_all(self) -> list[DataSource]:
        return list(self._sources.values())

    def available_metrics(self) -> list[str]:
        return [m for source in self._sources.values() for m in source.available_metrics()]

    def clear(self) -> None:
        self._sources.clear()


registry = SourceRegistry()
