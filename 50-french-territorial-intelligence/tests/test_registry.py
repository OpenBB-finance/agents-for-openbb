from french_territorial_intelligence.sources.registry import registry


class FakeSource:
    name = "fake"
    description = "A fake data source for testing"

    async def fetch_territory(self, code_commune: str) -> dict:
        return {"population": 1000}

    async def search(self, query: str) -> list[dict]:
        return [{"name": "Fakeville"}]

    def available_metrics(self) -> list[str]:
        return ["population"]


def test_register_and_retrieve():
    registry.clear()
    source = FakeSource()
    registry.register(source)
    assert len(registry.get_all()) == 1
    assert registry.get("fake") is source


def test_registry_clear():
    registry.clear()
    registry.register(FakeSource())
    registry.clear()
    assert len(registry.get_all()) == 0


def test_available_metrics_aggregated():
    registry.clear()
    registry.register(FakeSource())
    assert "population" in registry.available_metrics()
