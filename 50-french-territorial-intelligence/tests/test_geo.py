import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from french_territorial_intelligence.sources.geo import GeoSource


@pytest.fixture
def geo():
    return GeoSource()


GEO_RESPONSE_LIST = [
    {
        "nom": "Lyon",
        "code": "69123",
        "population": 519127,
        "surface": 4797.43,
        "codesPostaux": ["69001", "69002", "69003"],
        "departement": {"code": "69", "nom": "Rhone"},
        "centre": {"type": "Point", "coordinates": [4.8351, 45.758]},
    }
]

GEO_RESPONSE_SINGLE = GEO_RESPONSE_LIST[0]


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_fetch_territory(geo):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(GEO_RESPONSE_SINGLE)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await geo.fetch_territory("69123")
    assert result["name"] == "Lyon"
    assert result["population"] == 519127
    assert result["department_code"] == "69"
    assert result["surface_km2"] == pytest.approx(47.97, abs=0.01)
    assert result["density"] > 0


@pytest.mark.asyncio
async def test_search(geo):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(GEO_RESPONSE_LIST)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        results = await geo.search("Lyon")
    assert len(results) == 1
    assert results[0]["name"] == "Lyon"


def test_available_metrics(geo):
    metrics = geo.available_metrics()
    assert "population" in metrics
    assert "surface" in metrics
