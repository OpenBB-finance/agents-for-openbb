import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from french_territorial_intelligence.sources.entreprises import EntreprisesSource


@pytest.fixture
def ent():
    return EntreprisesSource()


API_RESPONSE = {
    "results": [
        {
            "siren": "403052111",
            "nom_complet": "BOULANGERIES PAUL",
            "activite_principale": "10.71A",
            "categorie_entreprise": "ETI",
            "nombre_etablissements_ouverts": 144,
            "date_creation": "1995-12-07",
            "siege": {
                "commune": "59378",
                "departement": "59",
                "tranche_effectif_salarie": "12",
                "etat_administratif": "A",
            },
        }
    ],
    "total_results": 624,
    "page": 1,
    "per_page": 25,
    "total_pages": 25,
}


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_fetch_territory(ent):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(API_RESPONSE)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await ent.fetch_territory("59378")
    assert result["total_enterprises"] == 624
    assert len(result["enterprises"]) == 1
    assert result["enterprises"][0]["name"] == "BOULANGERIES PAUL"


@pytest.mark.asyncio
async def test_sector_breakdown(ent):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(API_RESPONSE)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await ent.fetch_territory("59378")
    assert "sector_breakdown" in result
    assert "Manufacturing" in result["sector_breakdown"]


@pytest.mark.asyncio
async def test_search(ent):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(API_RESPONSE)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        results = await ent.search("boulangerie")
    assert len(results) == 1
    assert results[0]["name"] == "BOULANGERIES PAUL"


def test_available_metrics(ent):
    metrics = ent.available_metrics()
    assert "total_enterprises" in metrics
    assert "sector_breakdown" in metrics
