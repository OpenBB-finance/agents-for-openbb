import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from french_territorial_intelligence.sources.dvf import DvfSource


@pytest.fixture
def dvf():
    return DvfSource()


API_RESPONSE = {
    "count": 1134,
    "results": [
        {
            "valeurfonc": "297000.00",
            "sbati": "62.00",
            "libtypbien": "UN APPARTEMENT",
            "libnatmut": "Vente",
            "datemut": "2023-06-15",
            "anneemut": 2023,
            "coddep": "69",
        },
        {
            "valeurfonc": "450000.00",
            "sbati": "95.00",
            "libtypbien": "UNE MAISON",
            "libnatmut": "Vente",
            "datemut": "2023-07-20",
            "anneemut": 2023,
            "coddep": "69",
        },
    ],
}


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_fetch_territory(dvf):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(API_RESPONSE)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await dvf.fetch_territory("69381")
    assert result["total_transactions"] == 1134
    assert result["avg_price_sqm"] > 0
    assert "property_type_breakdown" in result


@pytest.mark.asyncio
async def test_avg_price_calculation(dvf):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(API_RESPONSE)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await dvf.fetch_territory("69381")
    # 297000/62 = 4790.32, 450000/95 = 4736.84 => avg ~4763.58
    assert 4700 < result["avg_price_sqm"] < 4900


def test_available_metrics(dvf):
    metrics = dvf.available_metrics()
    assert "avg_price_sqm" in metrics
    assert "total_transactions" in metrics
