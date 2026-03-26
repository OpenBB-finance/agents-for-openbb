from unittest.mock import patch

import pytest

from french_territorial_intelligence.sources.dvf import DvfSource
from tests.conftest import mock_httpx_response, patch_httpx_client


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


@pytest.mark.asyncio
async def test_fetch_territory(dvf):
    with patch("httpx.AsyncClient") as mock_cls:
        patch_httpx_client(mock_cls, mock_httpx_response(API_RESPONSE))
        result = await dvf.fetch_territory("69381")
    assert result["total_transactions"] == 1134
    assert result["avg_price_sqm"] > 0
    assert "property_type_breakdown" in result


@pytest.mark.asyncio
async def test_avg_price_calculation(dvf):
    with patch("httpx.AsyncClient") as mock_cls:
        patch_httpx_client(mock_cls, mock_httpx_response(API_RESPONSE))
        result = await dvf.fetch_territory("69381")
    # 297000/62 = 4790.32, 450000/95 = 4736.84 => avg ~4763.58
    assert 4700 < result["avg_price_sqm"] < 4900


def test_available_metrics(dvf):
    metrics = dvf.available_metrics()
    assert "avg_price_sqm" in metrics
    assert "total_transactions" in metrics
