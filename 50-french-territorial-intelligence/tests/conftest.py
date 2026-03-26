from unittest.mock import AsyncMock, MagicMock


def mock_httpx_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx response with given JSON data."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def patch_httpx_client(mock_client_cls: MagicMock, response: MagicMock) -> None:
    """Wire up a patched httpx.AsyncClient context manager to return the given response."""
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
