from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from french_territorial_intelligence.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_agents_json(client):
    resp = client.get("/agents.json")
    assert resp.status_code == 200
    data = resp.json()
    assert "french-territorial-intelligence" in data
    agent = data["french-territorial-intelligence"]
    assert "name" in agent
    assert "description" in agent
    assert agent["endpoints"]["query"] == "/v1/query"
    assert agent["features"]["streaming"] is True


def test_query_returns_stream(client):
    """Test that /v1/query returns a streaming response.
    We mock the OpenAI client to avoid needing an API key."""
    mock_choice = MagicMock()
    mock_choice.message.tool_calls = None
    mock_choice.message.content = "Lyon is a dynamic city."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("french_territorial_intelligence.agent.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        resp = client.post(
            "/v1/query",
            json={"messages": [{"role": "human", "content": "Tell me about Lyon"}]},
        )
    assert resp.status_code == 200


def test_query_with_conversation(client):
    """Test multi-turn conversation with human/ai roles."""
    mock_choice = MagicMock()
    mock_choice.message.tool_calls = None
    mock_choice.message.content = "Marseille has a strong economy."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("french_territorial_intelligence.agent.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        resp = client.post(
            "/v1/query",
            json={
                "messages": [
                    {"role": "human", "content": "Tell me about Lyon"},
                    {"role": "ai", "content": "Lyon is the third largest city."},
                    {"role": "human", "content": "Now tell me about Marseille"},
                ]
            },
        )
    assert resp.status_code == 200
