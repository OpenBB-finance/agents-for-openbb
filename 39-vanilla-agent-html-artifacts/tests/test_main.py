"""Tests for the HTML artifacts agent."""

import json

from fastapi.testclient import TestClient

from vanilla_agent_html.main import app, html_artifact

client = TestClient(app)


def test_agents_json():
    """Test that the agents.json endpoint returns valid configuration."""
    response = client.get("/agents.json")
    assert response.status_code == 200

    data = response.json()
    assert "vanilla_agent_html" in data
    assert data["vanilla_agent_html"]["name"] == "HTML Artifacts Agent"
    assert data["vanilla_agent_html"]["endpoints"]["query"] == "/v1/query"


def test_html_artifact_helper():
    """Test that the html_artifact helper creates valid SSE events."""
    result = html_artifact(
        content="<div>Test</div>",
        name="test_artifact",
        description="A test artifact",
    )

    assert result["event"] == "copilotMessageArtifact"
    # data is now a JSON string, parse it
    data = json.loads(result["data"])
    assert data["type"] == "html"
    assert data["name"] == "test_artifact"
    assert data["description"] == "A test artifact"
    assert data["content"] == "<div>Test</div>"
    assert "uuid" in data
