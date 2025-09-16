import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vanilla_agent_dashboard_widgets.main import app
from openbb_ai.testing import CopilotResponse


test_client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_sse_starlette_appstatus_event():
    """
    Fixture that resets the appstatus event in the sse_starlette app.
    Should be used on any test that uses sse_starlette to stream events.
    """
    # See https://github.com/sysid/sse-starlette/issues/59
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


def test_agents_json_has_dashboard_search_feature_enabled():
    response = test_client.get("/agents.json")
    assert response.status_code == 200
    data = response.json()
    agent = data.get("vanilla_agent_dashboard_widgets")
    assert agent is not None
    assert agent["features"]["widget-dashboard-search"] is True


def test_query_lists_dashboard_widgets_from_secondary():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "retrieve_widget_from_dashboard.json"
    )
    payload = json.load(open(test_payload_path))

    # Simulate no explicit primary selection
    payload["widgets"]["primary"] = []
    # Ask to list widgets
    payload["messages"][0]["content"] = "What widgets are available in the dashboard?"

    response = test_client.post("/v1/query", json=payload)
    assert response.status_code == 200

    # We expect a message listing dashboard widgets (secondary only)
    CopilotResponse(response.text).has_any("copilotMessage", "Company News")
    # Should show dashboard context header
    CopilotResponse(response.text).has_any("copilotMessage", "Dashboard Context")


def test_query_lists_primary_widgets_when_selected():
    # Use same payload but keep primary set
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "retrieve_widget_from_dashboard.json"
    )
    payload = json.load(open(test_payload_path))

    response = test_client.post("/v1/query", json=payload)
    assert response.status_code == 200

    # We expect a message listing widgets including primary context
    CopilotResponse(response.text).has_any("copilotMessage", "Explicit Context")
    # Should also show dashboard context if available
    CopilotResponse(response.text).has_any("copilotMessage", "Dashboard Context")


def test_query_shows_widget_metadata_and_parameters():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "retrieve_widget_from_dashboard.json"
    )
    payload = json.load(open(test_payload_path))
    payload["messages"][0]["content"] = "Show me widget details"

    response = test_client.post("/v1/query", json=payload)
    assert response.status_code == 200

    # Should show widget metadata fields
    CopilotResponse(response.text).has_any("copilotMessage", "Description")
    CopilotResponse(response.text).has_any("copilotMessage", "UUID")
    # Should show parameters table
    CopilotResponse(response.text).has_any("copilotMessage", "Parameters")


def test_query_does_not_fetch_widget_data():
    """Test that the agent only lists widgets and does not fetch data"""
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "retrieve_widget_from_dashboard.json"
    )
    payload = json.load(open(test_payload_path))
    payload["messages"][0]["content"] = "Hello"

    response = test_client.post("/v1/query", json=payload)
    assert response.status_code == 200
    
    # Should NOT have any function calls for get_widget_data
    response_text = response.text
    assert "get_widget_data" not in response_text
    assert "copilotFunctionCall" not in response_text
    
    # Should only have messages listing widgets
    CopilotResponse(response.text).has_any("copilotMessage")