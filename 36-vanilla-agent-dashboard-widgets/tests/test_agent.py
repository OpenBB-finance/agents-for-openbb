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


def test_agents_json_has_dashboard_search_feature():
    response = test_client.get("/agents.json")
    assert response.status_code == 200
    data = response.json()
    agent = data.get("vanilla_agent_dashboard_widgets")
    assert agent is not None
    assert agent["features"]["widget-dashboard-search"] is True


def test_query_recognizes_dashboard_widgets_from_secondary():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "retrieve_widget_from_dashboard.json"
    )
    payload = json.load(open(test_payload_path))

    # Simulate no explicit primary selection so agent must recognize from secondary
    payload["widgets"]["primary"] = []

    response = test_client.post("/v1/query", json=payload)
    assert response.status_code == 200

    # We expect a function call to fetch data for the recognized widget
    CopilotResponse(response.text).starts("copilotFunctionCall").with_(
        {"function": "get_widget_data"}
    ).with_(
        {"input_arguments": {"data_sources": [{"id": "stock_price"}]}}
    )


def test_query_respects_primary_selection_and_calls_get_widget_data():
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

    # We expect a function call – let the UI handle the actual retrieval
    CopilotResponse(response.text).starts("copilotFunctionCall").with_(
        {"function": "get_widget_data"}
    )


def test_query_lists_dashboard_widgets():
    # Ask to list widgets and expect a direct message with names
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "retrieve_widget_from_dashboard.json"
    )
    payload = json.load(open(test_payload_path))
    payload["widgets"]["primary"] = []
    payload["messages"][0]["content"] = "What widgets are available in the dashboard?"

    response = test_client.post("/v1/query", json=payload)
    assert response.status_code == 200

    CopilotResponse(response.text).has_any("copilotMessage", "Stock Price")
