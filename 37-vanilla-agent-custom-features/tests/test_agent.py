import json
from unittest import mock
from fastapi.testclient import TestClient
from vanilla_agent_custom_features.main import app
import pytest
from pathlib import Path
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


def test_agents_json():
    """Test that the agents.json endpoint returns the correct configuration."""
    response = test_client.get("/agents.json")
    assert response.status_code == 200
    
    data = response.json()
    assert "vanilla_agent_custom_features" in data
    
    agent_config = data["vanilla_agent_custom_features"]
    assert agent_config["name"] == "Vanilla Agent Custom Features"
    assert "deep-research" in agent_config["features"]
    assert "web-search" in agent_config["features"]
    assert agent_config["features"]["deep-research"]["default"] == False
    assert agent_config["features"]["web-search"]["default"] == True


def test_query_simple_message():
    """Test basic query with a simple message."""
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "single_message.json"
    )
    test_payload = json.load(open(test_payload_path))

    response = test_client.post("/v1/query", json=test_payload)
    assert response.status_code == 200
    copilot_response = CopilotResponse(response.text)
    assert copilot_response.has_any("copilotMessage", "2")


def test_query_with_workspace_options():
    """Test query with workspace options to verify feature detection."""
    test_payload = {
        "messages": [
            {
                "role": "human",
                "content": "Hello, what features are enabled?"
            }
        ],
        "workspace_options": {
            "deep-research": True,
            "web-search": False
        }
    }
    
    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        # Mock the OpenAI client
        mock_client = mock.MagicMock()
        mock_openai.return_value = mock_client
        
        # Verify that the system message contains the correct feature status
        async def mock_create(**kwargs):
            messages = kwargs["messages"]
            system_message = messages[0]["content"]
            
            # Check that the system message contains the feature status
            assert "Deep Research: ✅ Enabled" in system_message
            assert "Web Search: ❌ Disabled" in system_message
            
            # Return a mock stream
            class MockEvent:
                class Choice:
                    class Delta:
                        content = "Hello! Features are configured."
                    delta = Delta()
                choices = [Choice()]
            
            yield MockEvent()
        
        mock_client.chat.completions.create = mock_create
        
        response = test_client.post("/v1/query", json=test_payload)
        assert response.status_code == 200


def test_query_default_workspace_options():
    """Test query without workspace options uses defaults."""
    test_payload = {
        "messages": [
            {
                "role": "human", 
                "content": "Hi there!"
            }
        ]
    }
    
    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        # Mock the OpenAI client
        mock_client = mock.MagicMock()
        mock_openai.return_value = mock_client
        
        # Verify that the system message contains the default feature status
        async def mock_create(**kwargs):
            messages = kwargs["messages"]
            system_message = messages[0]["content"]
            
            # Check that defaults are used (deep-research: False, web-search: True)
            assert "Deep Research: ❌ Disabled" in system_message
            assert "Web Search: ✅ Enabled" in system_message
            
            # Return a mock stream
            class MockEvent:
                class Choice:
                    class Delta:
                        content = "Hello!"
                    delta = Delta()
                choices = [Choice()]
            
            yield MockEvent()
        
        mock_client.chat.completions.create = mock_create
        
        response = test_client.post("/v1/query", json=test_payload)
        assert response.status_code == 200


def test_query_conversation():
    """Test query with multiple messages in conversation."""
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "multiple_messages.json"
    )
    test_payload = json.load(open(test_payload_path))

    response = test_client.post("/v1/query", json=test_payload)
    assert response.status_code == 200
    copilot_response = CopilotResponse(response.text)
    assert copilot_response.has_any("copilotMessage", "4")


def test_query_no_messages():
    """Test query with empty messages list."""
    test_payload = {
        "messages": [],
    }
    response = test_client.post("/v1/query", json=test_payload)
    assert "messages list cannot be empty" in response.text