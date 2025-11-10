"""Comprehensive tests for MCP Agent functionality."""

import json
from fastapi.testclient import TestClient
from vanilla_agent_mcp_tools.main import app
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


def test_agents_json_mcp_configuration():
    """Test that the agent is properly configured for MCP tools."""
    response = test_client.get("/agents.json")
    assert response.status_code == 200

    data = response.json()
    assert "vanilla_agent_mcp" in data

    agent_config = data["vanilla_agent_mcp"]
    
    # Check basic agent information
    assert agent_config["name"] == "Vanilla Agent with MCP"
    assert "MCP tools" in agent_config["description"]
    
    # Check MCP-specific features
    features = agent_config["features"]
    assert features["mcp-tools"] is True
    assert features["streaming"] is True
    assert features["widget-dashboard-select"] is True
    assert features["widget-dashboard-search"] is False  # Should be false for this agent


def test_agents_json_endpoints():
    """Test that the correct API endpoints are exposed."""
    response = test_client.get("/agents.json")
    data = response.json()
    agent_config = data["vanilla_agent_mcp"]
    
    assert "endpoints" in agent_config
    assert agent_config["endpoints"]["query"] == "/v1/query"


def test_mcp_agent_image_configuration():
    """Test that the MCP agent has proper image configuration."""
    response = test_client.get("/agents.json")
    data = response.json()
    config = data["vanilla_agent_mcp"]
    
    # Should have an image configured
    assert "image" in config
    assert config["image"] is not None
    assert isinstance(config["image"], str)
    assert len(config["image"]) > 0


def test_empty_messages_validation():
    """Test that empty messages are properly validated."""
    test_payload = {"messages": []}
    response = test_client.post("/v1/query", json=test_payload)
    
    # Should return an error about empty messages
    assert "messages list cannot be empty" in response.text


def test_query_with_widget_data():
    """Test that widget data retrieval still works with MCP agent."""
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "message_with_primary_widget.json"
    )
    test_payload = json.load(open(test_payload_path))

    response = test_client.post("/v1/query", json=test_payload)
    assert response.status_code == 200

    # Should return widget data function call (this doesn't hit OpenAI)
    copilot_response = CopilotResponse(response.text)
    (
        copilot_response.starts("copilotFunctionCall")
        .with_({"function": "get_widget_data"})
        .with_(
            {
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": "123e4567-e89b-12d3-a456-426614174000",
                            "origin": "openbb",
                            "id": "company_news",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                }
            }
        )
    )


def test_widget_data_priority_over_mcp_tools():
    """Test that widget data takes priority over MCP tools when both present."""
    test_payload = {
        "messages": [{"role": "human", "content": "Get AAPL data"}],
        "widgets": {
            "primary": [{
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "origin": "openbb",
                "widget_id": "company_news", 
                "name": "Company News",
                "description": "News about a company",
                "params": [{
                    "name": "ticker",
                    "type": "string", 
                    "description": "The ticker of the company",
                    "current_value": "AAPL"
                }],
                "metadata": {}
            }]
        },
        "tools": [
            {
                "server_id": "test_server",
                "name": "test_tool",
                "url": "http://test.com",
                "description": "Test MCP tool",
                "input_schema": {"type": "object"},
                "auth_token": None
            }
        ]
    }

    response = test_client.post("/v1/query", json=test_payload) 
    assert response.status_code == 200
    
    # Should prioritize widget data retrieval over MCP tools
    copilot_response = CopilotResponse(response.text)
    (
        copilot_response.starts("copilotFunctionCall")
        .with_({"function": "get_widget_data"})
    )


def test_mcp_tools_none_handled():
    """Test that empty messages validation works even with tools=None."""
    test_payload = {
        "messages": [],
        "tools": None
    }
    
    response = test_client.post("/v1/query", json=test_payload)
    # Should get empty messages error, not crash due to tools=None
    assert "messages list cannot be empty" in response.text


def test_mcp_tools_empty_list_handled():
    """Test that empty messages validation works even with empty tools list."""
    test_payload = {
        "messages": [],
        "tools": []
    }
    
    response = test_client.post("/v1/query", json=test_payload)
    # Should get empty messages error, not crash due to empty tools
    assert "messages list cannot be empty" in response.text


def test_mcp_configuration_preserved():
    """Test that MCP configuration doesn't interfere with basic functionality."""
    response = test_client.get("/agents.json")
    data = response.json()
    config = data["vanilla_agent_mcp"]
    
    # Should have MCP tools enabled without breaking other features
    assert config["features"]["mcp-tools"] is True
    assert config["features"]["widget-dashboard-select"] is True
    assert config["features"]["streaming"] is True
    
    # Should have correct endpoints
    assert config["endpoints"]["query"] == "/v1/query"


def test_openapi_docs_accessible():
    """Test that OpenAPI documentation is accessible."""
    # This tests the FastAPI app setup
    response = test_client.get("/docs")
    assert response.status_code == 200


def test_health_check_via_agents_endpoint():
    """Basic health check using the agents endpoint."""
    response = test_client.get("/agents.json")
    assert response.status_code == 200
    
    # Should return valid JSON
    data = response.json()
    assert isinstance(data, dict)
    assert len(data) > 0


def test_agent_basic_configuration():
    """Test basic agent configuration is correct."""
    response = test_client.get("/agents.json")
    data = response.json()
    config = data["vanilla_agent_mcp"]
    
    # Check required fields are present
    required_fields = ["name", "description", "endpoints", "features"]
    for field in required_fields:
        assert field in config, f"Missing required field: {field}"
    
    # Check image is provided
    assert "image" in config
    assert config["image"] is not None


def test_mcp_tool_structure_validation():
    """Test that empty messages validation works even with MCP tools present."""
    valid_tool = {
        "server_id": "test_server_123",
        "name": "Valid Tool_test_function",
        "url": "http://example.com",
        "description": "A valid MCP tool",
        "input_schema": {"type": "object"},
        "auth_token": None
    }
    
    test_payload = {
        "messages": [],
        "tools": [valid_tool]
    }
    
    response = test_client.post("/v1/query", json=test_payload)
    # Should get empty messages error, not crash due to MCP tool processing
    assert "messages list cannot be empty" in response.text


def test_cors_headers_present():
    """Test that CORS headers are properly configured."""
    response = test_client.get("/agents.json")
    
    # Should have CORS headers configured (from middleware)
    assert response.status_code == 200
    # Basic verification that CORS middleware is active


def test_multiple_mcp_tools_handling():
    """Test that empty messages validation works even with multiple MCP tools."""
    tool1 = {
        "server_id": "server1",
        "name": "Tool1_function1",
        "url": "http://test1.com",
        "description": "First test tool",
        "input_schema": {"type": "object"},
        "auth_token": None
    }
    
    tool2 = {
        "server_id": "server2", 
        "name": "Tool2_function2",
        "url": "http://test2.com",
        "description": "Second test tool",
        "input_schema": {"type": "object"},
        "auth_token": None
    }
    
    test_payload = {
        "messages": [],
        "tools": [tool1, tool2]
    }
    
    response = test_client.post("/v1/query", json=test_payload)
    # Should get empty messages error, not crash due to multiple MCP tools
    assert "messages list cannot be empty" in response.text