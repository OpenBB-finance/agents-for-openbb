import json
from unittest import mock
from fastapi.testclient import TestClient
from vanilla_agent_mcp_tools.main import app
import pytest
from pathlib import Path
from openbb_ai.testing import CopilotResponse
from openbb_ai.models import QueryRequest, AgentTool

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
    assert "vanilla_agent_mcp" in data

    agent_config = data["vanilla_agent_mcp"]
    assert agent_config["name"] == "Vanilla Agent with MCP"
    assert agent_config["features"]["mcp-tools"] is True
    assert agent_config["features"]["streaming"] is True


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


def test_query_with_mcp_tools():
    """Test query with MCP tools available."""
    # Create mock MCP tools
    mock_tool = AgentTool(
        server_id="test_server_123",
        name="Test Tool_test_function",
        url="http://test.com",
        endpoint="test_endpoint",
        description="A test MCP tool",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    )

    test_payload = {
        "messages": [{"role": "human", "content": "What MCP tools do you have?"}],
        "tools": [mock_tool]
    }

    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        # Mock the OpenAI client
        mock_client = mock.MagicMock()
        mock_openai.return_value = mock_client

        # Verify that the system message contains MCP tool information
        async def mock_create(**kwargs):
            messages = kwargs["messages"]
            system_message = messages[0]["content"]

            # Check that the system message contains MCP tool info
            assert "Test Tool_test_function" in system_message
            assert "A test MCP tool" in system_message
            assert "test_server_123" in system_message

            # Return a mock stream
            class MockEvent:
                class Choice:
                    class Delta:
                        content = "I have MCP tools available."

                    delta = Delta()

                choices = [Choice()]

            yield MockEvent()

        mock_client.chat.completions.create = mock_create

        response = test_client.post("/v1/query", json=test_payload)
        assert response.status_code == 200


def test_query_mcp_function_call():
    """Test that MCP tool function calls are properly handled."""
    mock_tool = AgentTool(
        server_id="test_server_123",
        name="Test Tool_test_function",
        url="http://test.com",
        endpoint="test_endpoint", 
        description="A test MCP tool",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    )

    test_payload = {
        "messages": [{"role": "human", "content": "Use the test tool to search for widgets"}],
        "tools": [mock_tool]
    }

    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        # Mock the OpenAI client
        mock_client = mock.MagicMock()
        mock_openai.return_value = mock_client

        # Mock function call response
        class MockFunctionCall:
            name = "execute_agent_tool"
            arguments = json.dumps({
                "server_id": "test_server_123",
                "tool_name": "Test Tool_test_function",
                "parameters": {"query": "widgets"}
            })

        class MockMessage:
            function_call = MockFunctionCall()
            content = None

        class MockChoice:
            message = MockMessage()

        class MockResponse:
            choices = [MockChoice()]

        async def mock_create(**kwargs):
            if "functions" in kwargs:
                # Return function call
                return MockResponse()
            else:
                # Return regular streaming response
                class MockEvent:
                    class Choice:
                        class Delta:
                            content = "Based on the search..."
                        delta = Delta()
                    choices = [Choice()]
                yield MockEvent()

        mock_client.chat.completions.create = mock_create

        response = test_client.post("/v1/query", json=test_payload)
        assert response.status_code == 200
        
        # Should contain a function call response
        copilot_response = CopilotResponse(response.text)
        assert copilot_response.has_any("functionCall")


def test_query_mcp_tool_results():
    """Test processing of MCP tool results."""
    test_payload = {
        "messages": [
            {"role": "human", "content": "Search for widgets"},
            {"role": "ai", "content": "I'll search for widgets using the MCP tool."},
            {
                "role": "tool",
                "function": "execute_agent_tool", 
                "data": [
                    {
                        "items": [
                            {"content": "Widget documentation found: Widgets are UI components..."}
                        ]
                    }
                ]
            }
        ]
    }

    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        # Mock the OpenAI client
        mock_client = mock.MagicMock()
        mock_openai.return_value = mock_client

        # Verify that tool results are processed correctly
        async def mock_create(**kwargs):
            messages = kwargs["messages"]
            user_message = messages[-1]["content"]

            # Check that MCP OUTPUT and AI OUTPUT structure is included
            assert "## MCP OUTPUT" in user_message
            assert "## AI OUTPUT" in user_message
            assert "Widget documentation found" in user_message

            # Return a mock stream
            class MockEvent:
                class Choice:
                    class Delta:
                        content = "## MCP OUTPUT\nWidget documentation found: Widgets are UI components...\n\n## AI OUTPUT\nBased on the documentation..."

                    delta = Delta()

                choices = [Choice()]

            yield MockEvent()

        mock_client.chat.completions.create = mock_create

        response = test_client.post("/v1/query", json=test_payload)
        assert response.status_code == 200


def test_query_no_mcp_tools():
    """Test query without MCP tools (fallback behavior)."""
    test_payload = {
        "messages": [{"role": "human", "content": "Hello"}],
        "tools": None
    }

    response = test_client.post("/v1/query", json=test_payload)
    assert response.status_code == 200
    copilot_response = CopilotResponse(response.text)
    assert copilot_response.has_any("copilotMessage", "2")


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


def test_mcp_tool_parameter_validation():
    """Test that MCP tool parameters are properly validated."""
    mock_tool = AgentTool(
        server_id="test_server_123",
        name="Test Tool_search_docs",
        url="http://test.com",
        endpoint="test_endpoint",
        description="Search documentation",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Result limit", "default": 10}
            },
            "required": ["query"]
        }
    )

    test_payload = {
        "messages": [{"role": "human", "content": "Search docs for 'trading'"}],
        "tools": [mock_tool]
    }

    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        mock_client = mock.MagicMock()
        mock_openai.return_value = mock_client

        async def mock_create(**kwargs):
            # Verify function definition includes proper schema
            if "functions" in kwargs:
                functions = kwargs["functions"]
                execute_tool_func = functions[0]
                assert execute_tool_func["name"] == "execute_agent_tool"
                assert "server_id" in execute_tool_func["parameters"]["properties"]
                assert "tool_name" in execute_tool_func["parameters"]["properties"]
                assert "parameters" in execute_tool_func["parameters"]["properties"]
                
                # Mock function call response
                class MockFunctionCall:
                    name = "execute_agent_tool"
                    arguments = json.dumps({
                        "server_id": "test_server_123",
                        "tool_name": "Test Tool_search_docs", 
                        "parameters": {"query": "trading", "limit": 5}
                    })

                class MockMessage:
                    function_call = MockFunctionCall()
                    content = None

                class MockChoice:
                    message = MockMessage()

                class MockResponse:
                    choices = [MockChoice()]

                return MockResponse()

        mock_client.chat.completions.create = mock_create

        response = test_client.post("/v1/query", json=test_payload)
        assert response.status_code == 200