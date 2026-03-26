import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from openbb_ai.testing import CopilotResponse

from vanilla_agent_dynamic_skill.main import app

test_client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_sse_starlette_appstatus_event():
    """
    Fixture that resets the appstatus event in the sse_starlette app.
    Should be used on any test that uses sse_starlette to stream events.
    """
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


class MockStream:
    def __init__(self, *chunks: str):
        self._events = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=chunk))]
            )
            for chunk in chunks
        ]
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


def _mock_streaming_client(*chunks: str):
    async def mock_create(**kwargs):
        assert kwargs["stream"] is True
        return MockStream(*chunks)

    mock_client = mock.MagicMock()
    mock_client.chat.completions.create = mock_create
    return mock_client


def test_agents_json_configuration():
    response = test_client.get("/agents.json")
    assert response.status_code == 200

    data = response.json()
    assert "vanilla_agent_dynamic_skill" in data

    config = data["vanilla_agent_dynamic_skill"]
    assert config["name"] == "Vanilla Agent Dynamic Skill"
    assert "loads one skill" in config["description"]
    assert config["features"]["streaming"] is True
    assert config["features"]["widget-dashboard-select"] is False
    assert config["features"]["widget-dashboard-search"] is False


def test_query_streams_without_skills():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "single_message.json"
    )
    test_payload = json.load(open(test_payload_path))

    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = _mock_streaming_client("Hello", " world")

        response = test_client.post("/v1/query", json=test_payload)

    assert response.status_code == 200
    copilot_response = CopilotResponse(response.text)
    assert copilot_response.text == "Hello world"


def test_query_with_skill_catalog_emits_function_call():
    response_message = SimpleNamespace(
        function_call=SimpleNamespace(
            name="get_skill_content",
            arguments=json.dumps(
                {
                    "slug": "financial-analysis",
                    "reason": "The user asked for that skill by name.",
                }
            ),
        ),
        content=None,
    )
    response_object = SimpleNamespace(
        choices=[SimpleNamespace(message=response_message)]
    )

    async def mock_create(**kwargs):
        assert kwargs["stream"] is False
        assert kwargs["functions"][0]["name"] == "get_skill_content"
        system_prompt = kwargs["messages"][0]["content"]
        assert "financial-analysis" in system_prompt
        assert "Only request one skill." in system_prompt
        return response_object

    test_payload = {
        "messages": [
            {
                "role": "human",
                "content": "Use the financial-analysis skill to review AAPL.",
            }
        ],
        "skills_catalog": [
            {
                "slug": "financial-analysis",
                "description": "Analyze company financials and earnings",
                "updatedAt": "2026-03-22T12:00:00Z",
            }
        ],
    }

    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create = mock_create
        mock_openai.return_value = mock_client

        response = test_client.post("/v1/query", json=test_payload)

    assert response.status_code == 200
    copilot_response = CopilotResponse(response.text)
    copilot_response.has_any(
        "copilotFunctionCall",
        {
            "function": "get_skill_content",
            "input_arguments": {
                "slug": "financial-analysis",
                "reason": "The user asked for that skill by name.",
            },
            "extra_state": {
                "copilot_function_call_arguments": {
                    "slug": "financial-analysis",
                    "reason": "The user asked for that skill by name.",
                }
            },
        },
    )


def test_selected_skill_is_injected_and_no_function_is_exposed():
    async def mock_create(**kwargs):
        assert kwargs["stream"] is True
        assert "functions" not in kwargs
        system_prompt = kwargs["messages"][0]["content"]
        assert "## Active Skill" in system_prompt
        assert "financial-analysis" in system_prompt
        assert "Focus on revenue growth" in system_prompt
        return MockStream("Loaded skill answer.")

    test_payload = {
        "messages": [
            {
                "role": "human",
                "content": "Analyze AAPL with the selected skill.",
            }
        ],
        "selected_skills": [
            {
                "slug": "financial-analysis",
                "description": "Analyze company financials and earnings",
                "contentMarkdown": (
                    "# Financial Analysis\n\nFocus on revenue growth, margins, and guidance."
                ),
                "source": "model_selected",
            }
        ],
        "skills_catalog": [
            {
                "slug": "financial-analysis",
                "description": "Analyze company financials and earnings",
                "updatedAt": "2026-03-22T12:00:00Z",
            }
        ],
    }

    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create = mock_create
        mock_openai.return_value = mock_client

        response = test_client.post("/v1/query", json=test_payload)

    assert response.status_code == 200
    copilot_response = CopilotResponse(response.text)
    assert copilot_response.text == "Loaded skill answer."


def test_tool_result_loads_skill_and_answers_without_second_function_call():
    async def mock_create(**kwargs):
        assert kwargs["stream"] is True
        assert "functions" not in kwargs
        system_prompt = kwargs["messages"][0]["content"]
        assert "## Active Skill" in system_prompt
        assert "financial-analysis" in system_prompt
        assert "Focus on margins" in system_prompt
        return MockStream("Tool result answer.")

    test_payload = {
        "messages": [
            {
                "role": "human",
                "content": "Analyze AAPL with the loaded skill.",
            },
            {
                "role": "tool",
                "function": "get_skill_content",
                "input_arguments": {"slug": "financial-analysis"},
                "data": [
                    {
                        "status": "success",
                        "data": {
                            "skill": {
                                "slug": "financial-analysis",
                                "description": "Analyze company financials and earnings",
                                "contentMarkdown": (
                                    "# Financial Analysis\n\nFocus on margins and revenue growth."
                                ),
                            }
                        },
                    }
                ],
            },
        ]
    }

    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create = mock_create
        mock_openai.return_value = mock_client

        response = test_client.post("/v1/query", json=test_payload)

    assert response.status_code == 200
    copilot_response = CopilotResponse(response.text)
    assert copilot_response.text == "Tool result answer."


def test_multiple_selected_skills_uses_the_first_one():
    async def mock_create(**kwargs):
        system_prompt = kwargs["messages"][0]["content"]
        assert "first-skill" in system_prompt
        assert "Use the first skill." in system_prompt
        assert "second-skill" not in system_prompt
        return MockStream("First skill used.")

    test_payload = {
        "messages": [
            {
                "role": "human",
                "content": "Use whichever skill is active.",
            }
        ],
        "selected_skills": [
            {
                "slug": "first-skill",
                "description": "The first skill",
                "contentMarkdown": "Use the first skill.",
                "source": "model_selected",
            },
            {
                "slug": "second-skill",
                "description": "The second skill",
                "contentMarkdown": "Use the second skill.",
                "source": "model_selected",
            },
        ],
    }

    with mock.patch("openai.AsyncOpenAI") as mock_openai:
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create = mock_create
        mock_openai.return_value = mock_client

        response = test_client.post("/v1/query", json=test_payload)

    assert response.status_code == 200
    copilot_response = CopilotResponse(response.text)
    assert copilot_response.text == "First skill used."


def test_empty_messages_validation():
    response = test_client.post("/v1/query", json={"messages": []})
    assert "messages list cannot be empty" in response.text
