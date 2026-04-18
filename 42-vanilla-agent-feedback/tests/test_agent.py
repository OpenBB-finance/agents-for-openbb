import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vanilla_agent_feedback.main import FEEDBACK_FILE, app

test_client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_sse_starlette_appstatus_event():
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


@pytest.fixture(autouse=True)
def cleanup_feedback_file():
    yield
    if FEEDBACK_FILE.exists():
        FEEDBACK_FILE.unlink()


def test_agents_json():
    response = test_client.get("/agents.json")
    assert response.status_code == 200
    data = response.json()
    assert "vanilla_agent_feedback" in data
    agent = data["vanilla_agent_feedback"]
    assert agent["features"]["feedback"] is True
    assert agent["endpoints"]["feedback"] == "/v1/feedback"


def test_feedback_thumbs_up(tmp_path, monkeypatch):
    feedback_file = tmp_path / "feedback.json"
    monkeypatch.setattr("vanilla_agent_feedback.main.FEEDBACK_FILE", feedback_file)

    payload = {
        "vote": "thumbs_up",
        "tags": [],
        "user_comment": "",
        "ai_response": "Test response",
        "user_prompt": "Test prompt",
        "trace_id": "test-trace-123",
    }
    response = test_client.post("/v1/feedback", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    entries = json.loads(feedback_file.read_text())
    assert len(entries) == 1
    assert entries[0]["vote"] == "thumbs_up"


def test_feedback_thumbs_down(tmp_path, monkeypatch):
    feedback_file = tmp_path / "feedback.json"
    monkeypatch.setattr("vanilla_agent_feedback.main.FEEDBACK_FILE", feedback_file)

    payload = {
        "vote": "thumbs_down",
        "tags": ["Not factually correct / Hallucinations / Inaccurate"],
        "user_comment": "The data was wrong",
        "ai_response": "Test response",
        "user_prompt": "Test prompt",
        "trace_id": "test-trace-456",
    }
    response = test_client.post("/v1/feedback", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    entries = json.loads(feedback_file.read_text())
    assert len(entries) == 1
    assert entries[0]["vote"] == "thumbs_down"
    assert entries[0]["tags"] == ["Not factually correct / Hallucinations / Inaccurate"]


def test_query():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "testing"
        / "test_payloads"
        / "single_message.json"
    )
    test_payload = json.load(open(test_payload_path))
    response = test_client.post("/v1/query", json=test_payload)
    assert response.status_code == 200
