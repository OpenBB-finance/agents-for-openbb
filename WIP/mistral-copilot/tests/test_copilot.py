from ast import literal_eval
import json
from pathlib import Path
from fastapi.testclient import TestClient
from mistral_copilot.main import app
import pytest

from common.testing import capture_stream_response

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


def test_query():
    test_payload_path = (
        Path(__file__).parent.parent.parent / "test_payloads" / "single_message.json"
    )
    test_payload = json.load(open(test_payload_path))
    response = test_client.post("/v1/query", json=test_payload)
    event_name, captured_stream = capture_stream_response(response.text)
    assert response.status_code == 200
    assert event_name == "copilotMessageChunk"
    assert "2" in captured_stream


def test_query_conversation():
    test_payload_path = (
        Path(__file__).parent.parent.parent / "test_payloads" / "multiple_messages.json"
    )
    test_payload = json.load(open(test_payload_path))
    response = test_client.post("/v1/query", json=test_payload)
    event_name, captured_stream = capture_stream_response(response.text)
    assert response.status_code == 200
    assert event_name == "copilotMessageChunk"
    assert "4" in captured_stream


def test_query_with_context():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "test_payloads"
        / "message_with_context.json"
    )
    test_payload = json.load(open(test_payload_path))
    response = test_client.post("/v1/query", json=test_payload)
    event_name, captured_stream = capture_stream_response(response.text)
    assert response.status_code == 200
    assert event_name == "copilotMessageChunk"
    assert "pizza" in captured_stream.lower()


def test_query_no_messages():
    test_payload = {
        "messages": [],
    }
    response = test_client.post("/v1/query", json=test_payload)
    "messages list cannot be empty" in response.text


def test_query_function_call():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "test_payloads"
        / "retrieve_widget_from_dashboard.json"
    )
    test_payload = json.load(open(test_payload_path))
    response = test_client.post("/v1/query", json=test_payload)
    event_name, captured_stream = capture_stream_response(response.text)

    function_call = literal_eval(captured_stream)
    assert response.status_code == 200
    assert event_name == "copilotFunctionCall"
    assert function_call["function"] == "get_widget_data"
    assert function_call["input_arguments"] == {
        "widget_uuids": ["91ae1153-7b4d-451c-adc5-49856e90a0e6"]
    }


def test_query_function_call_gives_final_answer():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "test_payloads"
        / "retrieve_widget_from_dashboard_with_result.json"
    )
    test_payload = json.load(open(test_payload_path))
    response = test_client.post("/v1/query", json=test_payload)
    event_name, captured_stream = capture_stream_response(response.text)

    assert response.status_code == 200
    assert event_name == "copilotMessageChunk"
    assert "4.69%" in captured_stream  # month_1
    assert "4.60%" in captured_stream  # month_3
    assert "4.40%" in captured_stream  # month_6
    assert "4.31%" in captured_stream  # year_1
    assert "4.27%" in captured_stream  # year_2
    assert "4.25%" in captured_stream  # year_3
    assert "4.30%" in captured_stream  # year_5
    assert "4.38%" in captured_stream  # year_7
    assert "4.44%" in captured_stream  # year_10
    assert "4.73%" in captured_stream  # year_20
    assert "4.63%" in captured_stream  # year_30
