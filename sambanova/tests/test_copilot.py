import json
from fastapi.testclient import TestClient
from advanced_example_copilot.main import app
import pytest
from pathlib import Path
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


def test_query_handle_function_call():
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "What's the weather in San Francisco and London?",
            }
        ],
    }
    response = test_client.post("/v1/query", json=payload)


def test_query_handle_function_call_result():
    payload = {
        "widgets": [
            {
                "uuid": "91ae1153-7b4d-451c-adc5-49856e90a0e6",
                "name": "Yield Curve",
                "description": "Get yield curve data by country and date.",
                "metadata": {"source": "EconDB", "lastUpdated": 1732178181043},
            }
        ],
        "custom_direct_retrieval_endpoints": [],
        "messages": [
            {
                "role": "human",
                "content": "what can you tell me about the yield curve data?",
            },
            {
                "role": "ai",
                "content": '{"function":"get_widget_data","input_arguments":{"widget_uuid":"91ae1153-7b4d-451c-adc5-49856e90a0e6"},"copilot_function_call_arguments":{"widget_uuid":"91ae1153-7b4d-451c-adc5-49856e90a0e6"}}',
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "widget_uuid": "91ae1153-7b4d-451c-adc5-49856e90a0e6"
                },
                "data": {
                    "content": '[{"date":"2024-11-13","maturity":"month_1","rate":0.046900000000000004},{"date":"2024-11-13","maturity":"month_3","rate":0.046},{"date":"2024-11-13","maturity":"month_6","rate":0.044000000000000004},{"date":"2024-11-13","maturity":"year_1","rate":0.0431},{"date":"2024-11-13","maturity":"year_2","rate":0.042699999999999995},{"date":"2024-11-13","maturity":"year_3","rate":0.0425},{"date":"2024-11-13","maturity":"year_5","rate":0.043},{"date":"2024-11-13","maturity":"year_7","rate":0.0438},{"date":"2024-11-13","maturity":"year_10","rate":0.0444},{"date":"2024-11-13","maturity":"year_20","rate":0.0473},{"date":"2024-11-13","maturity":"year_30","rate":0.0463}]'
                },
            },
        ],
    }
    response = test_client.post("/v1/query", json=payload)
    event_name, captured_stream = capture_stream_response(response.text)
    assert response.status_code == 200
    assert event_name == "copilotFunctionCall"
    assert "get_widget_data" in captured_stream
