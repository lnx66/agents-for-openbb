import json
from fastapi.testclient import TestClient
from example_copilot.main import app
import pytest
from pathlib import Path
from common.testing import CopilotResponse, capture_stream_response

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


def test_query_handle_function_call():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "test_payloads"
        / "retrieve_widget_from_dashboard.json"
    )
    test_payload = json.load(open(test_payload_path))
    response = test_client.post("/v1/query", json=test_payload)
    assert response.status_code == 200
    (
        CopilotResponse(response.text)
        .starts_with(
            event_type="copilotFunctionCall", content_contains="get_widget_data"
        )
        .and_(content_contains="input_arguments")
        .and_(content_contains="OpenBB API")
        .and_(content_contains="stock_price")
        .and_(content_contains="TSLA")
        .and_(content_contains="copilot_function_call_arguments")
    )


def test_query_handle_function_call_result():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "test_payloads"
        / "retrieve_widget_from_dashboard_with_result.json"
    )
    test_payload = json.load(open(test_payload_path))
    response = test_client.post("/v1/query", json=test_payload)
    assert response.status_code == 200
    (
        CopilotResponse(response.text)
        .has_any(event_type="copilotMessage", content_contains="0.0444")  # month_1
        .has_any(event_type="copilotMessage", content_contains="0.0435")  # month_3
        .has_any(event_type="copilotMessage", content_contains="0.0424")  # month_6
        .has_any(event_type="copilotMessage", content_contains="0.0416")  # year_1
        .has_any(event_type="copilotMessage", content_contains="0.0427")  # year_2
        .has_any(event_type="copilotMessage", content_contains="0.0431")  # year_3
        .has_any(event_type="copilotMessage", content_contains="0.0446")  # year_5
        .has_any(event_type="copilotMessage", content_contains="0.0457")  # year_7
        .has_any(event_type="copilotMessage", content_contains="0.0468")  # year_10
        .has_any(event_type="copilotMessage", content_contains="0.0498")  # year_20
        .has_any(event_type="copilotMessage", content_contains="0.0492")  # year_30
    )


def test_query_no_messages():
    test_payload = {
        "messages": [],
    }
    response = test_client.post("/v1/query", json=test_payload)
    "messages list cannot be empty" in response.text
