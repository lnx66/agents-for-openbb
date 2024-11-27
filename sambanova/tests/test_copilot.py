import json
from fastapi.testclient import TestClient
from sambanova.main import app
import pytest
from pathlib import Path
from common.testing import CopilotResponse

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
    assert response.status_code == 200

    CopilotResponse(response.text).has_any(
        event_type="copilotMessage", content_contains="2"
    )


def test_query_conversation():
    test_payload_path = (
        Path(__file__).parent.parent.parent / "test_payloads" / "multiple_messages.json"
    )
    test_payload = json.load(open(test_payload_path))

    response = test_client.post("/v1/query", json=test_payload)
    assert response.status_code == 200
    CopilotResponse(response.text).has_any(
        event_type="copilotMessage", content_contains="4"
    )


def test_query_with_context():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "test_payloads"
        / "message_with_context.json"
    )
    test_payload = json.load(open(test_payload_path))
    response = test_client.post("/v1/query", json=test_payload)
    CopilotResponse(response.text).has_any(
        event_type="copilotMessage", content_contains="pizza"
    )


def test_query_no_messages():
    test_payload = {
        "messages": [],
    }
    response = test_client.post("/v1/query", json=test_payload)
    "messages list cannot be empty" in response.text


def test_query_handle_function_call():
    test_payload_path = (
        Path(__file__).parent.parent.parent
        / "test_payloads"
        / "retrieve_widget_from_dashboard.json"
    )
    test_payload = json.load(open(test_payload_path))
    response = test_client.post("/v1/query", json=test_payload)

    (
        CopilotResponse(response.text)
        .starts_with(
            event_type="copilotStatusUpdate", content_contains="Calling function"
        )
        .then(event_type="copilotFunctionCall", content_contains="get_widget_data")
        .and_(content_contains="91ae1153-7b4d-451c-adc5-49856e90a0e6")
    )


@pytest.mark.skip(
    reason="This test is still flaky with the 70-B model (but generally works in practice)."  # noqa: E501
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
        .has_any(event_type="copilotMessage", content_contains="0.0469")  # month_1
        .has_any(event_type="copilotMessage", content_contains="0.046")  # month_3
        .has_any(event_type="copilotMessage", content_contains="0.044")  # month_6
        .has_any(event_type="copilotMessage", content_contains="0.0431")  # year_1
        .has_any(event_type="copilotMessage", content_contains="0.0427")  # year_2
        .has_any(event_type="copilotMessage", content_contains="0.0425")  # year_3
        .has_any(event_type="copilotMessage", content_contains="0.043")  # year_5
        .has_any(event_type="copilotMessage", content_contains="0.0438")  # year_7
        .has_any(event_type="copilotMessage", content_contains="0.0444")  # year_10
        .has_any(event_type="copilotMessage", content_contains="0.0473")  # year_20
        .has_any(event_type="copilotMessage", content_contains="0.0463")  # year_30
    )
