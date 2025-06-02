import json
from fastapi.testclient import TestClient
from vanilla_agent_tables.main import app
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
    copilot_response = CopilotResponse(response.text)
    (copilot_response.has_any("copilotMessage", "2"))


def test_query_conversation():
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
    (copilot_response.has_any("copilotMessage", "4"))


def test_query_no_messages():
    test_payload = {
        "messages": [],
    }
    response = test_client.post("/v1/query", json=test_payload)
    "messages list cannot be empty" in response.text


def test_query_contains_table():
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
    (
        copilot_response.has_any(
            "copilotMessageArtifact",
            content_contains={
                "type": "table",
                "name": "My Table",
                "description": "This is a table of the data",
                "content": [
                    {"x": 0, "y": 1, "z": 2},
                    {"x": 1, "y": 2, "z": 3},
                    {"x": 2, "y": 3, "z": 4},
                    {"x": 3, "y": 5, "z": 6},
                ],
            },
        )
    )
