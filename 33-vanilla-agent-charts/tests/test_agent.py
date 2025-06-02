import json
from fastapi.testclient import TestClient
from vanilla_agent_charts.main import app
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


def test_query_contains_chart():
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
                "type": "chart",
                "name": "Line Chart",
                "description": "This is a line chart of the data",
                "content": [
                    {"x": 0, "y": 1},
                    {"x": 1, "y": 2},
                    {"x": 2, "y": 3},
                    {"x": 3, "y": 5},
                ],
                "chart_params": {"chartType": "line", "xKey": "x", "yKey": ["y"]},
            },
        )
        .has_any(
            "copilotMessageArtifact",
            content_contains={
                "type": "chart",
                "name": "Bar Chart",
                "description": "This is a bar chart of the data",
                "content": [
                    {"x": "A", "y": 1},
                    {"x": "B", "y": 2},
                    {"x": "C", "y": 3},
                    {"x": "D", "y": 5},
                ],
                "chart_params": {"chartType": "bar", "xKey": "x", "yKey": ["y"]},
            },
        )
        .has_any(
            "copilotMessageArtifact",
            content_contains={
                "type": "chart",
                "name": "Scatter Chart",
                "description": "This is a scatter chart of the data",
                "content": [
                    {"x": 0, "y": 1},
                    {"x": 1, "y": 2},
                    {"x": 2, "y": 3},
                    {"x": 3, "y": 5},
                ],
                "chart_params": {"chartType": "scatter", "xKey": "x", "yKey": ["y"]},
            },
        )
        .has_any(
            "copilotMessageArtifact",
            content_contains={
                "type": "chart",
                "name": "Pie Chart",
                "description": "This is a pie chart of the data",
                "content": [
                    {"value": 0, "label": "A"},
                    {"value": 1, "label": "B"},
                    {"value": 2, "label": "C"},
                    {"value": 3, "label": "D"},
                ],
                "chart_params": {
                    "chartType": "pie",
                    "angleKey": "value",
                    "calloutLabelKey": "label",
                },
            },
        )
        .has_any(
            "copilotMessageArtifact",
            content_contains={
                "type": "chart",
                "name": "Donut Chart",
                "description": "This is a donut chart of the data",
                "content": [
                    {"value": 0, "label": "A"},
                    {"value": 1, "label": "B"},
                    {"value": 2, "label": "C"},
                    {"value": 3, "label": "D"},
                ],
                "chart_params": {
                    "chartType": "donut",
                    "angleKey": "value",
                    "calloutLabelKey": "label",
                },
            },
        )
    )
