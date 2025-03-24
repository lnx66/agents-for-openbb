import json
from fastapi.testclient import TestClient
from simple_copilot_fc.main import app
import pytest
from pathlib import Path
from common.testing import capture_stream_response
from unittest.mock import AsyncMock, patch

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


def test_query_no_messages():
    test_payload = {
        "messages": [],
    }
    response = test_client.post("/v1/query", json=test_payload)
    "messages list cannot be empty" in response.text


def test_query_local_function_call():
    test_payload = {
        "messages": [
            {
                "role": "human",
                "content": "Fetch and describe a random colour palette.",
            }
        ]
    }
    with patch(
        "simple_copilot_fc.main.get_random_palettes", new_callable=AsyncMock
    ) as mock_get_random_palettes:
        mock_get_random_palettes.return_value = """
-- Palettes --
name: Cycling Trivialities
url: http://www.colourlovers.com/palette/2404704/Cycling_Trivialities
imageUrl (use this to display the palette image): http://www.colourlovers.com/paletteImg/C87FD4/AF8DC4/958CB8/928DCC/8A91D4/Cycling_Trivialities.png
colours: ['#C87FD4', '#AF8DC4', '#958CB8', '#928DCC', '#8A91D4']
-----------------------------------
"""  # noqa: E501
        response = test_client.post("/v1/query", json=test_payload)
        event_name, captured_stream = capture_stream_response(response.text)

        assert response.status_code == 200
        assert event_name == "copilotMessageChunk"
        mock_get_random_palettes.assert_called_once()
        assert "colourlovers.com" in captured_stream
