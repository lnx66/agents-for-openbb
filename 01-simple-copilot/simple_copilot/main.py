import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from magentic import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    AsyncStreamedStr,
    Chat,
)
from sse_starlette.sse import EventSourceResponse

from dotenv import load_dotenv
from common import agent
from common.models import (
    AgentQueryRequest,
    LlmClientMessage,
)
from .prompts import SYSTEM_PROMPT


load_dotenv(".env")
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:1420",
    "http://localhost:5050",
    "https://pro.openbb.dev",
    "https://pro.openbb.co",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WidgetDataRequest(BaseModel):
    origin: str
    widget_id: str


@app.get("/copilots.json")
def get_copilot_description():
    """Widgets configuration file for the OpenBB Terminal Pro"""
    return JSONResponse(
        content=json.load(open((Path(__file__).parent.resolve() / "copilots.json")))
    )


@app.post("/v1/query")
async def query(request: AgentQueryRequest) -> EventSourceResponse:
    """Query the Copilot."""
    chat_messages: list[Any] = [
        SystemMessage(SYSTEM_PROMPT),
    ]
    for message in request.messages:
        match message:
            case LlmClientMessage(role="human"):
                chat_messages.append(UserMessage(content=message.content))
            case LlmClientMessage(role="ai"):
                chat_messages.append(AssistantMessage(content=message.content))
            case _:
                raise ValueError(f"Unsupported message type: {message}")

    # This is the main execution loop for the Copilot.
    async def execution_loop():
        chat = Chat(
            messages=chat_messages,
            output_types=[AsyncStreamedStr],
        )
        chat = await chat.asubmit()
        if isinstance(chat.last_message.content, AsyncStreamedStr):
            async for event in agent.create_message_stream(chat.last_message.content):
                yield event

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
