import json
import logging
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .prompts import SYSTEM_PROMPT

from dotenv import load_dotenv
from common import agent
from common.models import (
    QueryRequest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@app.get("/copilots.json")
def get_copilot_description():
    """Widgets configuration file for the OpenBB Terminal Pro"""
    return JSONResponse(
        content=json.load(open((Path(__file__).parent.resolve() / "copilots.json")))
    )


async def get_weather(city: str) -> AsyncGenerator[str, None]:
    """Get the weather for a city."""
    yield f"The weather in {city} is sunny."


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot."""

    openbb_agent = agent.OpenBBAgent(
        query_request=request,
        system_prompt=SYSTEM_PROMPT,
        chat_class=agent.OpenRouterChat,
        model="openai/gpt-4o",  # "deepseek/deepseek-chat-v3-0324",
        functions=[get_weather],
    )

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=openbb_agent.run(),
        media_type="text/event-stream",
    )
