import json
import logging
from pathlib import Path

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


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot."""

    openbb_agent = agent.OpenBBAgent(
        query_request=request,
        system_prompt=SYSTEM_PROMPT,
        chat_class=agent.OpenRouterChat,
        model="deepseek/deepseek-r1",
    )

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=openbb_agent.run(),
        media_type="text/event-stream",
    )
