import json
import os
import re
from pathlib import Path
from typing import AsyncGenerator

import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from magentic import (
    AssistantMessage,
    AsyncStreamedStr,
    FunctionCall,
    FunctionResultMessage,
    SystemMessage,
    UserMessage,
    chatprompt,
)
from pydantic import BaseModel

# from magentic.chat_model.litellm_chat_model import LitellmChatModel
from sse_starlette.sse import EventSourceResponse

from common.models import (
    AgentQueryRequest,
    DataSourceRequest,
    FunctionCallSSE,
    FunctionCallSSEData,
    LlmFunctionCall,
)
from example_copilot.prompts import SYSTEM_PROMPT

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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


def sanitize_message(message: str) -> str:
    """Sanitize a message by escaping forbidden characters."""
    cleaned_message = re.sub(r"(?<!\{)\{(?!{)", "{{", message)
    cleaned_message = re.sub(r"(?<!\})\}(?!})", "}}", cleaned_message)
    return cleaned_message


async def create_message_stream(
    content: AsyncStreamedStr,
) -> AsyncGenerator[dict, None]:
    async for chunk in content:
        yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": chunk})}


@app.get("/copilots.json")
def get_copilot_description():
    """Widgets configuration file for the OpenBB Terminal Pro"""
    return JSONResponse(
        content=json.load(open((Path(__file__).parent.resolve() / "copilots.json")))
    )


@app.post("/v1/query")
async def query(request: AgentQueryRequest) -> EventSourceResponse:
    """Query the Copilot."""

    primary_widgets = None
    if request.widgets and request.widgets.primary:
        primary_widgets = request.widgets.primary
        widget_descriptions = [
            f"origin: {widget.origin}\nwidget_id: {widget.widget_id}\nparams: {json.dumps({param.name: param.current_value for param in widget.params})}"
            for widget in request.widgets.primary
        ]
        widgets = "\n\n".join(widget_descriptions)
    else:
        widgets = "No widgets are available."

    # This is the function that may be called by the LLM to retrieve data for the primary widgets.
    async def _llm_get_widget_data(data_sources: list[WidgetDataRequest]):
        target_widgets = [
            widget
            for widget in primary_widgets
            if widget.origin == data_sources[0].origin
            and widget.widget_id == data_sources[0].widget_id
        ]

        target_data_sources = [
            DataSourceRequest(
                origin=widget.origin,
                id=widget.widget_id,
                # In this example copilot, we keep the current parameters of the
                # widget as they currently are in the OpenBB workspace. But you
                # could also change them here (for example by using an LLM
                # sub-chain to generate new parameters based on the user's
                # query).
                input_args={param.name: param.current_value for param in widget.params},
            )
            for widget in target_widgets
        ]

        # Yield the function call, which in turn will be streamed back to the
        # client. This is the schema that must be followed for the
        # FunctionCallSSE to retrieve widget data. This is interpreted by the
        # OpenBB Workspace, which then will make a follow-up request to the LLM
        # containing the function call and its result.
        yield FunctionCallSSE(
            event="copilotFunctionCall",
            data=FunctionCallSSEData(
                function="get_widget_data",
                input_arguments={"data_sources": target_data_sources},
                copilot_function_call_arguments={"data_sources": data_sources},
            ),
        ).model_dump()

    # Prepare chat messages for the LLM.
    chat_messages = []
    for message in request.messages:
        if message.role == "ai":
            # Handle function calls
            if isinstance(message.content, LlmFunctionCall):
                function_call = FunctionCall(
                    function=_llm_get_widget_data,
                    **(
                        message.content.input_arguments
                        if message.content.input_arguments
                        else {}
                    ),
                )
                chat_messages.append(AssistantMessage(function_call))
            # Handle regular assistant messages
            elif isinstance(message.content, str):
                chat_messages.append(
                    AssistantMessage(content=sanitize_message(message.content))
                )
        # Handle tool messages
        elif message.role == "tool":
            chat_messages.append(
                FunctionResultMessage(
                    content=sanitize_message(str(message.data)),
                    function_call=function_call,
                )
            )
        elif message.role == "human":
            chat_messages.append(UserMessage(content=sanitize_message(message.content)))

    # This is the mean execution loop for the Copilot.
    async def execution_loop():
        # Format messages for OpenRouter API
        formatted_messages = []
        for msg in chat_messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})

        # Make request to OpenRouter API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "openbb.dev",
                    "X-Title": "OpenBB",
                },
                json={
                    "model": "deepseek/deepseek-r1",
                    "messages": formatted_messages,
                    "include_reasoning": True,
                    "provider": {
                        "order": ["Together", "Azure"],
                        "allow_fallbacks": False,
                    },
                    "functions": (
                        [_llm_get_widget_data.__dict__] if primary_widgets else None
                    ),
                },
                timeout=30.0,
            )

        # Stream response back
        if response.status_code == 200:
            buffer = ""
            async for chunk in response.aiter_bytes():
                if chunk:
                    buffer += chunk.decode()
                    try:
                        # Try to parse the complete JSON response
                        response_data = json.loads(buffer)
                        # Extract the actual message content
                        if (
                            "choices" in response_data
                            and len(response_data["choices"]) > 0
                            and "message" in response_data["choices"][0]
                        ):
                            content = response_data["choices"][0]["message"].get("content", "")
                            yield {
                                "event": "copilotMessageChunk",
                                "data": json.dumps({"delta": content}),
                            }
                        buffer = ""  # Clear buffer after successful processing
                    except json.JSONDecodeError:
                        # If JSON is incomplete, continue accumulating chunks
                        continue
        else:
            error_message = f"Error {response.status_code}: {response.text}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_message = f"Error {response.status_code}: {error_data['error']}"
            except json.JSONDecodeError:
                pass

            yield {
                "event": "copilotMessageChunk",
                "data": json.dumps({
                    "delta": error_message
                })
            }

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
