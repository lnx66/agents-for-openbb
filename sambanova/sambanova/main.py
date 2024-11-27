import os
import re
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Generator
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from magentic import (
    FunctionCall,
)
from magentic.chat_model.function_schema import FunctionCallFunctionSchema
import openai
from sse_starlette.sse import EventSourceResponse

from dotenv import load_dotenv
from common.models import (
    FunctionCallSSE,
    FunctionCallSSEData,
    AgentQueryRequest,
    LlmClientFunctionCallResult,
    LlmFunctionCall,
    LlmMessage,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)
from .prompts import SYSTEM_PROMPT, format_widgets
from .functions import llm_get_widget_data

from logging import getLogger

logger = getLogger("uvicorn.error")

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


def sanitize_message(message: str) -> str:
    """Sanitize a message by escaping forbidden characters."""
    cleaned_message = re.sub(r"(?<!\{)\{(?!{)", "{{", message)
    cleaned_message = re.sub(r"(?<!\})\}(?!})", "}}", cleaned_message)
    return cleaned_message


@app.get("/copilots.json")
def get_copilot_description():
    """Widgets configuration file for the OpenBB Terminal Pro"""
    return JSONResponse(
        content=json.load(open((Path(__file__).parent.resolve() / "copilots.json")))
    )


class StreamedStr:
    def __init__(self, stream: Generator[str, None, None]):
        self.stream = stream

    def __iter__(self):
        for chunk in self.stream:
            if (
                chunk.choices[0].delta.content is not None
                and chunk.choices[0].delta.content != ""
            ):
                yield chunk.choices[0].delta.content


def create_message_stream(
    content: StreamedStr,
) -> Generator[dict, None, None]:
    for chunk in content:
        yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": chunk})}

def create_status_update_stream(
    content: StatusUpdateSSE,
) -> Generator[dict, None, None]:
    yield content.model_dump()


async def create_function_call_stream(
    content: FunctionCallSSE,
) -> AsyncGenerator[dict, None]:
    yield content.model_dump()


def do_completion(
    messages: list[LlmClientFunctionCallResult | LlmMessage],
    functions: list[Callable] | None = None,
    model: str = "Meta-Llama-3.1-70B-Instruct",
    stream: bool = True,
    **kwargs,
) -> StreamedStr | str:
    widgets = kwargs.get("widgets", [])

    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )

    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(**kwargs)}]
    for message in messages:
        match message:
            case LlmMessage(role="ai"):
                if isinstance(message.content, str):
                    formatted_messages.append(
                        {
                            "role": "assistant",
                            "content": sanitize_message(message.content),
                        }
                    )
                elif isinstance(message.content, LlmFunctionCall):
                    tool_call_id = str(uuid.uuid4())[:9]
                    formatted_messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "arguments": message.content.input_arguments,
                                        "name": message.content.function,
                                    },
                                }
                            ],
                        }
                    )
            case LlmMessage(role="human"):
                formatted_messages.append(
                    {"role": "user", "content": sanitize_message(message.content)}
                )
            case LlmClientFunctionCallResult():
                formatted_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": message.data.content,
                    }
                )

    tools = []
    if functions:
        for function in functions:
            function_schema = FunctionCallFunctionSchema(function)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": function_schema.name,
                        "description": function_schema.description,
                        "parameters": function_schema.parameters,
                    },
                }
            )

    if stream:
        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            stream=stream,
            tools=tools,
        )
        return StreamedStr(response)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            tools=tools,
            stream=stream,
        )
        if response.choices is None:
            raise ValueError(f"No choices returned from OpenAI: {response}")
        if response.choices[0].message.content:
            return response.choices[0].message.content
        elif response.choices[0].message.tool_calls:
            # For now, we'll only handle a single tool call at a time.
            tool_call = response.choices[0].message.tool_calls[0]
            arguments = tool_call.function.arguments
            function_call = FunctionCall(
                function=next(
                    f for f in functions if f.__name__ == tool_call.function.name
                ),
                **arguments,
            )
            return function_call


@app.post("/v1/query")
async def query(request: AgentQueryRequest) -> EventSourceResponse:
    """Query the Copilot."""

    logger.info(f"Received query: {request}")

    async def execution_loop() -> AsyncGenerator[Any, None]:
        calls = 0
        MAX_CALLS = 10
        while calls < MAX_CALLS:
            calls += 1

            response = do_completion(
                messages=request.messages,
                context=request.context,
                widgets=format_widgets(request.widgets) if request.widgets else [],
                functions=[llm_get_widget_data],
                stream=False,
            )

            if isinstance(response, StreamedStr):
                for event in create_message_stream(response):
                    yield event
                break
            elif isinstance(response, str):
                for event in create_message_stream(response):
                    yield event
                break
            elif isinstance(response, FunctionCall):
                logger.info(f"Function call: {response}")

                # Status update for function call
                yield StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="INFO",
                        message="Calling function",
                        group="reasoning",
                    )
                ).model_dump()

                match response.function.__name__:
                    case "llm_get_widget_data":
                        print("Streaming back function call", response.arguments)
                        yield FunctionCallSSE(
                            data=FunctionCallSSEData(
                                function="get_widget_data",
                                input_arguments=response.arguments,
                                copilot_function_call_arguments=response.arguments,
                            )
                        ).model_dump()
                        return

    return EventSourceResponse(execution_loop())
