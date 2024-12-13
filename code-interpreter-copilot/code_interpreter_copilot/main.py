from multiprocessing import Process, Queue
import re
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from magentic import (
    FunctionCall,
    FunctionResultMessage,
    chatprompt,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    AsyncStreamedStr,
)
from sse_starlette.sse import EventSourceResponse

from dotenv import load_dotenv
from common.models import AgentQueryRequest
from .prompts import SYSTEM_PROMPT
from .code_interpreter import repl_worker


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


def llm_run_code(code: str) -> str:
    """Use this tool to run Python code and get the output."""
    input_queue = Queue()
    output_queue = Queue()
    worker_process = Process(target=repl_worker, args=(input_queue, output_queue))
    worker_process.start()
    input_queue.put(code)
    result = output_queue.get()
    input_queue.put(None)  # Signal to shut down
    worker_process.join()
    return result


@app.post("/v1/query")
async def query(request: AgentQueryRequest) -> EventSourceResponse:
    """Query the Copilot."""

    chat_messages = []
    for message in request.messages:
        if message.role == "ai":
            chat_messages.append(
                AssistantMessage(content=sanitize_message(message.content))
            )
        elif message.role == "human":
            chat_messages.append(UserMessage(content=sanitize_message(message.content)))

    count = 0
    MAX_CALLS = 10
    while count < MAX_CALLS:
        count += 1

        @chatprompt(SystemMessage(SYSTEM_PROMPT), *chat_messages, functions=[llm_run_code])
        async def _llm(context: str) -> AsyncStreamedStr | FunctionCall: ...

        result = await _llm(context=request.context)
        if isinstance(result, FunctionCall):
            print("Function call: ", result)
            output = result()
            print("Output: ", output)
            # Add the function call to the chat messages
            chat_messages.append(
                AssistantMessage(result)
            )
            # Add the function result to the chat messages
            chat_messages.append(
                FunctionResultMessage(
                    content=output,
                    function_call=result,
                )
            )
        elif isinstance(result, AsyncStreamedStr):
            return EventSourceResponse(
                content=create_message_stream(result),
                media_type="text/event-stream",
            )
