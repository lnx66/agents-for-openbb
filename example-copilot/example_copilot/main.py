import base64
import io
import re
import json
from pathlib import Path
from typing import AsyncGenerator

import pdfplumber
from pydantic import BaseModel
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
from common.models import (
    AgentQueryRequest,
    DataContent,
    DataSourceRequest,
    FunctionCallSSE,
    FunctionCallSSEData,
    LlmFunctionCall,
    PdfDataFormat,
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
    chat_messages: list = []
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
            function_result_message_content = ""
            for data in message.data:
                if isinstance(data, DataContent):
                    if isinstance(data.data_format, PdfDataFormat):
                        # Handle Base64 PDF data
                        with pdfplumber.open(
                            io.BytesIO(base64.b64decode(data.content))
                        ) as pdf:
                            document_text = ""
                            for page in pdf.pages:
                                document_text += page.extract_text()
                                document_text += "\n\n"
                        function_result_message_content += (
                            f"===== {data.data_format.filename} =====\n"
                        )
                        function_result_message_content += document_text
                        function_result_message_content += "=== END CONTEXT ====\n\n"
                    else:
                        # Handle other data types
                        function_result_message_content += "===========\n"
                        function_result_message_content += data.content
                        function_result_message_content += "=== END CONTEXT ====\n\n"
            chat_messages.append(
                FunctionResultMessage(
                    content=sanitize_message(function_result_message_content),
                    function_call=function_call,
                )
            )
        elif message.role == "human":
            chat_messages.append(UserMessage(content=sanitize_message(message.content)))

    # This is the mean execution loop for the Copilot.
    async def execution_loop():
        @chatprompt(
            SystemMessage(SYSTEM_PROMPT),
            *chat_messages,
            functions=[_llm_get_widget_data] if primary_widgets else [],
        )
        async def _llm(widgets: str) -> AsyncStreamedStr: ...

        result = await _llm(widgets=widgets)
        if isinstance(result, AsyncStreamedStr):
            async for event in create_message_stream(result):
                yield event
        elif isinstance(result, FunctionCall):
            async for event in result():
                yield event
                break

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
