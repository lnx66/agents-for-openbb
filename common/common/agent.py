import json
from common.models import LlmClientFunctionCallResult, LlmClientMessage
from magentic import (
    AssistantMessage,
    AsyncStreamedStr,
    Chat,
    FunctionCall,
    AnyMessage,
    SystemMessage,
    UserMessage,
)
from typing import AsyncGenerator
import re


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


def prepare_messaages(
    system_prompt: str,
    messages: list[LlmClientFunctionCallResult | LlmClientMessage],
) -> list[AnyMessage]:
    chat_messages: list[AnyMessage] = [SystemMessage(system_prompt)]
    for message in messages:
        match message:
            case LlmClientMessage(role="human"):
                chat_messages.append(UserMessage(content=message.content))
            case LlmClientMessage(role="ai"):
                chat_messages.append(AssistantMessage(content=message.content))
            case _:
                raise ValueError(f"Unsupported message type: {message}")
    return chat_messages


async def run_agent(
    chat: Chat, max_completions: int = 10
) -> AsyncGenerator[dict, None]:
    completion_count = 0
    # We set a limit to avoid infinite loops.
    while completion_count < max_completions:
        completion_count += 1
        chat = await chat.asubmit()
        # Handle a streamed text response.
        if isinstance(chat.last_message.content, AsyncStreamedStr):
            async for event in create_message_stream(chat.last_message.content):
                yield event
            return
        # Handle a function call.
        elif isinstance(chat.last_message.content, FunctionCall):
            chat = await chat.aexec_function_call()
