import inspect
import json
from magentic import Chat, AsyncStreamedStr, FunctionCall
from common.models import (
    AgentQueryRequest,
    Citation,
    CitationCollection,
    CitationCollectionSSE,
    DataContent,
    DataSourceRequest,
    FunctionCallSSE,
    FunctionCallSSEData,
    LlmClientFunctionCallResult,
    LlmFunctionCall,
    LlmClientMessage,
    StatusUpdateSSE,
    StatusUpdateSSEData,
    Widget,
)
from magentic import (
    AssistantMessage,
    FunctionResultMessage,
    AnyMessage,
    SystemMessage,
    UserMessage,
)
from typing import Any, AsyncGenerator, Awaitable, Callable, Literal, cast
import re
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
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


def reasoning_step(
    event_type: Literal["INFO", "WARNING", "ERROR"],
    message: str,
    details: dict[str, Any] | None = None,
) -> StatusUpdateSSE:
    return StatusUpdateSSE(
        data=StatusUpdateSSEData(
            eventType=event_type,
            message=message,
            details=[details] if details else [],
        )
    )


def remote_function_call(
    function: Literal["get_widget_data"],
    output_formatter: Callable[..., Awaitable[str]] | None = None,
    callbacks: list[Callable[..., Awaitable[Any]]] | None = None,
) -> Callable:
    if function not in ["get_widget_data"]:
        raise ValueError(
            f"Unsupported function: {function}. Must be 'get_widget_data'."
        )

    def outer_wrapper(func: Callable):
        class InnerWrapper:
            def __init__(self):
                self.__name__ = func.__name__
                self.__signature__ = self._mask_signature(func)
                self.__doc__ = func.__doc__
                self.local_function = func
                self.function = function
                self.post_process_function = output_formatter
                self.callbacks = callbacks
                self._request = None

            @property
            def request(self) -> AgentQueryRequest:
                return self._request

            @request.setter
            def request(self, request: AgentQueryRequest):
                self._request = request

            def _mask_signature(self, func: Callable):
                """Hide the `request` argument from the signature, since we want
                to pass this in automatically, but not expose it to the LLM."""
                signature = inspect.signature(func)
                masked_params = [
                    p for p in signature.parameters.values() if p.name != "request"
                ]
                return signature.replace(parameters=masked_params)

            async def execute_callbacks(
                self,
                function_call_result: LlmClientFunctionCallResult,
                request: AgentQueryRequest,
            ) -> AsyncGenerator[Any, None]:
                if self.callbacks:
                    for callback in self.callbacks:
                        if inspect.isasyncgenfunction(callback):
                            async for event in callback(function_call_result, request):
                                yield event
                        else:
                            await callback(function_call_result, self.request)

            async def execute_post_processing(self, data: list[DataContent]) -> str:
                if self.post_process_function:
                    return await self.post_process_function(data)
                return str(data)

            async def __call__(
                self, *args, **kwargs
            ) -> AsyncGenerator[FunctionCallSSE | StatusUpdateSSE, None]:
                bound_args = self.__signature__.bind(*args, **kwargs).arguments

                async for event in func(*args, request=self._request, **kwargs):
                    if isinstance(event, StatusUpdateSSE):
                        yield event
                    elif isinstance(event, DataSourceRequest):
                        yield FunctionCallSSE(
                            data=FunctionCallSSEData(
                                function=self.function,
                                input_arguments={
                                    "data_sources": [
                                        DataSourceRequest(
                                            widget_uuid=event.widget_uuid,
                                            origin=event.origin,
                                            id=event.id,
                                            input_args=event.input_args,
                                        )
                                    ]
                                },
                                extra_state={
                                    "copilot_function_call_arguments": {
                                        **bound_args,
                                    },
                                    "_locally_bound_function": func.__name__,
                                },
                            )
                        )
                        return
                    else:
                        yield event

        return InnerWrapper()

    return outer_wrapper


def get_remote_data(
    widget: Widget,
    input_arguments: dict[str, Any],
) -> DataSourceRequest:
    return DataSourceRequest(
        widget_uuid=str(widget.uuid),
        origin=widget.origin,
        id=widget.widget_id,
        input_args=input_arguments,
    )


def get_wrapped_function(function_name: str, functions: list[Any]) -> Callable:
    matching_local_functions = list(
        filter(
            lambda x: x.__name__ == function_name,
            functions,
        )
    )
    wrapped_function = matching_local_functions[0] if matching_local_functions else None
    if not wrapped_function:
        raise ValueError(f"Local function not found: {function_name}")
    return wrapped_function


async def process_messages(
    system_prompt: str,
    messages: list[LlmClientFunctionCallResult | LlmClientMessage],
) -> list[AnyMessage] | list[ChatCompletionMessageParam]:
    return await _process_messages_openai(system_prompt, messages)


async def _process_messages_openai(
    system_prompt: str,
    messages: list[LlmClientFunctionCallResult | LlmClientMessage],
) -> list[ChatCompletionMessageParam]:
    chat_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt)
    ]
    for message in messages:
        match message:
            case LlmClientMessage(role="human"):
                chat_messages.append(
                    ChatCompletionUserMessageParam(
                        role="user", content=cast(str, message.content)
                    )
                )
            case LlmClientMessage(role="ai") if isinstance(message.content, str):
                chat_messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=message.content
                    )
                )
            case _:
                raise ValueError(f"Unsupported message type: {message}")
    return chat_messages


class OpenBBAgent:
    def __init__(
        self,
        query_request: AgentQueryRequest,
        system_prompt: str,
        functions: list[Callable] | None = None,
    ):
        self.request = query_request
        self.widgets = query_request.widgets
        self.system_prompt = system_prompt
        self.functions = functions
        self._chat: Chat | None = None
        self._citations: CitationCollection | None = None
        self._messages: list[AnyMessage] = []

    async def run(self, max_completions: int = 10) -> AsyncGenerator[dict, None]:
        self._messages = await self._handle_request()
        self._citations = await self._handle_callbacks()

        self._chat = Chat(
            messages=self._messages,
            output_types=[AsyncStreamedStr, FunctionCall],
            functions=self.functions if self.functions else None,
        )
        async for event in self._execute(max_completions=max_completions):
            yield event

        if self._citations.citations:
            yield CitationCollectionSSE(data=self._citations).model_dump()

    async def _handle_callbacks(self) -> CitationCollection:
        if not self.functions:
            return CitationCollection(citations=[])
        citations: list[Citation] = []
        for message in self.request.messages:
            if isinstance(message, LlmClientFunctionCallResult):
                wrapped_function = get_wrapped_function(
                    function_name=message.extra_state.get(
                        "_locally_bound_function", ""
                    ),
                    functions=self.functions,
                )
                async for event in wrapped_function.execute_callbacks(  # type: ignore
                    request=self.request, function_call_result=message
                ):
                    if isinstance(event, Citation):
                        citations.append(event)
        return CitationCollection(citations=citations)

    async def _handle_request(self) -> list[AnyMessage]:
        chat_messages: list[AnyMessage] = [SystemMessage(self.system_prompt)]
        for i, message in enumerate(self.request.messages):
            match message:
                case LlmClientMessage(role="human"):
                    chat_messages.append(UserMessage(content=message.content))
                case LlmClientMessage(role="ai") if isinstance(message.content, str):
                    chat_messages.append(AssistantMessage(content=message.content))
                case LlmClientMessage(role="ai") if isinstance(
                    message.content, LlmFunctionCall
                ):
                    # Everything is handle in the function call result message.
                    pass
                case LlmClientFunctionCallResult(role="tool"):
                    if not self.functions:
                        continue
                    wrapped_function = get_wrapped_function(
                        function_name=message.extra_state.get(
                            "_locally_bound_function", ""
                        ),
                        functions=self.functions,
                    )

                    function_call: FunctionCall = FunctionCall(
                        function=wrapped_function,
                        **message.extra_state.get(
                            "copilot_function_call_arguments", {}
                        ),
                    )
                    chat_messages.append(AssistantMessage(function_call))

                    chat_messages.append(
                        FunctionResultMessage(
                            content=await wrapped_function.execute_post_processing(
                                message.data
                            ),
                            function_call=function_call,
                        )
                    )
                case _:
                    raise ValueError(f"Unsupported message type: {message}")
        return chat_messages

    async def _execute(self, max_completions: int) -> AsyncGenerator[dict, None]:
        completion_count = 0
        # We set a limit to avoid infinite loops.
        while completion_count < max_completions:
            completion_count += 1
            self._chat = await cast(Chat, self._chat).asubmit()
            # Handle a streamed text response.
            if isinstance(self._chat.last_message.content, AsyncStreamedStr):
                async for event in create_message_stream(
                    self._chat.last_message.content
                ):
                    yield event
                return

            # Handle a function call.
            elif isinstance(self._chat.last_message.content, FunctionCall):
                function_call_result: str = ""

                # We sneak in the request as extra state.
                self._chat.last_message.content._function.request = self.request  # type: ignore[attr-defined]
                # Execute the function.
                async for event in self._chat.last_message.content():
                    # Yield reasoning steps.
                    if isinstance(event, StatusUpdateSSE):
                        yield event.model_dump()
                    # Or an SSE to execute a function on the client-side.
                    elif isinstance(event, FunctionCallSSE):
                        yield event.model_dump()
                        return
                    # Otherwise, append to the function call result.
                    else:
                        function_call_result += str(event)
                self._chat = self._chat.add_message(
                    FunctionResultMessage(
                        content=function_call_result,
                        function_call=self._chat.last_message.content,
                    )
                )


async def run_openrouter_agent(
    messages: list[ChatCompletionMessageParam],
    model: str,
    api_key: str,
) -> AsyncGenerator[dict | StatusUpdateSSE, None]:
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    stream = await client.chat.completions.create(
        model=model, messages=messages, stream=True
    )

    reasoning = ""
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            if reasoning:
                yield reasoning_step(
                    event_type="INFO",
                    message="Reasoning complete",
                    details={"Reasoning": reasoning},
                ).model_dump()
                reasoning = ""
            yield {
                "event": "copilotMessageChunk",
                "data": json.dumps({"delta": chunk.choices[0].delta.content}),
            }
        elif (
            hasattr(chunk.choices[0].delta, "reasoning")
            and chunk.choices[0].delta.reasoning
        ):
            if not reasoning:
                yield reasoning_step(
                    event_type="INFO",
                    message="Reasoning...",
                ).model_dump()
            reasoning += chunk.choices[0].delta.reasoning  # type: ignore
    return
