from typing import AsyncGenerator
import openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai.models import MessageChunkSSE, QueryRequest
from openbb_ai import message_chunk

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:1420", "http://localhost:5050", "https://pro.openbb.dev", "https://pro.openbb.co"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/agents.json")
def get_copilot_description():
    """Widgets configuration file for the OpenBB Terminal Pro"""
    return JSONResponse(
        content={
            "financial-prompt-optimizer": {
                "name": "Financial Prompt Optimizer",
                "description": "Specializes in enhancing and optimizing prompts for financial analysis queries. Improves clarity, specificity, and effectiveness of financial data requests. Maintained by Causeway Capital Management with deep expertise in financial terminology, market analysis requirements, and investment research prompting best practices.",
                "image": "https://uspto.report/TM/85344539/mark",
                "endpoints": {
                    "query": "http://localhost:7779/v1/query",
                },
                "features": {
                    "streaming": True,
                },
            },          
        }
    )

@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot."""

    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="""
You are a Financial Prompt Optimization Agent created by Causeway Capital Management.

## Your Primary Function:
Transform user queries and prompts related to finance into more effective, precise, and actionable versions.

## Core Capabilities:
- **Enhance financial terminology** for better data retrieval
- **Add relevant context** that improves query specificity  
- **Structure prompts** for optimal financial analysis
- **Include time periods, metrics, and parameters** commonly needed in finance
- **Suggest alternative phrasings** that yield better results

## Enhancement Process:
1. **Analyze the original query** for financial context and intent
2. **Identify missing specificity** (time frames, metrics, scope)
3. **Add relevant financial terminology** and context
4. **Structure the enhanced prompt** for clarity and actionability
5. **Provide the improved version** with brief explanation of changes
"""
        )
    ]

    context_str = ""
    for index, message in enumerate(request.messages):
        if message.role == "human":
            openai_messages.append(
                ChatCompletionUserMessageParam(role="user", content=message.content)
            )
        elif message.role == "ai":
            if isinstance(message.content, str):
                openai_messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=message.content
                    )
                )
        elif message.role == "tool" and index == len(request.messages) - 1:
            context_str += "Use the following data to answer the question:\n\n"
            result_str = "--- Data ---\n"
            for result in message.data:
                for item in result.items:
                    result_str += f"{item.content}\n"
                    result_str += "------\n"
            context_str += result_str

    if context_str:
        openai_messages[-1]["content"] += "\n\n" + context_str  # type: ignore

    async def execution_loop() -> AsyncGenerator[MessageChunkSSE, None]:
        client = openai.AsyncOpenAI()
        async for event in await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
        ):
            if chunk := event.choices[0].delta.content:
                yield message_chunk(chunk)

    return EventSourceResponse(
        content=(event.model_dump(exclude_none=True) async for event in execution_loop()),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7779, reload=True)