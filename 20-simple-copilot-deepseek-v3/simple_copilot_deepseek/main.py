import json
import logging
import os
from pathlib import Path
import httpx
from typing import AsyncGenerator, Any, Dict, List
import inspect
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from magentic import (
    AsyncStreamedStr,
    Chat,
    FunctionCall,
    FunctionResultMessage,
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from sse_starlette.sse import EventSourceResponse

from .prompts import SYSTEM_PROMPT

from dotenv import load_dotenv
from common import agent
from common.models import (
    AgentQueryRequest,
    StatusUpdateSSE,
    StatusUpdateSSEData,
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

# Direct web search function that returns a string
async def perplexity_web_search(query: str) -> str:
    """Search the web using Perplexity's API through OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY environment variable is not set"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://pro.openbb.dev",  # Required by OpenRouter
        "X-Title": "OpenBB Terminal Pro",  # Optional but recommended
    }
    data = {
        "model": "perplexity/sonar",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and up-to-date information from the web."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": False  # Not using streaming here - will get complete response
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
    except Exception as e:
        error_message = f"Error searching the web: {str(e)}"
        logger.error(error_message)
        return error_message

# Custom patched version of run_agent that properly handles our perplexity_web_search function
async def custom_run_agent(chat: Chat, max_completions: int = 10) -> AsyncGenerator[dict, None]:
    completion_count = 0
    # We set a limit to avoid infinite loops.
    while completion_count < max_completions:
        completion_count += 1
        chat = await chat.asubmit()
        # Handle a streamed text response.
        if isinstance(chat.last_message.content, AsyncStreamedStr):
            async for chunk in chat.last_message.content:
                yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": chunk})}
            yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": ""})}
            return
        # Handle a function call.
        elif isinstance(chat.last_message.content, FunctionCall):
            # Check if it's our perplexity function
            if chat.last_message.content.function.__name__ == "perplexity_web_search":
                # Special handling for perplexity_web_search
                try:
                    # Print function call details for debugging
                    function_call = chat.last_message.content
                    logger.info(f"Function call details: {dir(function_call)}")
                    logger.info(f"Function call dict: {function_call.__dict__}")
                    
                    # Try to extract the query in multiple ways
                    query = None
                    
                    # Method 1: Try to get from kwargs
                    kwargs = getattr(function_call, 'kwargs', {})
                    if kwargs and 'query' in kwargs:
                        query = kwargs['query']
                        logger.info(f"Got query from kwargs: {query}")
                    
                    # Method 2: Try to access arguments through function signature
                    if query is None and hasattr(function_call, 'arguments'):
                        arguments = function_call.arguments
                        if isinstance(arguments, dict) and 'query' in arguments:
                            query = arguments['query']
                            logger.info(f"Got query from arguments: {query}")
                    
                    # Method 3: Try to get from args if it exists
                    if query is None and hasattr(function_call, 'args'):
                        args = function_call.args
                        if args and len(args) > 0:
                            query = args[0]
                            logger.info(f"Got query from args: {query}")
                    
                    # Final fallback
                    if query is None:
                        logger.warning("Could not extract query from function call")
                        query = "Unable to determine query"
                    
                    # First, yield an info message that we're searching
                    yield {"event": "statusUpdate", "data": json.dumps({
                        "eventType": "INFO",
                        "message": f"Searching the web for: {query}",
                        "details": []
                    })}
                    
                    # Call the function directly
                    result = await perplexity_web_search(query)
                    
                    # Add the result to the chat
                    chat = chat.add_message(
                        FunctionResultMessage(
                            content=result,
                            function_call=chat.last_message.content,
                        )
                    )
                except Exception as e:
                    logger.error(f"Error in perplexity_web_search: {str(e)}", exc_info=True)
                    error_result = f"Error searching the web: {str(e)}"
                    chat = chat.add_message(
                        FunctionResultMessage(
                            content=error_result,
                            function_call=chat.last_message.content,
                        )
                    )
            else:
                # For other functions, use the standard approach
                function_call_result = ""
                try:
                    # Call the function and iterate over its results
                    function_call = chat.last_message.content
                    # Instead of iterating, just call it directly
                    function_result = await function_call()
                    function_call_result = str(function_result)
                except Exception as e:
                    logger.error(f"Error calling function: {str(e)}", exc_info=True)
                    function_call_result = f"Error: {str(e)}"
                
                # Add the function result to the chat
                chat = chat.add_message(
                    FunctionResultMessage(
                        content=function_call_result,
                        function_call=chat.last_message.content,
                    )
                )
        else:
            # If the last message is not a function call or streamed output, 
            # it's probably just regular content - yield it and return
            if hasattr(chat.last_message, 'content') and chat.last_message.content:
                yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": str(chat.last_message.content)})}
                yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": ""})}
            else:
                yield {"event": "error", "data": json.dumps({"message": "Unexpected response type from LLM"})}
            return

@app.get("/copilots.json")
def get_copilot_description():
    """Widgets configuration file for the OpenBB Terminal Pro"""
    return JSONResponse(
        content=json.load(open((Path(__file__).parent.resolve() / "copilots.json")))
    )

@app.post("/v1/query")
async def query(request: AgentQueryRequest) -> EventSourceResponse:
    """Query the Copilot."""
    
    # Simple, direct approach without complex streaming logic
    async def direct_response():
        try:
            # Process the messages
            processed_messages = await agent.process_messages(
                system_prompt=SYSTEM_PROMPT,
                messages=request.messages,
            )
            
            # Format messages for OpenAI API
            formatted_messages = []
            for msg in processed_messages:
                if isinstance(msg, SystemMessage):
                    formatted_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, UserMessage):
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AssistantMessage) and isinstance(msg.content, str):
                    formatted_messages.append({"role": "assistant", "content": msg.content})
            
            # Create tools definition for web search
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "perplexity_web_search",
                        "description": "Search the web using Perplexity's API through OpenRouter.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to look up on the web"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
            
            # Get API key
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                yield {"event": "error", "data": json.dumps({"message": "OPENROUTER_API_KEY not set"})}
                return
                
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://pro.openbb.dev",
                "X-Title": "OpenBB Terminal Pro",
            }
            
            # Start with non-streaming call to detect tool calls
            data = {
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": formatted_messages,
                "stream": False, # Start with non-streaming
                "tools": tools,
                "tool_choice": "auto"
            }
            
            logger.info("Making initial request to detect tool calls")
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                
                # Check for tool calls
                choices = result.get("choices", [])
                if choices and "tool_calls" in choices[0].get("message", {}):
                    tool_calls = choices[0]["message"]["tool_calls"]
                    
                    # Process each tool call
                    for tool_call in tool_calls:
                        if tool_call["function"]["name"] == "perplexity_web_search":
                            try:
                                # Extract query
                                args = json.loads(tool_call["function"]["arguments"])
                                query = args.get("query", "")
                                
                                # Inform user we're searching
                                yield {"event": "statusUpdate", "data": json.dumps({
                                    "eventType": "INFO",
                                    "message": f"Searching the web for: {query}",
                                    "details": []
                                })}
                                
                                # Call perplexity
                                search_result = await perplexity_web_search(query)
                                
                                # Add result to messages
                                new_messages = formatted_messages.copy()
                                new_messages.append({
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [{
                                        "type": "function",
                                        "function": {
                                            "name": "perplexity_web_search",
                                            "arguments": json.dumps({"query": query})
                                        },
                                        "id": tool_call["id"]
                                    }]
                                })
                                new_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": search_result
                                })
                                
                                # Get final response with streaming
                                final_data = {
                                    "model": "deepseek/deepseek-chat-v3-0324",
                                    "messages": new_messages,
                                    "stream": True,
                                }
                                
                                logger.info("Making final streaming request with search results")
                                async with client.stream("POST", url, headers=headers, json=final_data) as stream_response:
                                    stream_response.raise_for_status()
                                    async for line in stream_response.aiter_lines():
                                        if not line.startswith("data: "):
                                            continue
                                            
                                        line = line[6:].strip()
                                        if line == "[DONE]":
                                            break
                                            
                                        try:
                                            chunk = json.loads(line)
                                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                                            if content:
                                                yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": content})}
                                        except json.JSONDecodeError:
                                            continue
                                
                                # Signal end of response
                                yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": ""})}
                                
                            except Exception as e:
                                logger.error(f"Error processing web search: {str(e)}", exc_info=True)
                                yield {"event": "error", "data": json.dumps({"message": f"Error with web search: {str(e)}"})}
                                return
                    
                # No tool calls detected, just stream the normal response
                else:
                    # Extract the assistant's message and stream it chunk by chunk
                    content = ""
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                    
                    # Stream the content in small chunks to simulate streaming
                    chunk_size = 6  # Small chunks for smooth streaming
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i+chunk_size]
                        yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": chunk})}
                        await asyncio.sleep(0.01)  # Small delay for smooth streaming
                        
                    # Signal end of response
                    yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": ""})}
                
        except Exception as e:
            logger.error(f"Error in direct_response: {str(e)}", exc_info=True)
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
    
    # Use our simpler direct approach
    return EventSourceResponse(
        content=direct_response(),
        media_type="text/event-stream",
    )
