**Note: This is a work-in-progress. API definitions, models, specifications, etc. are still in active development and may change without notice.**

# Bring your own Agent to the OpenBB Workspace

Welcome to the example repository for integrating custom agents into the OpenBB Workspace.

This repository provides everything you need to build and add your own custom
agents that are compatible with the OpenBB Workspace.

Here are a few common reasons why you might want to build your own agent:
- You have a unique data source that you don't want to add as a custom integration to OpenBB.
- You want to use a specific LLM.
- You want to use a local LLM.
- You want a agent that is self-hosted on your infrastructure.
- You are running on-premise in a locked-down environment that doesn't allow data to leave your VPC.


## Overview

To integrate a custom agent that you can interact with from the OpenBB Workspace,
you'll need to create a backend API that the OpenBB Workspace can make requests to.  


Your custom agent API will respond with Server-Sent Events
([SSEs](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)).

**Note: If you're looking to get started
quickly, we suggest running one of the example agents included as part of
this repository, and adding it as a custom copilot to the OpenBB Workspace (each example copilot includes instructions on how to run them). Cloning and modifying an example copilot is a great way to build a custom copilot.**

## Examples
If you prefer diving straight into code, we have a growing list of examples of
custom agents in this repository, varying in complexity and features:

- [A basic conversational agent](./01-simple-copilot)
- [A simple agent with local function calling](./02-simple-copilot-local-function-calling)
- [A simple agent that yields reasoning steps to OpenBB Workspace](./03-simple-copilot-reasoning-steps)
- [A simple agent that uses remote function calling to retrieve data from OpenBB Workspace](./04-simple-copilot-openbb-function-calling)
- [A simple agent that can retrieve data from OpenBB Workspace and produce citations](./05-simple-copilot-openbb-citations)
- [A simple agent using DeepSeek v3](./20-simple-copilot-deepseek-v3)
- [A reasoning agent using DeepSeek R1](./21-reasoning-copilot-deepseek-r1)

These examples are a good starting point for building your own custom agent if
you are interested in a specific feature or use case.

## Usage

We recommend using FastAPI to build your custom agent API. All examples and documentation will use FastAPI. If you are new to FastAPI, we recommend checking out the [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/).

### A basic conversational agent

The most basic custom agent is capable only of chatting with the user.

First, let's set our OPENAI_API_KEY:

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

Now, let's create a basic FastAPI app that uses the `OpenBBAgent` to chat with the user:

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from dotenv import load_dotenv
from common import agent
from common.models import (
    QueryRequest,
)

load_dotenv(".env")
app = FastAPI()

origins = [
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
    return JSONResponse(
        content={
            "simple_copilot": {
                "name": "Simple Copilot",
                "description": "A simple copilot that can answer questions.",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {"query": "http://localhost:7777/query"},
                "features": {"streaming": True},
            }
        }
    )


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot."""
    openbb_agent = agent.OpenBBAgent(
        query_request=request,
        system_prompt="You are a helpful assistant that can answer questions.",
    )

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=openbb_agent.run(),
        media_type="text/event-stream",
    )

```

A custom agent consists of an API with two endpoints:

1. `/copilots.json` - Describes your custom agent to the OpenBB Workspace
2. `/query` - Receives requests from the OpenBB Workspace and responds with SSEs

#### `agent.OpenBBAgent`

The `agent.OpenBBAgent` object is responsible for handling the request from the
OpenBB Workspace, and for streaming the response back to the OpenBB Workspace
via SSEs. All function calling execution, state management, request parsing, etc. is handled automatically for you by the `OpenBBAgent` object.

To stream back messages to the OpenBB Workspace, you must return an
`EventSourceResponse` object from your `/query` endpoint, where the `content` is
the result of calling `OpenBBAgent.run()`.

And that's it! All you need to do now is run the FastAPI app:

```bash
uvicorn main:app --reload --port 7777
```

And add it to the OpenBB Workspace as a custom agent:

https://github.com/user-attachments/assets/bd8daae2-6ef4-473e-abbc-cbb92fdec098

You can now chat with your custom agent in the OpenBB Workspace.

### Function calling

There are two types of function calling supported by a custom agent:

1. Local function calling
2. Remote function calling

**Local function calling** is what you're probably used to from other agents. The
function call is made by the LLM powering your custom agent, and executed
directly by the same process / backend that is running your agent (for example,
locally on your machine).

**Remote function calling** is a special type of function call specific to
OpenBB Workspace that is executed by the OpenBB Workspace web app (on the front
end, rather than locally on your machine) to fetch widget data and send it to
your custom agent.

OpenBB Workspace is uniquely architected in such a way that it acts not only as
a front-end, but also as a data gateway for all widgets added to the OpenBB
Workspace. This unique architecture allows you to connect a custom agent,
running locally on your machine, to widgets added to the OpenBB Workspace
(whether they're running locally on your machine or remotely on OpenBB Cloud).

#### Local function calling

To add a local function call to your custom agent, you must define a function
that can be called by the LLM powering your custom agent, and then add that
function to the `functions` argument of the `OpenBBAgent` object.

For example, let's add a function that returns a list of random stout beers:

```python
import random
from typing import AsyncGenerator
from pydantic import BaseModel
import httpx


class Rating(BaseModel):
    average: float
    reviews: int


class Beer(BaseModel):
    id: int
    price: str
    name: str
    rating: Rating
    image: str


async def get_random_stout_beers(n: int = 1) -> AsyncGenerator[str, None]:
    """Get a random stout beer from the Beer API.

    It is recommended to display the image url in the UI
    for the user so that they can see the beer.

    Parameters:
        n: int = 1
            The number of beers to return.
            Maximum is 10.


    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.sampleapis.com/beers/stouts",
            headers={"User-Agent": "OpenBB Example Copilot"},
        )
        if response.status_code != 200:
            yield "Failed to fetch beers."
            return

        data = response.json()
        random_sample = random.sample(data, n)
        beers = [Beer(**beer) for beer in random_sample]

        response_str = "-- Beers --\n"
        for beer in beers:
            response_str += f"name: {beer.name}\n"
            response_str += f"price: {beer.price}\n"
            response_str += f"rating: {beer.rating.average}\n"
            response_str += f"reviews: {beer.rating.reviews}\n"
            response_str += (
                f"image (use this to display the beer image): {beer.image}\n"
            )
            response_str += "-----------------------------------\n"
        yield response_str
        return


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot."""
    openbb_agent = agent.OpenBBAgent(
        query_request=request,
        system_prompt="You are a helpful assistant that can answer questions and retrieve stout beers.",
        functions=[get_random_stout_beers],  # <-- add the function to the `functions` argument
    )

    return EventSourceResponse(
        content=openbb_agent.run(),
        media_type="text/event-stream",
    )

```

A few important things to note:
- The function must be an `async` function.
- The function must be type-hinted.
- The function must yield a string as its output.

It is also recommended to:
- Use a descriptive function name (this is passed to the LLM).
- Use a docstring to describe the function and its parameters (this is passed to the LLM).

Now, when you query the agent, it will be able to call the `get_random_stout_beers` function:

<img src="https://openbb-assets.s3.us-east-1.amazonaws.com/docs/custom_copilot/local_function_calling_example.png" alt="Local Function Calling Example" style="max-width: 600px;">

#### Remote function calling

To allow our custom agent to retrieve data from widgets on a dashboard on OpenBB Workspace,
we must use remote function calling.

To do this, we must first define a function that can be called by the LLM powering our custom agent, and decorate it with the `@remote_function_call` decorator. We must also specify an `output_formatter` that will be used to format the output of the data retrieved from the widget, so that it can be passed back to the LLM.

Here is a minimal example that will allow our custom agent to retrieve data from widgets
that has been added as priority context (without modifying their input parameters):

```python
from typing import AsyncGenerator
from common import agent
from common.models import (
    QueryRequest,
    DataContent,
    FunctionCallSSE,
    StatusUpdateSSE,
)

async def handle_widget_data(data: list[DataContent]) -> str:
    result_str = "--- Data ---\n"
    for content in data:
        result_str += f"{content.content}\n"
        result_str += "------\n"
    return result_str

@agent.remote_function_call(
    function="get_widget_data",
    output_formatter=handle_widget_data,
)
async def get_widget_data(
    widget_uuid: str,
    request: QueryRequest,  # Must be included as an argument
) -> AsyncGenerator[FunctionCallSSE | StatusUpdateSSE, None]:
    """Retrieve data for a widget by specifying the widget UUID."""

    widgets = (
        request.widgets.primary + request.widgets.secondary if request.widgets else []
    )

    # Get the first widget that matches the UUID (there should be only one).
    matching_widgets = list(
        filter(lambda widget: str(widget.uuid) == widget_uuid, widgets)
    )
    widget = matching_widgets[0] if matching_widgets else None

    # If we're unable to find the widget, let's let the LLM know.
    if not widget:
        yield f"Unable to retrieve data for widget with UUID: {widget_uuid} (it is not present on the dashboard)"  # noqa: E501
        return

    # Yield the request to the front-end for the widget data.
    # NB: You *must* yield using `agent.remote_data_request` from inside
    # remote functions (i.e. those decorated with `@remote_function`).
    yield agent.remote_data_request(
        widget=widget,
        # In this example we will just re-use the currently-set values of
        # the widget parameters.
        input_arguments={param.name: param.current_value for param in widget.params},
    )
    return


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot."""
    openbb_agent = agent.OpenBBAgent(
        query_request=request,
        system_prompt="You are a helpful assistant that can answer questions and retrieve stout beers.",
        functions=[get_widget_data],  # <-- add the function to the `functions` argument
    )

    return EventSourceResponse(
        content=openbb_agent.run(),
        media_type="text/event-stream",
    )

```

In the example above, we choose to use the currently-set values of the widget
parameters. In a more advanced example, we could modify the input parameters of
the widget depending on the user query.

A few important things to note:
- The function must be an `async` function.
- The function must accept the `request` argument, which will be passed into the function when it is called.
- The function must be decorated with the `@remote_function_call` decorator.
- In the `@remote_function_call` decorator, we must specify the `function` name that corresponds to the functions supported by the OpenBB Workspace. Currently only "get_widget_data" is supported.
- It's optional, but it is recommended to specify an `output_formatter` function that will be used to format the output of the data retrieved from the widget, so that it can be passed back to the LLM in a readable format. If not specified, the
result of the function call is simply dumped as a string to the LLM.
- The function must yield the result of the `get_remote_data` function, which must specify the `widget` and `input_arguments` to retrieve data for.

The `request` argument is the same `QueryRequest` object passed into the `query`
endpoint. It contains the current conversation's messages, any explicitly-added
context, information about widgets on the currently-active dashboard, and so on.
You can view the full schema either by looking at the `QueryRequest` model in
the
[common/models.py](https://github.com/OpenBB-finance/copilot-for-terminal-pro/blob/main/common/common/models.py)
file, or by inspecting the `QueryRequest` model in the Swagger UI of a custom
agent (at `<your-custom-agent-url>/docs`, eg. `http://localhost:7777/docs`).

### Reasoning Steps / Status Updates

Sometimes, it can be valuable to send reasoning steps (also sometimes referred
to as status updates) that contain extra information back to the OpenBB
Workspace while your custom agent is performing tasks. This is useful for
providing the user with feedback on the status of the task, or for providing the
user with additional information that can help them understand the task at hand.

To send a reasoning step back to the OpenBB Workspace, you `yield` from the `reasoning_step` function from within one of the functions you've added to your custom agent.

For example, let's modify the `get_random_stout_beers` function above to send reasoning steps back to the OpenBB Workspace while it executes:

```python
from common import agent

async def get_random_stout_beers(n: int = 1) -> AsyncGenerator[str, None]:
    """Get a random stout beer from the Beer API.

    It is recommended to display the image url in the UI
    for the user so that they can see the beer.

    Parameters:
        n: int = 1
            The number of beers to return.
            Maximum is 10.


    """

    # ** NEW ** 
    yield agent.reasoning_step(
        event_type="INFO",
        message="Fetching random stout beers...",
        details={"number of beers": n},
    )

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.sampleapis.com/beers/stouts",
            headers={"User-Agent": "OpenBB Example Copilot"},
        )
        if response.status_code != 200:
            # ** NEW ** 
            yield agent.reasoning_step(
                event_type="ERROR",
                message="Failed to fetch beers.",
                details={"error": "Failed to fetch beers."},
            )
            yield "Failed to fetch beers."
            return

        # ** NEW ** 
        yield agent.reasoning_step(
            event_type="INFO",
            message="Beers fetched successfully.",
        )

        data = response.json()
        random_sample = random.sample(data, n)
        beers = [Beer(**beer) for beer in random_sample]

        response_str = "-- Beers --\n"
        for beer in beers:
            response_str += f"name: {beer.name}\n"
            response_str += f"price: {beer.price}\n"
            response_str += f"rating: {beer.rating.average}\n"
            response_str += f"reviews: {beer.rating.reviews}\n"
            response_str += (
                f"image (use this to display the beer image): {beer.image}\n"
            )
            response_str += "-----------------------------------\n"
        yield response_str
        return
```

This results in the following reasoning step being displayed in the OpenBB Workspace:



Some things to note:
- The `yield reasoning_step(...)` must be called from within the function you've added to your custom agent.
- The reasoning step can have an `event_type` of `INFO`, `WARNING`, or `ERROR`.
- The reasoning step must specify a `message`, which will be displayed to the user in the OpenBB Workspace.
- The reasoning step can optionally include a `details` dictionary, which will be displayed to the user as a table in the OpenBB Workspace, if they expand the reasoning step.

Reasoning steps can be yielded from both local and remote functions.


## Handling requests from the OpenBB Workspace

The OpenBB Workspace will make POST requests to the `query` endpoint defined in your
`copilots.json` file (more on this later). The payload of this request will
contain data such as the current conversation's messages, any explicitly-added
context, information about widgets on the currently-active dashboard, URLs to
retrieve, and so on.

### API Request Schema

The core of the query request schema you must implement is as follows:

```python
{
  "messages": [  # <-- the chat messages between the user and the copilot (including function calls and results)
    {
      "role": "human",  # <-- each message has a role: "human", "ai", or "tool"
      "content": "Hi there."  # <-- the content of the message
    },
    ...
  ],
  "widgets": {  # <-- an object with primary, secondary, and optional extra lists
    "primary": [  # <-- explicitly added widgets (highest priority)
      {
        "origin": "<origin>", # <-- the origin of the widget (OpenBB API, custom backend name, etc.)
        "widget_id": "<widget_id>", # <-- the ID of the widget  (eg. "stock_price_quote")
        "name": "<widget name>", # <-- the name of the widget (eg. "Stock Price Quote")
        "description": "<widget description>", # <-- the description of the widget
        "params": [  # <-- parameters for the widget
          {
            "name": "<parameter_name>", # <-- parameter name
            "type": "<parameter_type>", # <-- parameter type (string, number, etc.)
            "description": "<parameter description>",  # <-- parameter description
            "current_value": "<current value>", # <-- currently set value of the parameter on the widget
            "default_value": "<default value>" # <-- default value of the parameter on the widget if it's not set
          },
          ...
        ],
        "metadata": {
          "<metadata key>": "<metadata value>",
          ...
        }
      },
      ...
    ],
    "secondary": [  # <-- dashboard widgets (second priority)
      {
        "origin": "<origin>",
        "widget_id": "<widget_id>",
        "name": "<widget name>",
        "description": "<widget description>",
        "params": [],  # <-- same schema as primary params  
        "metadata": {} # <-- same schema as primary params  
      },
      ...
    ],
    "extra": [  # <-- optional list of all other available widgets (only present of the global data toggle is enabled)
      {
        "origin": "<origin>",
        "widget_id": "<widget_id>",
        "name": "<widget name>",
        "description": "<widget description>",
        "params": [  # <-- `extra` widgets don't have current values, since the widgets are not currently visible on the dashboard
          {
            "name": "<parameter_name>",
            "type": "<parameter_type>", 
            "description": "<parameter description>",
            "default_value": "<default value>"
          },
          ...
        ],
        "metadata": {}
      },
      ...
    ]
  },
  "context": [  # <-- reserved for artifacts returned by the custom copilot (optional)
    {
      "uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",  # <-- the UUID of the context
      "name": "<context name>",  # <-- the name of the context
      "description": "<context description>",  # <-- the description of the context
      "data": {
        "content": "<data>"  # <-- the data of the context
      },
      "metadata": {
        "<metadata key>": "<metadata value>",  # <-- the metadata of the context
        ...
      }
    },
    ...
  ]
}
```

We'll go over each of these fields in more detail below.

#### `messages`
This is the list of messages between the user and the copilot. This includes the
user's messages, the copilot's messages, function calls, and function call
results. Each message has a `role` and `content`.

The simplest example is when no function calling is involved, which simply
consists of an array of `human` and `ai` messages.

The OpenBB Workspace automatically appends all return `ai` messages (from your Copilot)
to the `messages` array of any follow-up request.

```python
# Only one human message
{
  "messages": [
    {
      "role": "human", 
      "content": "Hi there."
    }
  ],
  ...
}
```

```python
# Multiple messages
{
  "messages": [
    {
      "role": "human", 
      "content": "Hi there."
    },
    {
      "role": "ai", 
      "content": "Hi there, I'm a copilot. How are you?"
    },
    {
      "role": "human", 
      "content": "I'm fine, thank you. What is the weather in Tokyo?"
    }
  ],
  ...
}
```

Function calls to the OpenBB Workspace (such as when retrieving widget data), as well as the results of those function calls (containing the widget data), are also included in the `messages` array. For information on function calling, see the "Function Calling" section below.

#### `widgets`

This is an object containing a collection of widgets that added explicitly by the user as context (`primary`),
widgets in the currently-active dashboard (`secondary`), or widgets that are globally available if "global data" toggle is enabled (`extra`).

```python
{
  ...
  "widgets": {
    "primary": [
      {
        "origin": "OpenBB API",
        "widget_id": "stock_price_quote",
        "name": "Stock Price Quote Widget",
        "description": "Contains the current stock price of a ticker",
        "params": [
          {
            "name": "ticker",
            "type": "string",
            "description": "Stock ticker symbol",
            "current_value": "TSLA",
            "default_value": "AAPL"
          }
        ],
        "metadata": {
          "ticker": "TSLA"
        }
      }
    ],
    "secondary": [
      {
        "origin": "OpenBB API",
        "widget_id": "financial_ratios",
        "name": "Financial Ratios Widget",
        "description": "Displays key financial ratios for a company",
        "params": [
          {
            "name": "ticker",
            "type": "string",
            "description": "Stock ticker symbol",
            "current_value": "AAPL",
            "default_value": "AAPL"
          },
          {
            "name": "period",
            "type": "string",
            "description": "Time period for ratios",
            "current_value": "TTM",
            "default_value": "TTM"
          }
        ],
        "metadata": {
          "lastUpdated": 1737111708292,  
          "source": "Financial Modelling Prep"  
        }
      }
    ],
    "extra": []
  },
  ...
}
```

#### `context`

This is an optional array of artifact data that will be sent by the OpenBB Workspace
when artifacts have been returned by your custom copilot. 

The `context` field works as follows:

```python
{
  ...
  "context": [
    {
      "uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",  # <-- each context has a UUID
      "name": "chart_artifact_123a5",
      "description": "A chart showing the stock price of AAPL over the last 30 days",
      "data": {
        "content": "<data>"  # <-- the data of the context could either be a JSON string or plaintext (you must choose how to handle this in your copilot)
      },
      "metadata": {  # <-- additional metadata about the context
          "lastUpdated": 1737111708292,  
          "source": "Financial Modelling Prep"  
      }
    },
    {
      "uuid": "8b2e5f79-3a1d-4c9e-b6f8-1e7d2a9c0b3d",  # <-- there can be multiple contexts
      "name": "table_artifact_4534as",  
      "description": "A table showing the management team of AAPL",
      "data": {
        "content": "<data>"  # <-- the data of the context
      },
      "metadata": {
        ...
      }
    },
    ...
  ],
  ...
}
```


## Responding to the OpenBB Workspace

Your custom copilot must respond to the OpenBB Workspace's request using a variety of Server-Sent Events (SSEs).

The OpenBB Workspace can process the following SSEs:

- `copilotMessageChunk`: Used to return streamed copilot tokens (partial
responses) back to the OpenBB Workspace These responses can be streamed as they are
generated.
- `copilotMessageArtifact`: Used to return an artifact to the OpenBB Workspace as part
of the Copilot's response. This allows your copilot to return tables, charts,
and long-form text excerpts that will be rendered by the OpenBB Workspace Often
interleaved the `copilotMessageChunk` SSEs.
- `copilotCitationCollection`: Used to return a collection of citations back to the OpenBB Workspace This is useful for returning structured data, such as a list of news articles, research reports, or other sources that were used to generate the Copilot's response. This should be returned after the `copilotMessageChunk` SSEs have finished streaming.
- `copilotFunctionCall`: Used to request data (e.g., widget data) or perform a
specific function. This instructs the OpenBB Workspace to take further action on the
client's side. This is only necessary if you're planning on implementing
function calling in your custom copilot.
- `copilotStatusUpdate`: Used to send status updates or "reasoning steps" back to
the OpenBB Workspace These are user-friendly "updates" that are displayed in the
copilot window of the OpenBB Workspace, and are useful for informing the user about what your custom copilot is doing under-the-hood.

#### `copilotMessageChunk`
The message chunk SSE has the following format:

```
event: copilotMessageChunk
data: {"delta":"H"}  # <-- the `data` field must be a JSON object.
```
The `delta` must be a string, but can be of any length. We suggest streaming
back each chunk you receive from your LLM as soon as it's generated as a `delta`.

For example, if you wanted to stream back the message "Hi!", you would send the
following SSEs:

```
event: copilotMessageChunk
data: {"delta":"H"}

event: copilotMessageChunk
data: {"delta":"i"}

event: copilotMessageChunk
data: {"delta":"!"}
```

#### `copilotMessageArtifact`
The message artifact SSE has the following format:

```
event: copilotMessageArtifact
data: {"type": "<type>", "uuid": "<uuid>", "content": "<content>", "chart_params": <chart_params>}
```

An artifact can be a table, chart, or long-form text excerpt.
Let's go over each of these with examples:

A text artifact:

```
{
    "type": "text",
    "uuid": "123e4567-e89b-12d3-a456-426614174000",
    "content": "This is a sample text content",
}
```

#### `copilotFunctionCall` (required for retrieving widget data -- see below)
The function call SSE has the following format:

```
event: copilotFunctionCall
data: {"function":"get_widget_data","input_arguments":{"data_sources":[{"origin":"OpenBB API","id":"company_news","input_args":{"symbol":"AAPL","channels":"All","start_date":"2021-01-16","end_date":"2025-01-16","topics":"","limit":50}}]},"copilot_function_call_arguments":{"data_sources":[{"origin":"OpenBB API","widget_id":"company_news"}]}}
```

Again, the `data` field must be a JSON object. The `function` field is the name
of the function to be called (currently only `get_widget_data` is supported),
and the `input_arguments` field is a dictionary of arguments to be passed to the
function. For the `get_widget_data` function, the only required argument is
`data_sources`, which is a list of objects specifying the origin, widget ID, and input arguments for each widget to retrieve data for.


## Function Calling

Function calling makes it possible for your copilot to request data from widgets. 

A list of all available widgets is sent to your copilot in the `widgets` field
of the request payload. They are categorized according to explicitly-added widgets (`primary`), widgets in the user's currently-active dashboard (`secondary`), and widgets that are globally available if the "global data" toggle is enabled (`extra`).

To retrieve the data from a widget, your copilot should respond with a
`copilotFunctionCall` event, specifying the widget origin, widget ID, and input arguments:

```
event: copilotFunctionCall
data: {"function":"get_widget_data","input_arguments":{"data_sources":[{"origin":"OpenBB API","id":"company_news","input_args":{"symbol":"AAPL","channels":"All","start_date":"2021-01-16","end_date":"2025-01-16","topics":"","limit":50}}]},"copilot_function_call_arguments":{"data_sources":[{"origin":"OpenBB API","widget_id":"company_news"}]}}
```


After emitting a `copilotFunctionCall` event, you must close the connection and wait for a new query request from the OpenBB Workspace

When a `copilotFunctionCall` event is received, the OpenBB Workspace will retrieve
the data, and initiate a **new** query request. This new query request will
include the original function call, as well as the function call result in the
`messages` array.

```python
{
  ...
  "messages": [
    ...
    {
      "role": "ai",
      "content": "{\"function\":\"get_widget_data\",\"input_arguments\":{\"data_sources\":[{\"origin\":\"OpenBB API\",\"id\":\"company_news\",\"input_args\":{\"symbol\":\"AAPL\",\"channels\":\"All\",\"start_date\":\"2021-01-16\",\"end_date\":\"2025-01-16\",\"topics\":\"\",\"limit\":50}}]},\"copilot_function_call_arguments\":{\"data_sources\":[{\"origin\":\"OpenBB API\",\"widget_id\":\"company_news\"}]}}"
    },
    {
      "role": "tool", 
      "function": "get_widget_data",
      "input_arguments": {
        "data_sources": [
          {
            "origin": "OpenBB API",
            "id": "company_news",
            "input_args": {
              "symbol": "AAPL",
              "channels": "All", 
              "start_date": "2021-01-16",
              "end_date": "2025-01-16",
              "topics": "",
              "limit": 50
            }
          }
        ]
      },
      "copilot_function_call_arguments": {
        "data_sources": [
          {
            "origin": "OpenBB API",
            "widget_id": "company_news"
          }
        ]
      },
      "data": [  # <-- the data field is a list of objects, each containing the data for a widget (in order) that was requested in the `data_sources` field of the `copilotFunctionCall` event.
        {
          "content": "<data>"
        },
        ...
      ]
    }
  ]
}
```

Notice that:
- Both the function call and the function call result are included in the `messages` array. 
- The `content` field of the function call `ai` message is a verbatim string-encoded JSON object of the `data` field of the `copilotFunctionCall` event (this is a very useful mechanism for smuggling additional metadata related to the function call, if your copilot needs it).

Currently, the only function call supported by the OpenBB Workspace is `get_widget_data`, which retrieves data from a specific widget.

### Function call example

Your custom copilot receives the following request from the OpenBB Workspace:

```json
{
  "messages": [
    {
      "role": "human",
      "content": "What is the current stock price of AAPL?"
    }
  ],
  "widgets": {
    "primary": [
      {
        "origin": "openbb_api",
        "widget_id": "historical_stock_price",
        "name": "Historical Stock Price",
        "description": "Historical Stock Price",
        "params": [
          {
            "name": "symbol",
            "type": "string",
            "description": "Stock ticker symbol",
            "current_value": "AAPL",
            "default_value": "AAPL"
          }
        ],
        "metadata": {
          "symbol": "AAPL",
          "source": "Financial Modelling Prep",
          "lastUpdated": 1728994470324
        }
      }
    ],
    "secondary": [],
    "extra": []
  }
}
```

You then parse the response, format the messages to your LLM (including information on which widgets are available).  Let's assume
that your copilot determines that the user's query can be answered using the widget available, and generates a function call to retrieve the data.

Your copilot then responds with the following SSE and close the connection:

```
event: copilotFunctionCall
data: {"function":"get_widget_data","input_arguments":{"data_sources":[{"origin":"openbb_api","id":"historical_stock_price","input_args":{"symbol":"AAPL"}}]},"copilot_function_call_arguments":{"data_sources":[{"origin":"openbb_api","widget_id":"historical_stock_price"}]}}
```

The OpenBB Workspace will then execute the specified function, and make a new query request to your custom copilot:

```python
{
  "messages": [
    {
      "role": "human",
      "content": "What is the current stock price of AAPL?"
    },
    {
      "role": "ai",
      "content": "{\"function\":\"get_widget_data\",\"input_arguments\":{\"data_sources\":[{\"origin\":\"openbb_api\",\"id\":\"historical_stock_price\",\"input_args\":{\"symbol\":\"AAPL\"}}]},\"copilot_function_call_arguments\":{\"data_sources\":[{\"origin\":\"openbb_api\",\"widget_id\":\"historical_stock_price\"}]}}"
    },
    {
      "role": "tool",
      "function": "get_widget_data",
      "input_arguments": {
        "data_sources": [
          {
            "origin": "openbb_api",
            "id": "historical_stock_price",
            "input_args": {
              "symbol": "AAPL"
            }
          }
        ]
      },
      "copilot_function_call_arguments": {
        "data_sources": [
          {
            "origin": "openbb_api",
            "widget_id": "historical_stock_price"
          }
        ]
      },
      "data": [
        {
          "content": "[{\"date\":\"2024-10-15T00:00:00-04:00\",\"open\":233.61,\"high\":237.49,\"low\":232.37,\"close\":233.85,\"volume\":61901688,\"vwap\":234.33,\"adj_close\":233.85,\"change\":0.24,\"change_percent\":0.0010274},{\"date\":\"2024-10-14T00:00:00-04:00\",\"open\":228.7,\"high\":231.73,\"low\":228.6,\"close\":231.3,\"volume\":39882100,\"vwap\":230.0825,\"adj_close\":231.3,\"change\":2.6,\"change_percent\":0.0114},{\"date\":\"2024-10-11T00:00:00-04:00\",\"open\":229.3,\"high\":233.2,\"low\":228.9,\"close\":231.0,\"volume\":32581944,\"vwap\":231.0333,\"adj_close\":231.0,\"change\":1.7,\"change_percent\":0.0074}, ... ]"
        }
      ]
    }
  ],
  "widgets": {
    "primary": [
      {
        "origin": "OpenBB API",
        "widget_id": "historical_stock_price",
        "name": "Historical Stock Price",
        "description": "Historical Stock Price",
        "params": [
          {
            "name": "symbol",
            "type": "string", 
            "description": "Stock ticker symbol",
            "current_value": "AAPL",
            "default_value": "AAPL"
          }
        ],
        "metadata": {
          "lastUpdated": 1728994470324,
          "source": "Financial Modelling Prep"
        }
      }
    ],
    "secondary": [],
    "extra": []
  }
}
```


You then parse the response, process the data, and format the messages to your LLM. Let's assume that the LLM then generates a string of tokens to answer the user's query. These are then streamed back to the user using the `copilotMessageChunk` SSE:

```
event: copilotMessageChunk
data: {"delta":"The"}

event: copilotMessageChunk
data: {"delta":" current"}

event: copilotMessageChunk
data: {"delta":" stock"}

event: copilotMessageChunk
data: {"delta":" price"}

event: copilotMessageChunk
data: {"delta":" of"}

event: copilotMessageChunk
data: {"delta":" Apple"}

event: copilotMessageChunk
data: {"delta":" Inc."}

event: copilotMessageChunk
data: {"delta":" (AAPL)"}

event: copilotMessageChunk
data: {"delta":" is"}

event: copilotMessageChunk
data: {"delta":" $150.75."}
```


## Configuring your custom copilot for the OpenBB Workspace (`copilots.json`)

To integrate your custom copilot with the OpenBB Workspace, you need to configure and
serve a `copilots.json` file. This file defines how your custom copilot connects
with the frontend, including which features are supported by your custom
copilot, and where requests  should be sent.  

Here is an example copilots.json configuration:

```python
{
  "example_copilot": { # <-- the ID of your copilot
    "name": "Mistral Example Co. Copilot", # <-- the display name of your copilot
    "description": "AI-powered financial copilot that uses Mistral Large as its LLM.", # <-- a short description of your copilot
    "image": "<url>", # <-- a URL to an image icon for your copilot
    "endpoints": {
      "query": "<url>" # <-- the URL that the OpenBB Workspace will send requests to. For example, "http://localhost:7777/v1/query"
    },
    "hasStreaming": true,  // Deprecated: use features.streaming instead
    "hasDocuments": false,  // Deprecated: not applicable anymore
    "hasFunctionCalling": false,  // Deprecated: use features.widget-dashboard-select, widget-dashboard-search or widget-global-search instead
    "features": {
        "streaming": true, // <-- whether your copilot supports streaming responses via SSEs. This must always be true.
        "file-upload": false,  // <-- whether the copilot supports uploading files
        "widget-dashboard-select": false,  // <-- specifies if the copilot supports manually selected widgets. These will be sent in `widgets.primary` in the payload.
        "widget-dashboard-search": false,  // <-- specifies if the copilot supports sending dashboard widgets. These will be sent in `widgets.secondary` in the payload.
        "widget-global-search": false  // <-- specifies if the copilot supports searching through all available widgets These will be sent in `widgets.extra` in the payload.
    }
  }
}
```

Your `copilots.json` file must be served at `<your-host>/copilots.json`, for example, `http://localhost:7777/copilots.json`.

## Custom Agent Protocol Reference

**Note: This section is for advanced or curious users who want to understand the
details of the custom agent protocol, or who do
not wish to use the included `agent` micro framework, and want to manualy handle
their own SSEs and requests.**

**We recommend starting with the `agent` micro framework in the [Usage](#usage)
section above, and only reference this section if you need to understand the
details of the custom agent protocol.**

OpenBB Workspace uses a specific protocol / API contract to communicate with your custom copilot.

### The API is stateless

The most important concept to understand is that the API is
_stateless_.  This means that every request from the OpenBB Workspace to your copilot
will include all previous messages (such as AI completions, human messages,
function calls, and function call results) in the request payload.

This means it is not necessary for your custom copilot to maintain any state
between requests. It should simply use the request payload to generate a response.

The OpenBB Workspace is solely responsible for maintaining the conversation state, and will
append the responses to the `messages` array in the request payload. 
