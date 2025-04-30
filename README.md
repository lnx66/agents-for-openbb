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

## Table of Contents

- [Introduction](#introduction)
- [Examples](#examples)
- [Usage](#usage)
  - [Basic conversational agent](#basic-conversational-agent)
  - [Function calling](#function-calling)
    - [Local function calling](#local-function-calling)
    - [Remote function calling](#remote-function-calling)
  - [Reasoning Steps / Status Updates](#reasoning-steps--status-updates)


## Introduction

To integrate a custom agent that you can interact with from the OpenBB Workspace,
you'll need to create a backend API that the OpenBB Workspace can make requests to.  


Your custom agent API will respond with Server-Sent Events
([SSEs](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)).

**Note: If you're looking to get started
quickly, we suggest running one of the example agents included as part of
this repository, and adding it as a custom copilot to the OpenBB Workspace (each
example copilot includes instructions on how to run them). Cloning and modifying
an example copilot is a great way get started building a custom agent.**

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

### Basic conversational agent

The most basic custom agent is capable only of chatting with the user.

First, let's set our OPENAI_API_KEY:

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

Now, let's create a basic FastAPI app that uses the `OpenBBAgent` to chat with the user:

```python
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
    openbb_agent = agent.OpenBBAgent(
        query_request=request,
        system_prompt="You are a helpful assistant that can answer questions.",
    )

    return EventSourceResponse(
        content=openbb_agent.run(),
        media_type="text/event-stream",
    )

```

And that's it! All you need to do now is run the FastAPI app:

```bash
uvicorn main:app --reload --port 7777
```

And add it to the OpenBB Workspace as a custom agent:

https://github.com/user-attachments/assets/bd8daae2-6ef4-473e-abbc-cbb92fdec098

You can now chat with your custom agent in the OpenBB Workspace.

### What is a custom OpenBB Workspace agent?

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
(whether they're running locally on your machine or remotely on on a web server).

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
    openbb_agent = agent.OpenBBAgent(
        query_request=request,
        system_prompt="You are a helpful assistant that can answer questions and retrieve stout beers.",
        functions=[get_random_stout_beers],  # 👈 add the function to the `functions` argument
    )

    return EventSourceResponse(
        content=openbb_agent.run(),
        media_type="text/event-stream",
    )

```

<img width="726" src="https://openbb-assets.s3.us-east-1.amazonaws.com/docs/custom_copilot/local_function_calling_example.png" alt="Local Function Calling Example">

A few important things to note:
- The function must be an `async` function.
- The function must be type-hinted.
- The function must yield a string as its output.

It is also recommended to:
- Use a descriptive function name (this is passed to the LLM).
- Use a docstring to describe the function and its parameters (this is passed to the LLM).

Now, when you query the agent, it will be able to call the `get_random_stout_beers` function.

#### Remote function calling

To allow our custom agent to retrieve data from widgets on a dashboard on OpenBB
Workspace, we must use remote function calling.

To do this, we must:

- Define a function that can be called by the LLM powering our custom agent, and decorate it with the `@remote_function_call` decorator.
- Update the system prompt to list all of the available widgets that the custom agent can retrieve data from.
- Enable remote function calling for the custom agent in the `copilots.json` endpoint (this is used by the OpenBB Workspace to know that remote function calling is supported by the custom agent).

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
    Widget,
    WidgetCollection,
)

SYSTEM_PROMPT_TEMPLATE = """\n
You are a helpful financial assistant.

You can use the following functions to help you answer the user's query:
- get_widget_data(widget_uuid: str) -> str: Get the data for a widget. You can use this function multiple times to get the data for multiple widgets.

{widgets_prompt}
"""


def _render_widget(widget: Widget) -> str:
    """Formats a widget's information into a LLM-friendly string representation."""
    widget_str = ""
    widget_str += (
        f"uuid: {widget.uuid} <-- use this to retrieve the data for the widget\n"
    )
    widget_str += f"name: {widget.name}\n"
    widget_str += f"description: {widget.description}\n"
    widget_str += "parameters:\n"
    for param in widget.params:
        widget_str += f"  {param.name}={param.current_value}\n"
    widget_str += "-------\n"
    return widget_str


def render_system_prompt(widget_collection: WidgetCollection | None = None) -> str:
    """Renders the system prompt for the custom agent, which includes a list of available widgets."""
    widgets_prompt = "# Available Widgets\n\n"
    # `primary` widgets are widgets that the user has manually selected
    # and added to the custom agent on OpenBB Workspace.
    widgets_prompt += "## Primary Widgets (prioritize using these widgets when answering the user's query):\n\n"
    for widget in widget_collection.primary if widget_collection else []:
        widgets_prompt += _render_widget(widget)

    # `secondary` widgets are widgets that are on the currently-active dashboard, but
    # have not been added to the custom agent explicitly by the user.
    widgets_prompt += "\n## Secondary Widgets (use these widgets if the user's query is not answered by the primary widgets):\n\n"
    for widget in widget_collection.secondary if widget_collection else []:
        widgets_prompt += _render_widget(widget)

    return SYSTEM_PROMPT_TEMPLATE.format(widgets_prompt=widgets_prompt)


async def handle_widget_data(data: list[DataContent]) -> str:
    """Formats the data retrieved from a widget into a LLM-friendly string representation."""
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

@app.get("/copilots.json")
def get_copilot_description():
    return JSONResponse(
        content={
            "simple_copilot": {
                "name": "Simple Copilot with OpenBB Function Calling",
                "description": "A simple copilot that can answer questions, execute OpenBB function calls, and return reasoning steps.",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {
                    "query": "http://localhost:7777/v1/query"
                },
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,  # 👈 We need to enable remote function calling for the custom agent
                    "widget-dashboard-search": True,  # 👈 We need to enable remote function calling for the custom agent
                },
            }
        }
    )

@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
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
- It is recommended to specify an `output_formatter` function that will be used to format the output of the data retrieved from the widget, so that it can be passed back to the LLM in a readable format.
- If an `output_formatter` is not specified, the
result is dumped as a string and returned to the LLM.
- The function must yield the result of the `get_remote_data` function, which must specify the `widget` and `input_arguments` to retrieve data for.

The `request` argument is the same `QueryRequest` object passed into the `query`
endpoint. It contains the current conversation's messages, any explicitly-added
context, information about widgets on the currently-active dashboard, and so on.
You can view the full schema either by looking at the `QueryRequest` model in
the
[common/models.py](https://github.com/OpenBB-finance/copilot-for-terminal-pro/blob/main/common/common/models.py)
file, or by inspecting the `QueryRequest` model in the Swagger UI of a custom
agent (at `<your-custom-agent-url>/docs`, eg. `http://localhost:7777/docs`).

#### Priority vs. background widgets
...

#### Global or "extra" widgets
...

### Reasoning steps / status updates

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

    # 👇 New
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
            # 👇 New
            yield agent.reasoning_step(
                event_type="ERROR",
                message="Failed to fetch beers.",
                details={"error": "Failed to fetch beers."},
            )
            yield "Failed to fetch beers."
            return

        # 👇 New
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

<img width="726" alt="reasoning steps example" src="https://github.com/user-attachments/assets/fd0494ad-ea80-41ff-8d30-b90c139cdeb2" />

Some things to note:
- The `yield reasoning_step(...)` must be called from within the function you've added to your custom agent.
- The reasoning step can have an `event_type` of `INFO`, `WARNING`, or `ERROR`.
- The reasoning step must specify a `message`, which will be displayed to the user in the OpenBB Workspace.
- The reasoning step can optionally include a `details` dictionary, which will be displayed to the user as a table in the OpenBB Workspace, if they expand the reasoning step.

Reasoning steps can be yielded from both local and remote functions.

### Citations
