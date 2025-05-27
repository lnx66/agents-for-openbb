# Vanilla Agent with Reasoning Steps

This is an example agent, powered by OpenAI, that can perform question answering
and demonstrates how to display reasoning steps (sometimes referred to as
"status updates") to the OpenBB Workspace.

## Understanding reasoning steps
OpenBB Workspace allows you to return "reasoning steps" (sometimes referred to
as "thought steps" or even "status updates") from your custom agent to the
front-end, while the agent is answering a query. This is often useful for
providing updates and extra information to the user, particularly for
long-running queries, or for complicated workflows.

Similar to text chunks that contain the streamed response from the LLM, these
reasoning steps are returned as SSEs (Server-Sent Events) to the front-end,
which then displays them.

## Overview

This implementation utilizes a FastAPI application to serve as the backend for
the agent. 

You're not limited to our setup! If you have preferences for different APIs or
LLM frameworks, feel free to adapt this implementation. The key is to adhere to
the schema defined by the `/query` endpoint and the specifications in
`copilot.json`.

## Getting started

Here's how to get your copilot up and running:

### Prerequisites

Ensure you have poetry, a tool for dependency management and packaging in
Python, as well as your OpenAI API key.

### Installation and Running

1. Clone this repository to your local machine.

2. Set the OpenAI API key as an environment variable in your .bashrc or .zshrc file:

    ``` sh
    # in .zshrc or .bashrc
    export OPENAI_API_KEY=<your-api-key>
    ```

3. Install the necessary dependencies:

``` sh
poetry install --no-root
```

4.Start the API server:

``` sh
cd 31-vanilla-agent-reasoning-steps
poetry run uvicorn vanilla_agent_reasoning_steps.main:app --port 7777 --reload
```

This command runs the FastAPI application, making it accessible on your network.

### Testing the Agent

The example agent has a small, basic test suite to ensure it's
working correctly. As you develop your agent, you are highly encouraged to
expand these tests.

You can run the tests with:

```sh
pytest tests
```

### Accessing the Documentation

Once the API server is running, you can view the documentation and interact with
the API by visiting: http://localhost:7777/docs
