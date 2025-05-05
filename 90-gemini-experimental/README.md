# Google Gemini Agent with Remote Function Calling
**Note: This is an alpha experimental agent that is under active development.**

This is an experimental example agent that uses Google's Gemini 2.0 Flash LLM.
It supports remote function calling to retrieve data from widgets on the OpenBB
Workspace.

We make use of the reasoning steps functionality of OpenBB to return the
reasoning to the frontend. To see how this functionality works (and how to use
it yourself), take a look at [this example agent](https://github.com/OpenBB-finance/copilot-for-openbb/tree/main/03-simple-copilot-reasoning-steps).

## Architecture

```sh
┌─────────────────────┐                ┌─────────────────────┐
│                     │                │                     │
│   OpenBB Workspace  │ ───────────>   │   Simple Copilot    │
│      (Frontend)     │     HTTP       │      (Backend)      │
│                     │    Request     │                     │
│                     │                │                     │
│                     │   <───────────-│                     │
│                     │      SSE       │                     │
└─────────────────────┘                └─────────────────────┘
```

The architecture consists of two main components:

1. **OpenBB Workspace (Frontend)**: The user interface where queries are entered
2. **Simple Copilot (Backend)**: Powered by Gemini, handles the processing of
   queries and returns answers

The frontend communicates with the backend via HTTP requests to the `/query`
endpoint as defined in the copilot.json schema.

## Overview

This implementation utilizes a FastAPI application to serve as the backend for
the copilot. The core functionality is powered by the official OpenAI Python
SDK.

You're not limited to our setup! If you have preferences for different APIs or
LLM frameworks, feel free to adapt this implementation. The key is to adhere to
the schema defined by the `/query` endpoint and the specifications in
`copilot.json`.

## Getting started

Here's how to get your copilot up and running:

### Prerequisites

Ensure you have poetry, a tool for dependency management and packaging in
Python, as well as your [Google AI Studio API key](https://aistudio.google.com/app/apikey).

### Installation and Running

1. Clone this repository to your local machine.

2. Set the Gemini API key as an environment variable in your .bashrc or .zshrc file:

    ``` sh
    # in .zshrc or .bashrc
    export GEMINI_API_KEY=<your-api-key>
    ```

3. Install the necessary dependencies:

``` sh
poetry install --no-root
```

4.Start the API server:

``` sh
cd 90-gemini-experimental
poetry run uvicorn gemini_experimental_agent.main:app --port 7777 --reload
```

This command runs the FastAPI application, making it accessible on your network.

### Testing the Copilot

The example copilot has a small, basic test suite to ensure it's
working correctly. As you develop your copilot, you are highly encouraged to
expand these tests.

You can run the tests with:

```sh
pytest tests
```

### Accessing the Documentation

Once the API server is running, you can view the documentation and interact with
the API by visiting: http://localhost:7777/docs
