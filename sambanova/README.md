# SambaNova Copilot
This example provides a custom copilot that uses the [Sambanova API](https://sambanova.ai/) that
connects to a Llama 3.1 model 70B model.

## Overview
This implementation utilizes a FastAPI application to serve as the backend for
the copilot, and uses the OpenAI Python client to interact with the Sambanova
API (which is compatible with the OpenAI API).

You're not limited to our setup! If you have preferences for different APIs or
LLM frameworks, feel free to adapt this implementation. The key is to adhere to
the schema defined by the `/query` endpoint and the specifications in
`copilot.json`.

This repository is a starting point. It's designed for you to experiment,
modify, and extend. You can build copilots with various capabilities, like RAG
(Retrieval-Augmented Generation), function calling, and more, all hosted on your
backend.

## Getting started

Here's how to get your copilot up and running:

### Prerequisites

Ensure you have poetry, a tool for dependency management and packaging in
Python, as well as your SambaNova API key.

### Installation and Running

1. Clone this repository to your local machine.
2. Set the SambaNova API key as an environment variable in your .bashrc or .zshrc file:

``` sh
# in .zshrc or .bashrc
export SAMBANOVA_API_KEY=<your-api-key>
```

3. Install the necessary dependencies:

``` sh
poetry install --no-root
```

4.Start the API server:

``` sh
poetry run uvicorn sambanova.main:app --port 7777 --reload
```

This command runs the FastAPI application, making it accessible on your network.

### Testing the Copilot
The example copilot has a small, basic test suite to ensure it's
working correctly. As you develop your copilot, you are highly encouraged to
expand these tests.

You can run the tests with:

``` sh
pytest tests
```

### Accessing the Documentation

Once the API server is running, you can view the documentation and interact with
the API by visiting: http://localhost:7777/docs
