# Agent that can handle PDF data

This is an example agent, powered by OpenAI, that can perform question answering
using data retrieved from widgets on OpenBB Workspace. It specifically demonstrates
how to handle PDF data. 

## Architecture

```sh
┌─────────────────────┐                ┌─────────────────────┐
│                     │                │                     │
│   OpenBB Workspace  │ ───────────>   │       Agent         │
│      (Frontend)     │     HTTP       │     (Backend)       │
│                     │    Request     │                     │
│                     │                │                     │
│                     │   <───────────-│                     │
│                     │      SSE       │                     │
│                     │(message chunks,│                     │
│                     │    citations,  │                     │
│                     │    artifacts,  │                     │
│                     │    etc.)       │                     │
└─────────────────────┘                └─────────────────────┘
```

The architecture consists of two main components:

1. **OpenBB Workspace (Frontend)**: The user interface where queries are entered
2. **Vanilla Agent (Backend)**: Powered by OpenAI, handles the processing of queries and returns answers

The frontend communicates with the backend via HTTP requests to the `/query`
endpoint as defined in the agents.json schema.

## Overview

This implementation utilizes a FastAPI application to serve as the backend for
the agent. 

You're not limited to our setup! If you have preferences for different APIs or
LLM frameworks, feel free to adapt this implementation. The key is to adhere to
the schema defined by the `/query` endpoint and the specifications in
`agents.json`.

## Getting started

Here's how to get your agent up and running:

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
cd 35-vanilla-agent-pdf
poetry run uvicorn vanilla_agent_pdf.main:app --port 7777 --reload
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
