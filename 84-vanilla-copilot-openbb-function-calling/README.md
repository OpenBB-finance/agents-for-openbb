# Simple Copilot with Remote Function Calling

This is an example agent, powered by OpenAI, that can perform question
answering and remote function calling to the OpenBB Workspace to retrieve widget
data. It also demonstrates "reasoning steps" to the front end as it is executing.

It does not support handling PDF files or images, but is able to process
table widgets or text-based widgets.

## Understanding the role of the OpenBB Workspace for custom agents
OpenBB Workspace is the front-end browser-based application for accessing
and interacting with your data on OpenBB. This is the application that you interact
with when using OpenBB.

However, OpenBB Workspace also serves as the data layer for all agents that are
added to your OpenBB Workspace. If configured correctly, custom agents can use
the OpenBB Workspace to retrieve data from widgets as part of their execution
and query answering. 

Importantly, this exchange of data is performed directly between the browser and your 
custom agent backend, without any data leaving your local machine.

## Understanding function calling to OpenBB Workspace
To retrieve data from widgets on the OpenBB Workspace, your custom agent must
execute a **remote** function call, which gets interpreted by the OpenBB
Workspace. This is in contrast to **local** function calling, which we explored
in earlier custom agent examples.

Unlike local function calling, where the function is executed entirely
on the custom agent backend, a remote function call to the OpenBB Workspace is partially
executed on the OpenBB Workspace, and the results are sent back to the custom agent
backend.

It works as follows:

```
OpenBB Workspace                    Custom Agent
       │                                │
       │ 1. POST /query                 │
       │ {                              │
       │   messages: [...],             │
       │   widgets: {...}               │
       │ }                              │
       │───────────────────────────────>│
       │                                │
       │     2. Function Call SSE       │
       │<───────────────────────────────│
       │    (Connection then closed)    │
       │                                │
       │ 3. POST /query                 │
       │ {                              │
       │   messages: [                  │
       │     ...(original messages),    │
       │     function_call,             │
       │     function_call_result       │
       │   ],                           │
       │   widgets: {...}               │
       │ }                              │
       │───────────────────────────────>│
       │                                │
       │     4. SSEs (text chunks,      │
       │        reasoning steps, etc.)  │
       │<───────────────────────────────│
       │                                │
```

## Architecture

```
┌─────────────────────┐                ┌───────────────────────────────────────────┐
│                     │                │                                           │
│                     │                │            Simple Copilot                 │
│                     │                │              (Backend)                    │
│                     │ 1. HTTP POST   │                                           │
│   OpenBB Workspace  │ ───────────>   │  ┌─────────────┐    ┌─────────────────┐   │
│      (Frontend)     │   /query       │  │             │    │                 │   │
│                     │                │  │    LLM      │───>│    Function     │   │
│  ┌───────────────┐  │                │  │  Processing │    │     Call        │   │
│  │ Widget Data   │  │ <───────────   │  │             │<───│   Processing    │   │
│  │  Retrieval    │  │  2. Function   │  │             │    │                 │   │
│  └───────────────┘  │   Call SSE     │  └─────────────┘    └─────────────────┘   │
│         ^           │                │                                           │
│         │           │ 3. HTTP POST   │                                           │
│         └───────────│ ───────────>   │                                           │
│    Execute &        │   /query       │                                           │
│  Return Results     │                │                                           │
│                     │ <───────────   │                                           │
│                     │  4. SSE        │                                           │
│                     │  (text chunks, │                                           │
│                     │reasoning steps)│                                           │
└─────────────────────┘                └───────────────────────────────────────────┘
```

The architecture consists of two main components:

1. **OpenBB Workspace (Frontend)**: The user interface where queries are entered
2. **Simple Copilot (Backend)**: Powered by OpenAI, handles the processing of queries, executing internal function calls, and returns answers

The frontend communicates with the backend via HTTP requests to the `/query`
endpoint as defined in the copilot.json schema.

## Overview

This implementation utilizes a FastAPI application to serve as the backend for
the copilot. The core functionality is powered by `magentic`, a robust, minimal
framework for working with Large Language Models (LLMs).

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
cd 04-simple-copilot-openbb-function-calling
poetry run uvicorn simple_copilot_rfc.main:app --port 7777 --reload
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
