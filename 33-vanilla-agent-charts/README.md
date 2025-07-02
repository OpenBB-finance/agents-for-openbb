# Example agent that can produce charts

This is an example agent, powered by OpenAI, that can perform question answering
and can return charts as part of its response. These charts are then rendered by
the OpenBB Workspace.

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
cd 33-vanilla-agent-charts
poetry run uvicorn vanilla_agent_charts.main:app --port 7777 --reload
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
