# Vanilla Agent with Custom Features

This is a simple example agent that demonstrates how to detect and report which custom features are enabled or disabled in the OpenBB Workspace UI.

## What it does

This agent:
- Greets users with a friendly hello message
- Reports the status of custom features (Deep Research and Web Search)
- Shows boolean, text, and select option values based on the UI settings

## Features

The agent defines four custom features in its configuration:
- **Deep Research**: Allows the agent to perform deep research (default: disabled)
- **Web Search**: Allows the agent to search the web (default: enabled)
- **Model**: Selects the model value sent to the agent
- **Agent Name**: Sets the display name the agent uses in its response

Users can update these options in the OpenBB Workspace UI, and the agent will detect and report the current values from `workspace_options`.

## Getting started

### Prerequisites

Ensure you have poetry, a tool for dependency management and packaging in Python, as well as your OpenAI API key.

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

4. Start the API server:

``` sh
cd 37-vanilla-agent-custom-features
poetry run uvicorn vanilla_agent_custom_features.main:app --port 7777 --reload
```

This command runs the FastAPI application, making it accessible on your network.

### Testing the Agent

The example agent has a small, basic test suite to ensure it's working correctly. As you develop your agent, you are highly encouraged to expand these tests.

You can run the tests with:

```sh
pytest tests
```

### Accessing the Documentation

Once the API server is running, you can view the documentation and interact with the API by visiting: http://localhost:7777/docs
