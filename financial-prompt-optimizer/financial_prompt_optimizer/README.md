# Example agent for financial prompt optimization

This is a minimal example agent, powered by OpenAI, that optimizes a user's
financial prompt for clarity, specificity, and actionability. It has no widget
integration and focuses solely on improving prompts.

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

4. Start the API server:

``` sh
cd financial-prompt-optimizer
poetry run uvicorn financial_prompt_optimizer.main:app --port 7777 --reload
```

This command runs the FastAPI application, making it accessible on your network.

### Accessing the Documentation

Once the API server is running, you can view the documentation and interact with
the API by visiting: http://localhost:7777/docs

### Using with OpenBB Workspace (Optional)

- The agent descriptor is available at: http://localhost:7777/agents.json
- Features are set to:
  - `streaming: true`
  - `widget-dashboard-select: false`
  - `widget-dashboard-search: false`

### Expected Behavior

When you send a user message, the agent streams a single answer containing two
sections:

- `Optimized Prompt: <improved prompt>`
