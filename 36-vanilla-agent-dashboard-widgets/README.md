# 36 - Vanilla Agent Dashboard Widgets

This example demonstrates a simple agent that receives the full list of widgets present on the current dashboard and passes that context directly to the LLM. The model decides which widget(s) to use and issues a function call accordingly.

Key behaviors:

- Exposes `agents.json` with `widget-dashboard-search` enabled so the Workspace sends dashboard widget metadata (as `widgets.secondary`, etc.).
- If the user has selected primary widgets, the agent immediately issues a function call to fetch data for them.
- Otherwise, the agent does not select widgets heuristically. It appends the full dashboard widget list to the prompt and instructs the LLM to respond with a `get_widget_data` JSON function call when needed.
- Falls back to a plain LLM reply if no data is needed.

## Run locally

- Install dependencies at repo root: `poetry install --no-root`
- Start the API from this directory:

  `poetry run uvicorn vanilla_agent_dashboard_widgets.main:app --port 7777 --reload`

## Test

- From this directory: `poetry run pytest tests`
