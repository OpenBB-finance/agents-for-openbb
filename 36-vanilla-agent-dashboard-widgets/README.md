# 36 - Vanilla Agent Dashboard Widgets

This example demonstrates a simple agent that recognizes widgets present on the current dashboard and can automatically select relevant ones to fetch data.

Key behaviors:

- Exposes `agents.json` with `widget-dashboard-search` enabled so the Workspace sends along dashboard widget metadata.
- If the user has selected primary widgets, the agent immediately issues a function call to fetch data for them.
- If no primary widgets are selected, the agent attempts to recognize relevant widgets from the dashboard (provided via `secondary`) and issues a function call to fetch their data.
- Falls back to a plain LLM reply if nothing relevant is found.

## Run locally

- Install dependencies at repo root: `poetry install --no-root`
- Start the API from this directory:

  `poetry run uvicorn vanilla_agent_dashboard_widgets.main:app --port 7777 --reload`

## Test

- From this directory: `poetry run pytest tests`

