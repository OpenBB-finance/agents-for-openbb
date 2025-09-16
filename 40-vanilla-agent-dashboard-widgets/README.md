# 40 - Vanilla Agent Dashboard Widgets

This example demonstrates a simple agent that lists all widgets available on the current dashboard, showing their metadata and parameters in a structured format.

Key behaviors:

- Exposes `agents.json` with `widget-dashboard-search` enabled so the Workspace sends dashboard widget metadata (as `widgets.secondary`, etc.).
- Lists all widgets from both explicit context (primary) and dashboard context (secondary).
- Shows detailed widget information including:
  - Widget metadata (name, description, ID, category, UUID)
  - Parameters table with type, default, current values, options, and descriptions
- Organizes dashboard widgets by tabs for better visibility.
- Does NOT automatically fetch widget data - focuses purely on widget discovery and listing.

## Run locally

- Install dependencies at repo root: `poetry install --no-root`
- Start the API from this directory:

  `poetry run uvicorn vanilla_agent_dashboard_widgets.main:app --port 7777 --reload`

## Test

- From this directory: `poetry run pytest tests`