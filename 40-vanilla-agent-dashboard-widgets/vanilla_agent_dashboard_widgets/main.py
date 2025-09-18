from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai.models import QueryRequest
from openbb_ai import message_chunk


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/agents.json")
def get_copilot_description():
    """Copilot descriptor for OpenBB Workspace."""
    return JSONResponse(
        content={
            "vanilla_agent_dashboard_widgets": {
                "name": "Vanilla Agent Dashboard Widgets",
                "description": "Lists dashboard widgets and retrieves data from the first widget on the dashboard (if any).",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {"query": "http://localhost:7777/v1/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,
                    "widget-dashboard-search": True,
                },
            }
        }
    )


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot."""

    # Get all widgets from dashboard (used in multiple places)
    all_widgets = []
    if request.widgets:
        # These are the widgets that are available in the explicit context tab
        if request.widgets.primary:
            all_widgets.extend(request.widgets.primary)
        # These are the widgets that are available in the dashboard
        if request.widgets.secondary:
            all_widgets.extend(request.widgets.secondary)

    # Helper function to format widget details
    def format_widget(w):
        msg = f"### {w.name or w.widget_id or 'Unnamed Widget'}\n\n"
        msg += "| Field | Value |\n"
        msg += "|-------|-------|\n"
        msg += f"| Name | {w.name or 'N/A'} |\n"
        # Clean up description - replace newlines with spaces
        description = str(w.description or "N/A")
        description = description.replace("\n", " ").replace("  ", " ").strip()
        if len(description) > 150:
            description = description[:147] + "..."
        msg += f"| Description | {description} |\n"
        msg += f"| ID | {w.widget_id or 'N/A'} |\n"
        msg += f"| Category | {getattr(w, 'category', 'N/A') or 'N/A'} |\n"
        msg += f"| UUID | {getattr(w, 'uuid', 'N/A') or 'N/A'} |\n\n"

        # Parameters table
        if w.params:
            msg += f"#### {w.name or w.widget_id or 'Unnamed Widget'} Parameters\n\n"
            msg += "| Parameter | Type | Default | Current | Options | Description |\n"
            msg += "|-----------|------|---------|---------|---------|-------------|\n"
            for p in w.params:
                param_type = str(getattr(p, "type", "N/A") or "N/A")
                default_val = str(getattr(p, "default_value", "N/A") or "N/A")
                current_val = str(getattr(p, "current_value", "N/A") or "N/A")
                # Clean up description - replace newlines with spaces
                param_desc = str(getattr(p, "description", "N/A") or "N/A")
                param_desc = param_desc.replace("\n", " ").replace("  ", " ").strip()
                # Truncate long descriptions
                if len(param_desc) > 100:
                    param_desc = param_desc[:97] + "..."
                # Handle possible options
                options = getattr(p, "options", None)
                options_str = ", ".join(str(o) for o in options) if options else ""
                # Truncate long options list
                if len(options_str) > 50:
                    options_str = options_str[:47] + "..."
                msg += f"| {p.name} | {param_type} | {default_val} | {current_val} | {options_str} | {param_desc} |\n"
            msg += "\n"
        return msg

    widget_list_msg = ""

    # Explicit Context (Primary) - widgets explicitly selected
    if request.widgets and request.widgets.primary:
        widget_list_msg += "# Explicit Context (Primary)\n\n"
        for w in request.widgets.primary:
            widget_list_msg += format_widget(w)
        widget_list_msg += "\n"

    # Dashboard Context (Secondary) - widgets from current dashboard
    # Although all widgets are already in all_widgets, we want to show them grouped by tab
    # and for that we need to use the workspace_state.current_dashboard_info
    if request.workspace_state and request.workspace_state.current_dashboard_info:
        tabs = request.workspace_state.current_dashboard_info.tabs
        active_tab = request.workspace_state.current_dashboard_info.current_tab_id

        if tabs:
            widget_list_msg += "# Dashboard Context (Secondary)\n\n"
            if active_tab:
                widget_list_msg += f"Active tab: {active_tab}\n\n"

            for tab in tabs:
                widget_list_msg += f"## Tab: {tab.tab_id}\n\n"
                if tab.widgets:
                    for widget in tab.widgets:
                        # Look up full widget details from all_widgets
                        full_widget = None
                        for w in all_widgets:
                            if str(getattr(w, "uuid", "")) == str(widget.widget_uuid):
                                full_widget = w
                                break

                        if full_widget:
                            widget_list_msg += format_widget(full_widget)
                        else:
                            # Widget not found in all_widgets, show basic info
                            widget_list_msg += f"### {widget.name}\n\n"
                            widget_list_msg += "| Field | Value |\n"
                            widget_list_msg += "|-------|-------|\n"
                            widget_list_msg += f"| Name | {widget.name or 'N/A'} |\n"
                            widget_list_msg += (
                                f"| UUID | {widget.widget_uuid or 'N/A'} |\n\n"
                            )
                else:
                    widget_list_msg += "  (no widgets)\n"
                widget_list_msg += "\n"
    elif all_widgets and not (request.widgets and request.widgets.primary):
        # Fallback if no primary widgets shown above
        widget_list_msg += "**Available widgets**\n\n"
        for w in all_widgets:
            widget_list_msg += format_widget(w)
        widget_list_msg += "\n"

    # Always show widget list on human messages
    if request.messages[-1].role == "human":
        # Stream widget list only
        async def show_widgets():
            if widget_list_msg:
                yield message_chunk(widget_list_msg.rstrip()).model_dump()
            else:
                yield message_chunk("No widgets found on your dashboard.").model_dump()

        # Return early with widget info only
        return EventSourceResponse(
            content=show_widgets(),
            media_type="text/event-stream",
        )

    # If we reach here, no specific handler matched - return empty response
    async def empty_response():
        yield message_chunk("No action taken.").model_dump()

    return EventSourceResponse(content=empty_response(), media_type="text/event-stream")
