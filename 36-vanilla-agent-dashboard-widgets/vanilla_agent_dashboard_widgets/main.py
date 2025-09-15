from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai.models import QueryRequest
from openbb_ai import get_widget_data, WidgetRequest, message_chunk


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
        if request.widgets.primary:
            all_widgets.extend(request.widgets.primary)
        if request.widgets.secondary:
            all_widgets.extend(request.widgets.secondary)
        if request.widgets.extra:
            all_widgets.extend(request.widgets.extra)
    
    # Build widget list string (used in multiple places)
    widget_list_msg = ""
    if request.workspace_state and request.workspace_state.current_dashboard_info:
        tabs = request.workspace_state.current_dashboard_info.tabs
        active_tab = request.workspace_state.current_dashboard_info.current_tab_id
        
        if tabs:
            widget_list_msg = "Widgets on your current dashboard:\n\n"
            if active_tab:
                widget_list_msg += f"Active tab: {active_tab}\n\n"
            
            for tab in tabs:
                widget_list_msg += f"[Tab: {tab.tab_id}]\n"
                if tab.widgets:
                    for widget in tab.widgets:
                        widget_list_msg += f"  - {widget.name}\n"
                else:
                    widget_list_msg += "  (no widgets)\n"
                widget_list_msg += "\n"
    elif all_widgets:
        widget_list_msg = "Available widgets:\n\n"
        for w in all_widgets:
            widget_list_msg += f"  - {w.name or w.widget_id or 'Unnamed Widget'}\n"
        widget_list_msg += "\n"

    # Always show widget list and fetch last widget data on human messages
    if request.messages[-1].role == "human":
        
        # Stream widget list and then fetch last widget data
        async def show_widgets_and_fetch():
            # Show the widget list (first message completes here)
            if widget_list_msg:
                yield message_chunk(widget_list_msg.rstrip()).model_dump()
            
            # Then fetch data from the last widget if available
            if all_widgets:
                last_widget = all_widgets[-1]
                widget_requests = [
                    WidgetRequest(
                        widget=last_widget,
                        input_arguments={
                            param.name: param.current_value for param in last_widget.params
                        },
                    )
                ]
                yield get_widget_data(widget_requests).model_dump()
            elif not widget_list_msg:
                yield message_chunk("No widgets found on your dashboard.").model_dump()
        
        # Return early with widget info and data fetch
        return EventSourceResponse(
            content=show_widgets_and_fetch(),
            media_type="text/event-stream",
        )

    # Check if we just received tool data - if so, show a sample and continue conversation
    if request.messages and request.messages[-1].role == "tool":
        # Extract widget name and data
        widget_name = "Unknown Widget"
        if all_widgets:
            last_widget = all_widgets[-1]
            widget_name = last_widget.name or last_widget.widget_id or 'Unnamed'
        
        # Extract data content
        data_content = ""
        for result in request.messages[-1].data:
            for item in result.items:
                data_content = item.content
                break
            if data_content:
                break
        
        if data_content:
            sample = data_content[:500] + "..." if len(data_content) > 500 else data_content
            async def show_data_sample():
                yield message_chunk(f"Fetching sample data from last widget: {widget_name}\n\nSample of widget data:\n```\n{sample}\n```").model_dump()
            
            return EventSourceResponse(
                content=show_data_sample(),
                media_type="text/event-stream",
            )
    
    # If we reach here, no specific handler matched - return empty response
    async def empty_response():
        yield message_chunk("No action taken.").model_dump()
    
    return EventSourceResponse(
        content=empty_response(),
        media_type="text/event-stream"
    )
