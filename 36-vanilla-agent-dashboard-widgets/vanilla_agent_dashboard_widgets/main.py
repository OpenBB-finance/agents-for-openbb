from __future__ import annotations

from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai import WidgetRequest, get_widget_data, message_chunk
from openbb_ai.models import (
    MessageChunkSSE,
    QueryRequest,
    FunctionCallSSE,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)


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
    """Stream either a function call or an AI answer."""
    if not request.messages:
        return JSONResponse(
            status_code=400, content={"detail": "messages list cannot be empty"}
        )  # type: ignore[return-value]

    last_message = request.messages[-1]

    # If the last message is a tool result, terminate cleanly to avoid loops
    if last_message.role == "tool":
        async def done() -> AsyncGenerator[StatusUpdateSSE, None]:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO", message="llm_complete", hidden=True
                )
            )

        return EventSourceResponse(
            content=(event.model_dump(exclude_none=True) async for event in done()),
            media_type="text/event-stream",
        )

    # Widgets added to explicit context
    primary = (request.widgets.primary if request.widgets else None) or []

    # Widgets in dashboard
    secondary = (request.widgets.secondary if request.widgets else None) or []

    # File uploads or artifacts
    extra = (request.widgets.extra if request.widgets else None) or []

    # Try to use workspace tabs organization
    tabs = None
    active_tab_id = None
    if request.workspace_state and request.workspace_state.current_dashboard_info:
        tabs = request.workspace_state.current_dashboard_info.tabs or None
        active_tab_id = request.workspace_state.current_dashboard_info.current_tab_id
        

    sections: list[str] = []
    first_from_tabs_uuid: str | None = None
    # This checks if we are in a dashboard (otherwise we may be in Apps or Widgets Library or other)
    if tabs:
        if active_tab_id:
            sections.append(f"Active tab: {active_tab_id}\n")
        else:
            sections.append("No tab detected\n")

        for t in tabs:
            sections.append(f"[Tab: {t.tab_id}]\n")
            if t.widgets:
                # record first widget uuid if we haven't yet
                if first_from_tabs_uuid is None and len(t.widgets) > 0:
                    first_from_tabs_uuid = t.widgets[0].widget_uuid
                for w in t.widgets:
                    sections.append(f"- {w.name}")
            sections.append("")

    list_text = (
        "Widgets on your current dashboard:\n\n" + "\n".join(sections)
        if sections
        else "I couldn't detect any widgets in your dashboard payload."
    )

    async def events() -> AsyncGenerator[MessageChunkSSE | FunctionCallSSE | StatusUpdateSSE, None]:
        # First, emit the list message
        yield message_chunk(list_text)
        # Emit an empty chunk to mark end of the textual message
        yield message_chunk("")

        # Then, retrieve data for the first widget found
        first = None
        # Prefer first widget in the active tab (if tabs are provided)
        if first_from_tabs_uuid:
            # Look up the full widget object by UUID across all available buckets
            all_widgets = list(primary) + list(secondary) + list(extra)
            for w in all_widgets:
                if str(getattr(w, "uuid", "")) == str(first_from_tabs_uuid):
                    first = w
                    break
        # Fallback order if tabs not available or not found
        if first is None:
            if primary:
                first = primary[0]
            elif secondary:
                first = secondary[0]
            elif extra:
                first = extra[0]

        if first is not None:
            input_args = {
                p.name: (
                    getattr(p, "current_value", None)
                    if getattr(p, "current_value", None) is not None
                    else getattr(p, "default_value", None)
                )
                for p in first.params
            }
            yield get_widget_data([WidgetRequest(widget=first, input_arguments=input_args)])

        # Finally, emit a completion signal to terminate the stream
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(eventType="INFO", message="llm_complete", hidden=True)
        )

    async def serialize_events():
        async for event in events():
            yield event.model_dump(exclude_none=True)

    return EventSourceResponse(
        content=serialize_events(), media_type="text/event-stream"
    )
