from __future__ import annotations

from typing import AsyncGenerator
import json
import csv
from io import StringIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai import WidgetRequest, get_widget_data, message_chunk, table
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

    # If the last message is a tool result, display a small preview and then terminate.
    if last_message.role == "tool":
        async def show_preview_and_done() -> AsyncGenerator[
            MessageChunkSSE | StatusUpdateSSE, None
        ]:
            # Announce which widget we retrieved
            widget_name = None
            try:
                ds_list = (last_message.input_arguments or {}).get("data_sources", [])
                target_uuid = None
                if ds_list:
                    target_uuid = ds_list[0].get("widget_uuid")
                    widget_name = ds_list[0].get("id")

                if target_uuid and request.widgets:
                    for col in [
                        request.widgets.primary or [],
                        request.widgets.secondary or [],
                        request.widgets.extra or [],
                    ]:
                        for w in col:
                            if str(getattr(w, "uuid", "")) == str(target_uuid):
                                widget_name = w.name or w.widget_id
                                raise StopIteration
            except StopIteration:
                pass
            except Exception:
                pass

            if widget_name:
                yield message_chunk(f"Retrieving data from: {widget_name}")
            # Best-effort parse of the first data content into a small table preview
            preview_rows: list[dict] | None = None
            try:
                if last_message.data:
                    # Find the first textual content
                    content_str = None
                    for result in last_message.data:
                        for item in getattr(result, "items", []) or []:
                            if getattr(item, "content", None):
                                content_str = item.content
                                break
                        if content_str:
                            break

                    if content_str:
                        # Try JSON
                        parsed = None
                        try:
                            parsed = json.loads(content_str)
                        except Exception:
                            parsed = None

                        def to_rows(obj) -> list[dict] | None:
                            if isinstance(obj, list):
                                if not obj:
                                    return []
                                # list of dicts
                                if isinstance(obj[0], dict):
                                    return obj
                                # list of lists with header in first row
                                if (
                                    isinstance(obj[0], list)
                                    and len(obj) > 1
                                    and all(isinstance(x, (str, int, float, bool, type(None))) for x in obj[0])
                                ):
                                    headers = [str(h) for h in obj[0]]
                                    return [
                                        {headers[i]: r[i] if i < len(r) else None}
                                        for r in obj[1:]
                                    ]
                            if isinstance(obj, dict):
                                # Common keys that may contain rows
                                for key in ("data", "rows", "records", "items", "result"):
                                    val = obj.get(key)
                                    rows = to_rows(val)
                                    if rows is not None:
                                        return rows
                            return None

                        rows = to_rows(parsed) if parsed is not None else None

                        if rows is None:
                            # Try CSV
                            try:
                                reader = csv.DictReader(StringIO(content_str))
                                rows = list(reader)
                            except Exception:
                                rows = None

                        if rows is None:
                            # As a last resort, show raw text snippet
                            yield message_chunk("Preview (raw):\n" + content_str[:500])
                        else:
                            # Limit to first 5 columns and 10 rows
                            if rows:
                                cols = list(rows[0].keys())[:5]
                                preview_rows = [
                                    {c: r.get(c) for c in cols} for r in rows[:10]
                                ]
            except Exception:
                preview_rows = None

            if preview_rows:
                yield table(
                    data=preview_rows,
                    name="Widget Preview",
                    description="First 10 rows × 5 columns",
                )

            # Completion signal
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO", message="llm_complete", hidden=True
                )
            )

        return EventSourceResponse(
            content=(
                event.model_dump(exclude_none=True)
                async for event in show_preview_and_done()
            ),
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
        # Prefer the first widget in the active tab if available
        if active_tab_id and any(t.tab_id == active_tab_id for t in tabs):
            active_tab = next(t for t in tabs if t.tab_id == active_tab_id)
            if active_tab.widgets and len(active_tab.widgets) > 0:
                first_from_tabs_uuid = active_tab.widgets[0].widget_uuid

        # If not found, fallback to the first widget of the first tab with widgets
        if first_from_tabs_uuid is None:
            for t in tabs:
                if t.widgets and len(t.widgets) > 0:
                    first_from_tabs_uuid = t.widgets[0].widget_uuid
                    break

        # Build visual listing grouped by tabs
        if active_tab_id:
            sections.append(f"Active tab: {active_tab_id}\n")
        else:
            sections.append("No tab detected\n")
        for t in tabs:
            sections.append(f"[Tab: {t.tab_id}]\n")
            if t.widgets:
                for w in t.widgets:
                    sections.append(f"- {w.name}")
            sections.append("")

    # Fallback listing by primary/secondary/extra if no tabs detected
    if not sections:
        if primary:
            sections.append("[Primary]\n")
            for w in primary:
                sections.append(f"- {w.name or w.widget_id or 'Unnamed Widget'}")
            sections.append("")
        if secondary:
            sections.append("[Secondary]\n")
            for w in secondary:
                sections.append(f"- {w.name or w.widget_id or 'Unnamed Widget'}")
            sections.append("")
        if extra:
            sections.append("[Extra]\n")
            for w in extra:
                sections.append(f"- {w.name or w.widget_id or 'Unnamed Widget'}")
            sections.append("")

    list_text = (
        "Widgets on your current dashboard:\n\n" + "\n".join(sections)
        if sections
        else "I couldn't detect any widgets in your dashboard payload."
    )

    async def events() -> AsyncGenerator[MessageChunkSSE | FunctionCallSSE | StatusUpdateSSE, None]:
        # First, emit the list message
        yield message_chunk(list_text)

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
            # IMPORTANT: Close immediately after function call; UI will send tool result next
            return

        # No widget found; emit completion to terminate the stream
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(eventType="INFO", message="llm_complete", hidden=True)
        )

    async def serialize_events():
        async for event in events():
            yield event.model_dump(exclude_none=True)

    return EventSourceResponse(
        content=serialize_events(), media_type="text/event-stream"
    )
