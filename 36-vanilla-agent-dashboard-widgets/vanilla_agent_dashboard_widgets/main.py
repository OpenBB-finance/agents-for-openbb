from __future__ import annotations

from typing import AsyncGenerator

import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai import WidgetRequest, get_widget_data, message_chunk
from openbb_ai.models import (
    MessageChunkSSE,
    QueryRequest,
    Widget,
)

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
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
                "description": "Recognizes relevant widgets from the current dashboard and fetches data.",
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


def _select_relevant_dashboard_widgets(query: str, candidates: list[Widget]) -> list[Widget]:
    """Simple heuristic to select relevant dashboard widgets based on the query.

    - Matches on widget name or id appearing in the query
    - Falls back to basic keyword mapping using the widget description
    - Returns at most 1–2 widgets to keep results focused
    """
    if not candidates:
        return []

    q = query.lower()

    # First pass: exact-ish name/id matches
    exact_matches: list[Widget] = []
    for w in candidates:
        name = (w.name or "").lower()
        wid = (w.widget_id or "").lower()
        if not name and not wid:
            continue
        if (name and name in q) or (wid and wid in q):
            exact_matches.append(w)

    if exact_matches:
        return exact_matches[:2]

    # Second pass: keyword mapping using description/name
    scored: list[tuple[int, Widget]] = []
    keywords_map = {
        "price": {"price", "stock price", "quote"},
        "news": {"news", "headline", "article"},
        "yield": {"yield", "curve", "rates"},
        "volume": {"volume", "liquidity"},
        "balance": {"balance", "sheet"},
        "income": {"income", "revenue", "earnings"},
        "transcript": {"transcript", "call"},
        "historical": {"historical", "history"},
    }

    for w in candidates:
        text = f"{w.name or ''} {w.description or ''} {w.widget_id or ''}".lower()
        score = 0
        for _, kws in keywords_map.items():
            if any(kw in q and kw in text for kw in kws):
                score += 1
        if score > 0:
            scored.append((score, w))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in scored[:2]]


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot.

    Behavior:
    - If primary widgets are present on a human message, immediately request their data.
    - Else, try to infer relevant dashboard widgets from `secondary` and request data.
    - Else, stream a plain LLM response.
    """
    if not request.messages:
        return JSONResponse(status_code=400, content={"detail": "messages list cannot be empty"})  # type: ignore[return-value]

    last_message = request.messages[-1]

    # 1) If the user is asking to list dashboard widgets, answer directly
    def _is_widgets_list_request(text: str | None) -> bool:
        if not text:
            return False
        q = text.lower()
        if "dashboard" in q and "widget" in q:
            triggers = ["what", "which", "list", "show", "available"]
            return any(t in q for t in triggers)
        return False

    if last_message.role == "human" and _is_widgets_list_request(last_message.content):
        primary = (request.widgets.primary if request.widgets else None) or []
        secondary = (request.widgets.secondary if request.widgets else None) or []
        # Prefer the full dashboard list (secondary). Include selected if any.
        secondary_names = [w.name or w.widget_id or "Unnamed Widget" for w in secondary]
        primary_names = [w.name or w.widget_id or "Unnamed Widget" for w in primary]

        lines: list[str] = []
        if secondary_names:
            lines.append("Widgets on your current dashboard:")
            for nm in secondary_names:
                lines.append(f"- {nm}")
        if primary_names:
            lines.append("")
            lines.append("Currently selected widgets:")
            for nm in primary_names:
                lines.append(f"- {nm}")

        text = "\n".join(lines) if lines else "I couldn't detect any widgets in your dashboard payload."

        async def list_widgets_events():
            # Prime UI so message container exists
            yield message_chunk(" ")
            # Stream the list in a couple of chunks for UX
            for part in ["\n".join(lines[:1]), "\n".join(lines[1:])]:
                if part:
                    yield message_chunk(part)

        async def serialize_list_events():
            async for event in list_widgets_events():
                yield event.model_dump(exclude_none=True)

        return EventSourceResponse(content=serialize_list_events(), media_type="text/event-stream")

    # 2) If user explicitly asks to fetch data from dashboard widgets, do it
    def _wants_dashboard_data_fetch(text: str | None) -> bool:
        if not text:
            return False
        q = text.lower()
        return (
            ("dashboard" in q or "widget" in q)
            and any(t in q for t in ["fetch", "get", "pull", "use", "show"])
            and any(t in q for t in ["data", "info", "information", "results"])
        )

    if last_message.role == "human" and request.widgets and request.widgets.secondary and _wants_dashboard_data_fetch(last_message.content):
        # Select relevant widgets if possible; otherwise take top 2 to avoid over-fetching
        selected = _select_relevant_dashboard_widgets(last_message.content or "", request.widgets.secondary)  # type: ignore[arg-type]
        if not selected:
            selected = list(request.widgets.secondary)[:2]

        widget_requests = [
            WidgetRequest(
                widget=w,
                input_arguments={param.name: param.current_value for param in w.params},
            )
            for w in selected
        ]

        async def retrieve_widget_data_dashboard():
            yield get_widget_data(widget_requests)

        async def serialize_widget_events_dashboard():
            async for event in retrieve_widget_data_dashboard():
                yield event.model_dump(exclude_none=True)

        return EventSourceResponse(content=serialize_widget_events_dashboard(), media_type="text/event-stream")

    # 3) Auto-fetch data for explicitly selected primary widgets
    if last_message.role == "human" and request.widgets and request.widgets.primary:
        widget_requests: list[WidgetRequest] = []
        for widget in request.widgets.primary:
            widget_requests.append(
                WidgetRequest(
                    widget=widget,
                    input_arguments={param.name: param.current_value for param in widget.params},
                )
            )

        async def retrieve_widget_data_primary():
            yield get_widget_data(widget_requests)

        async def serialize_widget_events_primary():
            async for event in retrieve_widget_data_primary():
                yield event.model_dump(exclude_none=True)

        return EventSourceResponse(content=serialize_widget_events_primary(), media_type="text/event-stream")

    # 4) Try to recognize relevant widgets from dashboard (secondary) and fetch data
    if last_message.role == "human" and request.widgets and request.widgets.secondary:
        candidates = request.widgets.secondary
        selected = _select_relevant_dashboard_widgets(last_message.content or "", candidates)  # type: ignore[arg-type]
        if not selected and len(candidates) == 1:
            selected = list(candidates)
        if selected:
            widget_requests = [
                WidgetRequest(
                    widget=w,
                    input_arguments={param.name: param.current_value for param in w.params},
                )
                for w in selected
            ]

            async def retrieve_widget_data_secondary():
                yield get_widget_data(widget_requests)

            async def serialize_widget_events_secondary():
                async for event in retrieve_widget_data_secondary():
                    yield event.model_dump(exclude_none=True)

            return EventSourceResponse(content=serialize_widget_events_secondary(), media_type="text/event-stream")

    # 5) Fallback to plain LLM response
    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=(
                "You are a helpful financial assistant named 'Vanilla Agent'. "
                "When possible, use data from widgets if the user has selected any."
            ),
        )
    ]

    context_str = ""
    for index, message in enumerate(request.messages):
        if message.role == "human":
            openai_messages.append(
                ChatCompletionUserMessageParam(role="user", content=message.content)
            )
        elif message.role == "ai":
            if isinstance(message.content, str):
                openai_messages.append(
                    ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
                )
        elif message.role == "tool":
            # Only use the most recent tool result to avoid context bloat
            if index == len(request.messages) - 1:
                context_str += "Use the following data to answer the question:\n\n"
                result_str = "--- Data ---\n"
                try:
                    for result in message.data:
                        for item in result.items:
                            # Prefer 'content' if present; otherwise include URL reference
                            if getattr(item, "content", None):
                                result_str += f"{item.content}\n"
                            elif getattr(item, "url", None):
                                filename = getattr(getattr(item, "data_format", None), "filename", None)
                                if filename:
                                    result_str += f"File available: {filename} ({item.url})\n"
                                else:
                                    result_str += f"File available at: {item.url}\n"
                            result_str += "------\n"
                except Exception:
                    # If schema differs, safely ignore rather than failing
                    pass
                context_str += result_str

    # If we have context from the latest tool message, append it to the last user/assistant message
    if context_str and len(openai_messages) > 1:
        try:
            openai_messages[-1]["content"] += "\n\n" + context_str  # type: ignore[index]
        except Exception:
            # If last message isn't content-bearing, attach to system as fallback
            openai_messages[0]["content"] += "\n\n" + context_str  # type: ignore[index]

    async def execution_loop() -> AsyncGenerator[MessageChunkSSE, None]:
        client = openai.AsyncOpenAI()

        # Prime UI to create an empty message
        yield message_chunk(" ")

        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
        )

        async for event in stream:
            if chunk := event.choices[0].delta.content:
                yield message_chunk(chunk)

    async def serialize_events():
        async for event in execution_loop():
            yield event.model_dump(exclude_none=True)

    return EventSourceResponse(content=serialize_events(), media_type="text/event-stream")
