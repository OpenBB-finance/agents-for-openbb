from __future__ import annotations

from typing import AsyncGenerator
import json
import re

import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai import WidgetRequest, get_widget_data, message_chunk
from openbb_ai.models import MessageChunkSSE, QueryRequest, FunctionCallSSE

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
                "description": "Passes all dashboard widgets to the LLM; the model selects widgets and issues function calls when needed.",
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
            # Single, clean message without a leading empty chunk
            yield message_chunk(text)

        return EventSourceResponse(
            content=(event.model_dump(exclude_none=True) async for event in list_widgets_events()),
            media_type="text/event-stream",
        )

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

        return EventSourceResponse(
            content=(event.model_dump(exclude_none=True) async for event in retrieve_widget_data_primary()),
            media_type="text/event-stream",
        )

    # Fallback to plain LLM response
    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=(
                "You are a helpful financial assistant named 'Vanilla Agent'.\n"
                "You have access to a list of available dashboard widgets.\n"
                "When you need data, respond ONLY with a JSON object using this exact schema (no extra text):\n"
                "{\n"
                "  \"function\": \"get_widget_data\",\n"
                "  \"input_arguments\": {\n"
                "    \"data_sources\": [{\n"
                "      \"widget_uuid\": \"<uuid_from_list>\",\n"
                "      \"origin\": \"<origin_from_list>\",\n"
                "      \"id\": \"<widget_id_from_list>\",\n"
                "      \"input_args\": { \"<param_name>\": \"<value>\" }\n"
                "    }]\n"
                "  }\n"
                "}\n"
                "Only choose widgets from the provided list. If no data is needed, answer normally."
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

    # If dashboard widgets are available, append them to context for the model to choose from.
    if request.widgets and (
        request.widgets.primary or request.widgets.secondary or request.widgets.extra
    ):
        lines: list[str] = ["Available dashboard widgets (choose from these if needed):\n"]
        for bucket_name, bucket in (
            ("primary", request.widgets.primary or []),
            ("secondary", request.widgets.secondary or []),
            ("extra", request.widgets.extra or []),
        ):
            if not bucket:
                continue
            lines.append(f"[{bucket_name}]\n")
            for w in bucket:
                # Render params: name + current_value snapshot
                param_pairs = [
                    f"{p.name}={(p.current_value if hasattr(p, 'current_value') else None)}"
                    for p in w.params
                ]
                params_str = ", ".join(param_pairs)
                lines.append(
                    "- uuid="
                    + str(getattr(w, "uuid", ""))
                    + ", id="
                    + (w.widget_id or "")
                    + ", origin="
                    + (w.origin or "")
                    + ", name="
                    + (w.name or "")
                    + (f", params: {params_str}" if params_str else "")
                )
            lines.append("")
        if lines:
            context_str += "\n" + "\n".join(lines)

    # If we have context from the latest tool message, append it to the last user/assistant message
    if context_str and len(openai_messages) > 1:
        try:
            openai_messages[-1]["content"] += "\n\n" + context_str  # type: ignore[index]
        except Exception:
            # If last message isn't content-bearing, attach to system as fallback
            openai_messages[0]["content"] += "\n\n" + context_str  # type: ignore[index]

    def _strip_code_fences(text: str) -> str:
        # Remove ```json ... ``` or ``` ... ``` fences if present
        fence_match = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            return fence_match.group(1).strip()
        return text.strip()

    def _extract_json_object(text: str) -> dict | None:
        # Try direct parse, then strip common code fences and retry.
        for candidate in (text, _strip_code_fences(text)):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    async def execution_loop() -> AsyncGenerator[MessageChunkSSE | FunctionCallSSE, None]:
        client = openai.AsyncOpenAI()

        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
        )

        full_text = ""
        async for event in stream:
            if chunk := event.choices[0].delta.content:
                full_text += chunk

        # Try to interpret the model output as a function call request
        try:
            data = _extract_json_object(full_text)
            # Accept both `get_widget_data` and escaped variants (defensive)
            func = (data or {}).get("function") if isinstance(data, dict) else None
            if isinstance(data, dict) and func and func.replace("\\_", "_") == "get_widget_data":
                # Build WidgetRequest list from declared data_sources by resolving
                # widget UUIDs against the provided widgets in the request
                ds_list = (
                    data.get("input_arguments", {}).get("data_sources", []) or []
                )
                # Flatten available widgets for lookup
                all_widgets = []
                if request.widgets:
                    for col in [
                        request.widgets.primary or [],
                        request.widgets.secondary or [],
                        request.widgets.extra or [],
                    ]:
                        all_widgets.extend(col)

                def find_widget(uuid: str | None, wid: str | None):
                    if uuid:
                        for w in all_widgets:
                            if str(getattr(w, "uuid", "")) == str(uuid):
                                return w
                    if wid:
                        for w in all_widgets:
                            if (w.widget_id or "") == wid:
                                return w
                    return None

                widget_requests: list[WidgetRequest] = []
                for ds in ds_list:
                    w_uuid = ds.get("widget_uuid")
                    w_id = ds.get("id")
                    widget = find_widget(w_uuid, w_id)
                    if not widget:
                        continue
                    input_args = ds.get("input_args") or {
                        p.name: p.current_value for p in widget.params
                    }
                    widget_requests.append(
                        WidgetRequest(widget=widget, input_arguments=input_args)
                    )

                if widget_requests:
                    # Emit a typed function call SSE so the UI retrieves data
                    yield get_widget_data(widget_requests)
                    return
        except Exception:
            # Not a function call; continue to output accumulated text
            pass

        # If no function call was detected, stream the final accumulated text (single message)
        if full_text.strip():
            yield message_chunk(full_text.strip())

    async def serialize_events():
        async for event in execution_loop():
            yield event.model_dump(exclude_none=True)

    return EventSourceResponse(content=serialize_events(), media_type="text/event-stream")
