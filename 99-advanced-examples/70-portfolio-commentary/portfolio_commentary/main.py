import os
import json
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openbb_ai import message_chunk, reasoning_step, get_widget_data, WidgetRequest
from openbb_ai.models import QueryRequest
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT

load_dotenv(".env")

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
            "portfolio_commentary": {
                "name": "Portfolio Commentary",
                "description": "Analyzes your portfolio widgets to produce a concise, structured portfolio commentary.",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {"query": "/v1/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,
                    "widget-dashboard-search": False,
                    "perplexity-search": {
                        "label": "Perplexity for Search",
                        "default": False,
                        "description": "Use Perplexity via OpenRouter for web search context in commentary",
                    },
                },
            }
        }
    )




@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot."""

    # Check workspace_options from request payload
    workspace_options = getattr(request, "workspace_options", []) or []
    perplexity_enabled = "perplexity-search" in workspace_options

    # We only automatically fetch widget data if the last message is from a
    # human, and widgets have been explicitly added to the request (primary widgets)
    last_message = request.messages[-1] if request.messages else None
    
    # Check if we should retrieve widget data (like vanilla agents)
    orchestration_requested = (
        last_message
        and getattr(last_message, "role", None) == "ai"
        and getattr(last_message, "agent_id", None) == "openbb-copilot"
    )
    if (
        last_message
        and (last_message.role == "human" or orchestration_requested)
        and request.widgets
        and request.widgets.primary
    ):
        # Show which widgets we're retrieving data from
        widget_names = [w.name or w.widget_id for w in request.widgets.primary]
        if len(widget_names) == 1:
            message = f"Retrieving data from {widget_names[0]} to answer your question..."
        else:
            message = f"Retrieving data from {len(widget_names)} widgets: {', '.join(widget_names[:3])}{'...' if len(widget_names) > 3 else ''}"
        
        async def retrieve_widget_data():
            yield reasoning_step(
                event_type="INFO",
                message=message
            ).model_dump()
            
            # Create widget requests using the OpenBB AI method (only primary widgets)
            widget_requests = []
            for widget in request.widgets.primary:
                input_args = {}
                try:
                    input_args = {
                        p.name: (getattr(p, "current_value", None) if getattr(p, "current_value", None) is not None else getattr(p, "default_value", None))
                        for p in widget.params
                    }
                except Exception:
                    input_args = {}

                widget_requests.append(
                    WidgetRequest(
                        widget=widget,
                        input_arguments=input_args,
                    )
                )
            
            # Retrieve the widget data for the selected widgets
            yield get_widget_data(widget_requests).model_dump()

        # Early exit to retrieve widget data
        return EventSourceResponse(
            content=retrieve_widget_data(),
            media_type="text/event-stream",
        )

    async def process_request():
        # Check if this is a tool response (widget data)
        last_message = request.messages[-1] if request.messages else None
        is_tool_response = (last_message and hasattr(last_message, 'role') 
                           and last_message.role == "tool" 
                           and hasattr(last_message, 'data') and last_message.data)
        
        # If it's a tool response, analyze the data with LLM
        if is_tool_response:
            yield reasoning_step(
                event_type="INFO",
                message="Analyzing the widget data to provide comprehensive insights..."
            ).model_dump()
            
            # Build context from all widget data
            context_str = "Use the following data to answer the question:\n\n"
            result_str = "--- Widget Data ---\n"
            for result in last_message.data:
                for item in result.items:
                    result_str += f"{item.content}\n"
                    result_str += "------\n"
            context_str += result_str
            
            # Build messages for LLM using the prompts.py system prompt
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add conversation history (skip the tool response)
            for msg in request.messages[:-1]:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    if msg.role == "human":
                        messages.append({"role": "user", "content": msg.content})
                    elif msg.role == "ai" and isinstance(msg.content, str):
                        messages.append({"role": "assistant", "content": msg.content})
            
            # Add the widget data context to the last user message
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + context_str
            else:
                # Fallback if no user message found
                messages.append({
                    "role": "user", 
                    "content": f"Please analyze this data and provide insights:\n\n{context_str}"
                })
            
            # Call LLM for analysis
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                yield message_chunk("Error: OPENROUTER_API_KEY not configured").model_dump()
                return
                
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://pro.openbb.dev",
                "X-Title": "OpenBB Terminal Pro",
                "Accept": "application/json",
            }
            
            # Model selection: env default, override with feature toggle for Perplexity
            model_name = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
            if perplexity_enabled:
                model_name = os.environ.get("OPENROUTER_MODEL_PERPLEXITY", "perplexity/sonar")
            # Prefer streaming; some models (e.g., certain Perplexity routes) may not support it reliably.
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    stream_flag = not perplexity_enabled
                    data = {
                        "model": model_name,
                        "messages": messages,
                        "stream": stream_flag,
                    }
                    if stream_flag:
                        async with client.stream("POST", url, headers=headers, json=data) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line or not line.startswith("data: "):
                                    continue
                                line = line[6:].strip()
                                if line == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(line)
                                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                                    if content:
                                        yield message_chunk(content).model_dump()
                                except json.JSONDecodeError:
                                    continue
                    else:
                        resp = await client.post(url, headers=headers, json=data)
                        resp.raise_for_status()
                        body = resp.json()
                        content = (
                            body.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        if content:
                            yield message_chunk(content).model_dump()
            except Exception:
                # Fallback to non-stream default model if Perplexity or streaming fails
                fallback_model = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
                payload = {"model": fallback_model, "messages": messages, "stream": False}
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        resp = await client.post(url, headers=headers, json=payload)
                        resp.raise_for_status()
                        body = resp.json()
                        content = (
                            body.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        if content:
                            yield message_chunk(content).model_dump()
                except Exception:
                    yield message_chunk("Error generating commentary. Please try again or switch models.").model_dump()
            return

        # For any remaining human messages (no Primary fetch and no tool data), defer to the model
        if last_message and last_message.role == "human":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                yield message_chunk("Error: OPENROUTER_API_KEY not configured").model_dump()
                return

            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://pro.openbb.dev",
                "X-Title": "OpenBB Terminal Pro",
                "Accept": "application/json",
            }

            # Build conversation history (system + user/assistant messages only)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in request.messages:
                if getattr(msg, "role", None) == "human":
                    messages.append({"role": "user", "content": msg.content})
                elif getattr(msg, "role", None) == "ai" and isinstance(msg.content, str):
                    messages.append({"role": "assistant", "content": msg.content})

            # Model selection with Perplexity toggle
            model_name = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
            if perplexity_enabled:
                model_name = os.environ.get("OPENROUTER_MODEL_PERPLEXITY", "perplexity/sonar")

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    stream_flag = not perplexity_enabled
                    payload = {"model": model_name, "messages": messages, "stream": stream_flag}
                    if stream_flag:
                        async with client.stream("POST", url, headers=headers, json=payload) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line or not line.startswith("data: "):
                                    continue
                                line = line[6:].strip()
                                if line == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(line)
                                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                                    if content:
                                        yield message_chunk(content).model_dump()
                                except json.JSONDecodeError:
                                    continue
                    else:
                        resp = await client.post(url, headers=headers, json=payload)
                        resp.raise_for_status()
                        body = resp.json()
                        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if content:
                            yield message_chunk(content).model_dump()
            except Exception:
                # Fallback to non‑stream default model
                fallback_model = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        payload = {"model": fallback_model, "messages": messages, "stream": False}
                        resp = await client.post(url, headers=headers, json=payload)
                        resp.raise_for_status()
                        body = resp.json()
                        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if content:
                            yield message_chunk(content).model_dump()
                except Exception:
                    yield message_chunk("Sorry, I couldn’t generate a response. Please try again.").model_dump()
            return
        
        # Default response (should rarely be hit)
        yield message_chunk("How can I help with your portfolio commentary?").model_dump()

    return EventSourceResponse(
        content=process_request(),
        media_type="text/event-stream",
    )
