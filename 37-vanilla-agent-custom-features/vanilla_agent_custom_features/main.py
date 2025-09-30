from typing import AsyncGenerator

import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openbb_ai import message_chunk
from openbb_ai.models import MessageChunkSSE, QueryRequest
from sse_starlette.sse import EventSourceResponse

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
    """Agent descriptor for the OpenBB Workspace."""
    return JSONResponse(
        content={
            "vanilla_agent_custom_features": {
                "name": "Vanilla Agent Custom Features",
                "description": "A simple agent that reports its feature status.",
                "image": (
                    "https://github.com/OpenBB-finance/copilot-for-terminal-pro/"
                    "assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf"
                ),
                "endpoints": {"query": "/v1/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": False,
                    "widget-dashboard-search": False,
                    "deep-research": {
                        "label": "Deep Research",
                        "default": False,
                        "description": "Allows the copilot to do deep research",
                    },
                    "web-search": {
                        "label": "Web Search",
                        "default": True,
                        "description": "Allows the copilot to search the web.",
                    },
                },
            }
        }
    )


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Stream a simple greeting with feature status."""

    # Check workspace_options from request payload
    # workspace_options is a list like ["web-search"] or ["deep-research", "web-search"]
    workspace_options = getattr(request, "workspace_options", [])

    # Check which features are enabled
    deep_research_enabled = "deep-research" in workspace_options
    web_search_enabled = "web-search" in workspace_options

    # Build the feature status message
    features_msg = (
        f"- Deep Research: {'✅ Enabled' if deep_research_enabled else '❌ Disabled'}\n"
        f"- Web Search: {'✅ Enabled' if web_search_enabled else '❌ Disabled'}"
    )

    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=(
                "You are a simple greeting agent.\n"
                "Greet the user and let them know their current feature settings:\n"
                f"{features_msg}\n"
                "Keep your response brief and friendly."
            ),
        )
    ]

    for message in request.messages:
        if message.role == "human":
            openai_messages.append(
                ChatCompletionUserMessageParam(role="user", content=message.content)
            )
        elif message.role == "ai" and isinstance(message.content, str):
            openai_messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant", content=message.content
                )
            )

    async def execution_loop() -> AsyncGenerator[MessageChunkSSE, None]:
        client = openai.AsyncOpenAI()
        async for event in await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
        ):
            if chunk := event.choices[0].delta.content:
                yield message_chunk(chunk)

    return EventSourceResponse(
        content=(
            event.model_dump(exclude_none=True) async for event in execution_loop()
        ),
        media_type="text/event-stream",
    )
