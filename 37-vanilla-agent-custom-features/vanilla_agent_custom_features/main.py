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
                    "model": {
                        "label": "Model",
                        "type": "select",
                        "default": "claude-sonnet-4-20250514",
                        "description": "Select the LLM model to use.",
                        "options": [
                            {
                                "label": "Claude Opus 4",
                                "value": "claude-opus-4-0-20250514",
                            },
                            {
                                "label": "Claude Sonnet 4",
                                "value": "claude-sonnet-4-20250514",
                            },
                            {"label": "GPT-4o", "value": "gpt-4o"},
                            {"label": "GPT-4o mini", "value": "gpt-4o-mini"},
                        ],
                    },
                    "agent-name": {
                        "label": "Name of Agent",
                        "type": "text",
                        "default": "Example Agent",
                        "description": "Set the name the agent uses to introduce itself.",
                        "placeholder": "e.g. My Custom Agent",
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
    # Text/select features are sent as "key=value" entries
    workspace_options = getattr(request, "workspace_options", [])

    # Helper to extract a value from "key=value" entries
    def get_option_value(key: str, default: str = "") -> str:
        for opt in workspace_options:
            if opt.startswith(f"{key}="):
                return opt.split("=", 1)[1]
        return default

    # Check which features are enabled
    deep_research_enabled = "deep-research" in workspace_options
    web_search_enabled = "web-search" in workspace_options

    # Read text/select feature values
    model = get_option_value("model", "claude-sonnet-4-20250514")
    agent_name = get_option_value("agent-name", "Example Agent")

    # Build the feature status message
    features_msg = (
        f"- Deep Research: {'✅ Enabled' if deep_research_enabled else '❌ Disabled'}\n"
        f"- Web Search: {'✅ Enabled' if web_search_enabled else '❌ Disabled'}\n"
        f"- Model: {model}\n"
        f"- Agent Name: {agent_name}"
    )

    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=(
                f'Your name is "{agent_name}".\n'
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
