import json
from datetime import datetime, timezone
from pathlib import Path
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
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co", "http://localhost:1420"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEEDBACK_FILE = Path(__file__).parent.parent / "feedback.json"


class FeedbackRequest(BaseModel):
    vote: str
    tags: list[str] = []
    user_comment: str = ""
    ai_response: str = ""
    user_prompt: str = ""
    trace_id: str = ""


@app.get("/agents.json")
def get_copilot_description():
    """Agent configuration for OpenBB Workspace."""
    return JSONResponse(
        content={
            "vanilla_agent_feedback": {
                "name": "Vanilla Agent Feedback",
                "description": "A vanilla agent that receives and persists user feedback (thumbs up/down).",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {
                    "query": "/v1/query",
                    "feedback": "/v1/feedback",
                },
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": False,
                    "widget-dashboard-search": False,
                    "feedback": True,
                },
            }
        }
    )


@app.post("/v1/feedback")
async def feedback(request: FeedbackRequest):
    """Receive and persist user feedback to a local JSON file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **request.model_dump(),
    }

    entries = []
    if FEEDBACK_FILE.exists():
        entries = json.loads(FEEDBACK_FILE.read_text())

    entries.append(entry)
    FEEDBACK_FILE.write_text(json.dumps(entries, indent=2))

    return JSONResponse(content={"status": "ok"})


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot."""
    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful financial assistant. Your name is 'Vanilla Agent Feedback'.",
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
                yield message_chunk(chunk).model_dump()

    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
