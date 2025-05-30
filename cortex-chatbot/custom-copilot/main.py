import json
import re
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from magentic import (
    AssistantMessage,
    AsyncStreamedStr,
    UserMessage,
)
from sse_starlette.sse import EventSourceResponse

from .cortex_search import _llm
from .models import (
    AgentQueryRequest,
)

load_dotenv(".env")
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:1420",
    "http://localhost:5050",
    "https://pro.openbb.dev",
    "https://pro.openbb.co",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sanitize_message(message: str) -> str:
    """Sanitize a message by escaping forbidden characters."""
    cleaned_message = re.sub(r"(?<!\{)\{(?!{)", "{{", message)
    cleaned_message = re.sub(r"(?<!\})\}(?!})", "}}", cleaned_message)
    return cleaned_message


async def create_message_stream(
    content: AsyncStreamedStr,
) -> AsyncGenerator[dict, None]:
    async for chunk in content:
        yield {
            "event": "copilotMessageChunk",
            "data": json.dumps({"delta": chunk}),
        }


@app.get("/copilots.json")
def get_copilot_description():
    """Widgets configuration file for the OpenBB Terminal Pro"""
    return JSONResponse(
        content=json.load(open((Path(__file__).parent.resolve() / "copilots.json")))
    )


@app.get("/")
async def index():
    return "Custom Snowflake Cortex Search Copilot"


@app.post("/v1/query")
async def query(request: AgentQueryRequest) -> EventSourceResponse:
    """Query the Copilot."""

    # Prepare chat messages for the LLM.
    chat_messages = []
    for message in request.messages:
        if message.role == "ai":
            chat_messages.append(
                AssistantMessage(content=sanitize_message(message.content))
            )
        elif message.role == "human":
            chat_messages.append(UserMessage(content=sanitize_message(message.content)))

    # This is the mean execution loop for the Copilot.
    async def execution_loop():
        # Right now we only do QA one message at a time
        result = _llm(sanitize_message(message.content))
        for event in result:
            yield event

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=create_message_stream(execution_loop()),
        media_type="text/event-stream",
    )
