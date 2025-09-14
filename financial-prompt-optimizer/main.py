from typing import AsyncGenerator
import openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai.models import MessageChunkSSE, QueryRequest
from openbb_ai import message_chunk

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
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
    """Agent descriptor for the OpenBB Workspace."""
    return JSONResponse(
        content={
            "financial_prompt_optimizer": {
                "name": "Financial Prompt Optimizer",
                "description": "Optimizes a user's prompt for finance: clearer, more specific, and actionable.",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {"query": "http://localhost:7777/v1/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": False,
                    "widget-dashboard-search": False,
                },
            }
        }
    )


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Stream a concise optimized prompt and rationale."""

    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=(
                "You are a concise Financial Prompt Optimizer.\n"
                "Rewrite the user's prompt to be clearer, more specific, and immediately actionable for financial analysis.\n"
                "Always return exactly the improved prompt:\n"
                "Optimized Prompt: <detailed improved prompt with step-by-step>\n"
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
                ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
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
        content=(event.model_dump(exclude_none=True) async for event in execution_loop()),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7777, reload=True)
