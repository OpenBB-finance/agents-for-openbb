from typing import AsyncGenerator
import openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai.models import MessageChunkSSE, QueryRequest
from openbb_ai import reasoning_step, message_chunk

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
    """Widgets configuration file for the OpenBB Terminal Pro"""
    return JSONResponse(
        content={
            "vanilla_agent_reasoning_steps": {
                "name": "Vanilla Agent Reasoning Steps",
                "description": "A vanilla agent that returns reasoning steps to the OpenBB Workspace.",
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
    """Query the Copilot."""

    # Format the messages into a list of OpenAI messages
    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful financial assistant. Your name is 'Vanilla Agent'.",
        )
    ]

    for message in request.messages:
        if message.role == "human":
            openai_messages.append(
                ChatCompletionUserMessageParam(role="user", content=message.content)
            )
        elif message.role == "ai":
            if isinstance(message.content, str):
                openai_messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=message.content
                    )
                )

    # Define the execution loop.
    async def execution_loop() -> AsyncGenerator[MessageChunkSSE, None]:
        # To send a reasoning step to the OpenBB Workspace, you yield the result
        # of the reasoning_step function from anywhere inside the execution
        # loop.
        yield reasoning_step(
            event_type="INFO",  # Can also be "WARNING" or "ERROR"
            message="Starting to answer the question...",
        ).model_dump()

        # Reasoning steps can also include a table of key-value
        # pairs that will be displayed in the OpenBB Workspace.
        yield reasoning_step(
            event_type="INFO",
            message="An example of a reasoning step with details.",
            details={"key1": "value1", "key2": "value2"},
        ).model_dump()

        client = openai.AsyncOpenAI()
        async for event in await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
        ):
            # Actually stream the LLM response to the client.
            if chunk := event.choices[0].delta.content:
                yield message_chunk(chunk).model_dump()

    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
