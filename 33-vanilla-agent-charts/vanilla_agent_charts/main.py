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
from openbb_ai import chart, message_chunk
from openbb_ai.models import MessageChunkSSE, QueryRequest
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/agents.json")
def get_copilot_description():
    """Agents configuration file for the OpenBB Workspace"""
    return JSONResponse(
        content={
            "vanilla_agent_charts": {
                "name": "Vanilla Agent Charts",
                "description": "A vanilla agent that can produce charts as part of its response.",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {"query": "/v1/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,
                    "widget-dashboard-search": False,
                },
            }
        }
    )


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Agent."""

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
        client = openai.AsyncOpenAI()
        async for event in await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
        ):
            if chunk := event.choices[0].delta.content:
                yield message_chunk(chunk).model_dump()

        # Return some inline charts as part of the response
        # Line chart
        yield message_chunk("\n\nHere is a line chart:\n\n").model_dump()
        yield chart(
            type="line",
            data=[
                {"x": 0, "y": 1},
                {"x": 1, "y": 2},
                {"x": 2, "y": 3},
                {"x": 3, "y": 5},
            ],
            x_key="x",
            y_keys=["y"],
            name="Line Chart",
            description="This is a line chart of the data",
        ).model_dump()

        # Bar chart
        yield message_chunk("\n\nHere is a bar chart:\n\n").model_dump()
        yield chart(
            type="bar",
            data=[
                {"x": "A", "y": 1},
                {"x": "B", "y": 2},
                {"x": "C", "y": 3},
                {"x": "D", "y": 5},
            ],
            x_key="x",
            y_keys=["y"],
            name="Bar Chart",
            description="This is a bar chart of the data",
        ).model_dump()

        # Scatter chart
        yield message_chunk("\n\nHere is a scatter chart:\n\n").model_dump()
        yield chart(
            type="scatter",
            data=[
                {"x": 0, "y": 1},
                {"x": 1, "y": 2},
                {"x": 2, "y": 3},
                {"x": 3, "y": 5},
            ],
            x_key="x",
            y_keys=["y"],
            name="Scatter Chart",
            description="This is a scatter chart of the data",
        ).model_dump()

        # Pie chart
        yield message_chunk("\n\nHere is a pie chart:\n\n").model_dump()
        yield chart(
            type="pie",
            data=[
                {"value": 0, "label": "A"},
                {"value": 1, "label": "B"},
                {"value": 2, "label": "C"},
                {"value": 3, "label": "D"},
            ],
            angle_key="value",
            callout_label_key="label",
            name="Pie Chart",
            description="This is a pie chart of the data",
        ).model_dump()

        # Donut chart
        yield message_chunk("\n\nHere is a donut chart:\n\n").model_dump()
        yield chart(
            type="donut",
            data=[
                {"value": 0, "label": "A"},
                {"value": 1, "label": "B"},
                {"value": 2, "label": "C"},
                {"value": 3, "label": "D"},
            ],
            angle_key="value",
            callout_label_key="label",
            name="Donut Chart",
            description="This is a donut chart of the data",
        ).model_dump()

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
