import json
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
    ChatCompletionToolMessageParam,
)
from openbb import obb
from openbb_ai import message_chunk, reasoning_step
from openbb_ai.models import MessageChunkSSE, QueryRequest
from sse_starlette.sse import EventSourceResponse

# --- 1. Tool Definitions ---


def get_financial_metrics(symbol: str):
    """
    Fetches key financial metrics (Market Cap, PE Ratio, ROE) for a stock.
    """
    try:
        # We use the OpenBB Platform SDK to get real data
        # Note: Using 'yfinance' as a free provider for this example.
        # In production, users might use 'fmp' or 'intrinio'.
        data = obb.equity.fundamental.metrics(symbol=symbol, provider="yfinance")

        # Convert to a clean JSON string for the LLM
        return data.to_json(orient="records")
    except Exception as e:
        return f"Error fetching data for {symbol}: {str(e)}"


# Schema to tell OpenAI about our tool
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_metrics",
            "description": "Get key financial metrics (PE, ROE, Market Cap) for a specific stock symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., AAPL, NVDA).",
                    }
                },
                "required": ["symbol"],
            },
        },
    }
]

# Map tool names to the actual Python functions
AVAILABLE_TOOLS = {"get_financial_metrics": get_financial_metrics}

# --- 2. App Setup ---

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
    """Register the agent with the OpenBB Interface."""
    return JSONResponse(
        content={
            "vanilla_agent_fundamental_analysis": {
                "name": "Fundamental Analyst",
                "description": "A vanilla agent that analyzes company fundamentals using real data.",
                # Standard OpenBB agent image
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {"query": "/v1/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": False,
                    "widget-dashboard-search": False,
                },
            }
        },
    )


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """The main agent loop handling tool calls and reasoning."""

    # Initialize conversation with system prompt
    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=(
                "You are a Fundamental Analysis Agent. "
                "You have access to real financial tools. "
                "ALWAYS fetch data using tools before answering questions about companies. "
                "Do not hallucinate metrics."
            ),
        )
    ]

    # Append user history
    for message in request.messages:
        if message.role == "human":
            openai_messages.append(
                ChatCompletionUserMessageParam(role="user", content=message.content)
            )
        elif message.role == "ai":
            openai_messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant", content=str(message.content)
                )
            )

    async def execution_loop() -> AsyncGenerator[MessageChunkSSE, None]:
        client = openai.AsyncOpenAI()

        # --- Step 1: Let LLM decide if it needs tools ---
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            stream=False,  # Wait for full response to check for tools
        )

        message = response.choices[0].message

        # --- Step 2: Did the agent ask for a tool? ---
        if message.tool_calls:
            openai_messages.append(message)  # Add intent to history

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                # *** THE FLEX: Yield a REAL reasoning step to the UI ***
                # This shows up as a "Thinking..." spinner in OpenBB Workspace
                yield reasoning_step(
                    event_type="INFO",
                    message=f"Fetching fundamental data for {fn_args.get('symbol')}...",
                    details={"tool": fn_name, "args": fn_args},
                ).model_dump()

                # Execute the tool
                if fn_name in AVAILABLE_TOOLS:
                    result = AVAILABLE_TOOLS[fn_name](**fn_args)

                    # Feed result back to LLM
                    openai_messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool", tool_call_id=tool_call.id, content=str(result)
                        )
                    )

        # --- Step 3: Stream the final answer ---
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
