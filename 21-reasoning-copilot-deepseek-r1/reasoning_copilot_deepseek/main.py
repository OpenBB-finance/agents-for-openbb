import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .prompts import SYSTEM_PROMPT

from dotenv import load_dotenv
from common import agent
from openbb_ai.models import (
    QueryRequest,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@app.get("/agents.json")
def get_copilot_description():
    """Widgets configuration file for the OpenBB Terminal Pro"""
    return JSONResponse(
        content={
            "reasoning_copilot_deepseek": {
                "name": "Reasoning Deepseek Copilot",
                "description": "A copilot that uses Deepseek R1 as its LLM.",
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

    openbb_agent = agent.OpenBBAgent(
        query_request=request,
        system_prompt=SYSTEM_PROMPT,
        chat_class=agent.OpenRouterChat,
        model="meta-llama/llama-4-maverick",
    )

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=openbb_agent.run(),
        media_type="text/event-stream",
    )
