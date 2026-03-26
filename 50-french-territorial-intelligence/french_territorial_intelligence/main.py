from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai.models import QueryRequest
from french_territorial_intelligence.agent import stream_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/agents.json")
async def get_copilot_description() -> JSONResponse:
    """Agent metadata for OpenBB Terminal Pro."""
    return JSONResponse(
        content={
            "french-territorial-intelligence": {
                "name": "French Territorial Intelligence",
                "description": (
                    "Cross-references French open data (enterprises, real estate, "
                    "demographics) to produce territorial intelligence insights. "
                    "Ask about any French city or department."
                ),
                "image": "https://raw.githubusercontent.com/tawiza/tawiza/main/docs/assets/logo.png",
                "endpoints": {"query": "/v1/query"},
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
    """Query the French Territorial Intelligence agent."""
    return EventSourceResponse(stream_response(request))
