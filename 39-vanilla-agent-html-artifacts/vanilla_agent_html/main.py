"""
Example agent that produces HTML artifacts.

This agent demonstrates how to return HTML content as artifacts that will be
rendered inline in the OpenBB Workspace copilot chat. Users can also create
dashboard widgets from these HTML artifacts.

NOTE: This requires openbb-ai SDK to be updated to support type="html" in ClientArtifact.
For now, we manually construct the SSE event to bypass SDK validation.
"""

import uuid
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
from openbb_ai.models import QueryRequest
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co", "http://localhost:1420"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def html_artifact(content: str, name: str, description: str) -> dict:
    """
    Create an HTML artifact SSE event.

    Args:
        content: The HTML content to render
        name: A unique name for the artifact
        description: A description of the artifact

    Returns:
        SSE event dict with type="html" artifact
    """
    import json

    return {
        "event": "copilotMessageArtifact",
        "data": json.dumps(
            {
                "type": "html",
                "uuid": str(uuid.uuid4()),
                "name": name,
                "description": description,
                "content": content,
            }
        ),
    }


# Example HTML templates
DASHBOARD_CARD_HTML = """
<div style="font-family: system-ui, -apple-system, sans-serif; padding: 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; color: white; max-width: 400px;">
  <h2 style="margin: 0 0 16px 0; font-size: 20px; font-weight: 600;">Portfolio Summary</h2>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
    <div style="background: rgba(255,255,255,0.15); padding: 16px; border-radius: 12px;">
      <div style="font-size: 28px; font-weight: bold;">$124.5K</div>
      <div style="font-size: 13px; opacity: 0.8; margin-top: 4px;">Total Value</div>
    </div>
    <div style="background: rgba(255,255,255,0.15); padding: 16px; border-radius: 12px;">
      <div style="font-size: 28px; font-weight: bold; color: #4ade80;">+12.3%</div>
      <div style="font-size: 13px; opacity: 0.8; margin-top: 4px;">Today's Change</div>
    </div>
  </div>
  <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.2);">
    <div style="display: flex; justify-content: space-between; font-size: 14px;">
      <span style="opacity: 0.8;">Last updated</span>
      <span>Just now</span>
    </div>
  </div>
</div>
"""

METRIC_CARDS_HTML = """
<div style="font-family: system-ui, -apple-system, sans-serif; display: flex; gap: 16px; flex-wrap: wrap;">
  <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px; min-width: 140px;">
    <div style="color: #64748b; font-size: 13px; margin-bottom: 8px;">Revenue</div>
    <div style="font-size: 24px; font-weight: 600; color: #0f172a;">$2.4M</div>
    <div style="color: #22c55e; font-size: 13px; margin-top: 4px;">↑ 14.2%</div>
  </div>
  <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px; min-width: 140px;">
    <div style="color: #64748b; font-size: 13px; margin-bottom: 8px;">Users</div>
    <div style="font-size: 24px; font-weight: 600; color: #0f172a;">48.2K</div>
    <div style="color: #22c55e; font-size: 13px; margin-top: 4px;">↑ 8.1%</div>
  </div>
  <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px; min-width: 140px;">
    <div style="color: #64748b; font-size: 13px; margin-bottom: 8px;">Conversion</div>
    <div style="font-size: 24px; font-weight: 600; color: #0f172a;">3.2%</div>
    <div style="color: #ef4444; font-size: 13px; margin-top: 4px;">↓ 0.4%</div>
  </div>
</div>
"""

ALERT_BOX_HTML = """
<div style="font-family: system-ui, -apple-system, sans-serif; background: #fef3c7; border: 1px solid #f59e0b; border-radius: 12px; padding: 16px; display: flex; gap: 12px; align-items: flex-start; max-width: 500px;">
  <div style="background: #f59e0b; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-weight: bold;">!</div>
  <div>
    <div style="font-weight: 600; color: #92400e; margin-bottom: 4px;">Market Alert</div>
    <div style="color: #a16207; font-size: 14px; line-height: 1.5;">Unusual trading volume detected in AAPL. Volume is 3.2x higher than the 20-day average. Consider reviewing your positions.</div>
  </div>
</div>
"""


@app.get("/agents.json")
def get_copilot_description():
    """Agents configuration file for the OpenBB Workspace"""
    return JSONResponse(
        content={
            "vanilla_agent_html": {
                "name": "HTML Artifacts Agent",
                "description": "An example agent that produces HTML artifacts rendered inline in the chat.",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
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
    """Query the Agent."""

    # Format the messages into a list of OpenAI messages
    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="""You are a helpful financial assistant that can create beautiful HTML visualizations.
When the user asks for visualizations, summaries, or dashboards, respond with helpful text
and indicate you'll show them an HTML artifact. The system will automatically render HTML artifacts inline.""",
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

    async def execution_loop() -> AsyncGenerator[dict, None]:
        client = openai.AsyncOpenAI()

        # Stream the LLM response
        async for event in await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
        ):
            if chunk := event.choices[0].delta.content:
                yield message_chunk(chunk).model_dump()

        # Append HTML artifacts to demonstrate the feature
        yield message_chunk("\n\nHere's a portfolio dashboard card:\n\n").model_dump()
        yield html_artifact(
            content=DASHBOARD_CARD_HTML,
            name="portfolio_dashboard",
            description="A portfolio summary dashboard card",
        )

        yield message_chunk("\n\nHere are some metric cards:\n\n").model_dump()
        yield html_artifact(
            content=METRIC_CARDS_HTML,
            name="metric_cards",
            description="Key metrics displayed as cards",
        )

        yield message_chunk("\n\nAnd here's an alert notification:\n\n").model_dump()
        yield html_artifact(
            content=ALERT_BOX_HTML,
            name="market_alert",
            description="A market alert notification box",
        )

        yield message_chunk(
            "\n\nYou can click the widget icon on any of these to add them to your dashboard!"
        ).model_dump()

    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
