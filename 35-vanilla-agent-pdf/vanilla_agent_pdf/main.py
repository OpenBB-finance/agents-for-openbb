import base64
from typing import AsyncGenerator
import httpx
import openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai.models import (
    MessageChunkSSE,
    QueryRequest,
    SingleFileReference,
    SingleDataContent,
    PdfDataFormat,
    DataContent,
    DataFileReferences,
    WidgetRequest,
)
from openbb_ai import message_chunk, get_widget_data

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
)

import logging
import pdfplumber
import io

logger = logging.getLogger(__name__)


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
            "vanilla_agent_pdf": {
                "name": "Vanilla Agent PDF",
                "description": "A vanilla agent that can handle PDF data as part of its response.",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {"query": "http://localhost:7777/v1/query"},
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
    """Query the Copilot."""

    # We only automatically fetch widget data if the last message is from a
    # human, and widgets have been explicitly added to the request.
    if (
        request.messages[-1].role == "human"
        and request.widgets
        and request.widgets.primary
    ):
        widget_requests: list[WidgetRequest] = []
        for widget in request.widgets.primary:
            widget_requests.append(
                WidgetRequest(
                    widget=widget,
                    input_arguments={
                        param.name: param.current_value for param in widget.params
                    },
                )
            )

        async def retrieve_widget_data():
            yield get_widget_data(widget_requests).model_dump()

        # Early exit to retrieve widget data
        return EventSourceResponse(
            content=retrieve_widget_data(),
            media_type="text/event-stream",
        )

    # Format the messages into a list of OpenAI messages
    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful financial assistant. Your name is 'Vanilla Agent'.",
        )
    ]

    context_str = ""
    for index, message in enumerate(request.messages):
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
        # We only add the most recent tool call / widget data to context.  We do
        # this **only for this particular example** to prevent
        # previously-retrieved widget data from piling up and exceeding the
        # context limit of the LLM.
        elif message.role == "tool" and index == len(request.messages) - 1:
            context_str += await handle_widget_data(message.data)

    if context_str:
        openai_messages[-1]["content"] += "\n\n" + context_str  # type: ignore

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

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )


async def _download_file(url: str) -> bytes:
    logger.info(f"Downloading file from {url}")
    async with httpx.AsyncClient() as client:
        file_content = await client.get(url)
        return file_content.content


# Files can either be served from a URL...
async def _get_url_pdf_text(data: SingleFileReference) -> str:
    file_content = await _download_file(str(data.url))
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        document_text = ""
        for page in pdf.pages:
            document_text += page.extract_text()
            document_text += "\n\n"
        return document_text


# ... or via base64 encoding.
async def _get_base64_pdf_text(data: SingleDataContent) -> str:
    file_content = base64.b64decode(data.content)
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        document_text = ""
        for page in pdf.pages:
            document_text += page.extract_text()
            document_text += "\n\n"
        return document_text


async def handle_widget_data(data: list[DataContent | DataFileReferences]) -> str:
    result_str = "--- Data ---\n"
    for result in data:
        for item in result.items:
            if isinstance(item.data_format, PdfDataFormat):
                result_str += f"===== {item.data_format.filename} =====\n"
                if isinstance(item, SingleDataContent):
                    # Handle the base64 PDF case.
                    result_str += await _get_base64_pdf_text(item)
                elif isinstance(item, SingleFileReference):
                    # Handle the URL PDF case.
                    result_str += await _get_url_pdf_text(item)
            else:
                # Handle other data formats by just dumping the content as a
                # string.
                result_str += f"{item.content}\n"
            result_str += "------\n"
    return result_str
