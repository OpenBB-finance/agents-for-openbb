import base64
import io
import logging
from typing import Any, AsyncGenerator, Dict, List, Tuple

import httpx
import openai
import pdfplumber
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openbb_ai import citations, cite, get_widget_data, message_chunk
from openbb_ai.models import (
    Citation,
    CitationCollectionSSE,
    CitationHighlightBoundingBox,
    DataContent,
    DataFileReferences,
    FunctionCallSSE,
    MessageChunkSSE,
    PdfDataFormat,
    QueryRequest,
    SingleDataContent,
    SingleFileReference,
    WidgetRequest,
)
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)


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
    """Agents configuration file for the OpenBB Workspace"""
    return JSONResponse(
        content={
            "vanilla_agent_pdf_citations": {
                "name": "Vanilla Agent PDF Citations",
                "description": "A vanilla agent that handles PDF data with citation support.",
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
    """Query the Copilot."""

    # We only automatically fetch widget data if the last message is from a
    # human, and widgets have been explicitly added to the request.
    last_message = request.messages[-1]
    orchestration_requested = (
        last_message.role == "ai" and last_message.agent_id == "openbb-copilot"
    )
    if (
        (last_message.role == "human" or orchestration_requested)
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

        async def retrieve_widget_data() -> AsyncGenerator[FunctionCallSSE, None]:
            yield get_widget_data(widget_requests)

        # Early exit to retrieve widget data
        return EventSourceResponse(
            content=(event.model_dump() async for event in retrieve_widget_data()),
            media_type="text/event-stream",
        )

    # Format the messages into a list of OpenAI messages
    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful financial assistant. Your name is 'Vanilla Agent PDF Citations'.",
        )
    ]

    context_str = ""
    citations_list: list[Citation] = []
    pdf_text_positions: List[
        Dict[str, Any]
    ] = []  # Store PDF text positions for citation
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
            widget_text, positions = await handle_widget_data(message.data)
            context_str += widget_text
            pdf_text_positions.extend(positions)

            # Create citations for the widget data
            for widget_data_request in message.input_arguments["data_sources"]:
                # Find matching widget
                matching_widgets = [
                    w
                    for w in request.widgets.primary
                    if str(w.uuid) == widget_data_request["widget_uuid"]
                ]

                if not matching_widgets:
                    continue

                widget = matching_widgets[0]

                # Citation 1: Basic widget citation
                basic_citation = cite(
                    widget=widget,
                    input_arguments=widget_data_request["input_args"],
                )
                citations_list.append(basic_citation)

                # Citation 2: Highlight first sentence from PDF
                # This is how you reference a specific sentence
                if pdf_text_positions and len(pdf_text_positions) > 0:
                    first_line = pdf_text_positions[0]

                    pdf_citation = cite(
                        widget=widget,
                        input_arguments=widget_data_request["input_args"],
                        extra_details={
                            "Page": first_line["page"],
                            "Reference": "First sentence of document",
                        },
                    )

                    # Add highlighting for the first sentence
                    pdf_citation.quote_bounding_boxes = [
                        [
                            CitationHighlightBoundingBox(
                                text=first_line["text"][:100],
                                page=first_line["page"],
                                x0=first_line["x0"],
                                top=first_line["top"],
                                x1=first_line["x1"],
                                bottom=first_line["bottom"],
                            )
                        ]
                    ]

                    citations_list.append(pdf_citation)

    if context_str:
        openai_messages[-1]["content"] += "\n\n" + context_str  # type: ignore

    # Define the execution loop.
    async def execution_loop() -> AsyncGenerator[
        MessageChunkSSE | CitationCollectionSSE, None
    ]:
        client = openai.AsyncOpenAI()
        async for event in await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
        ):
            if chunk := event.choices[0].delta.content:
                yield message_chunk(chunk)

        if citations_list:
            yield citations(citations_list)

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=(event.model_dump() async for event in execution_loop()),
        media_type="text/event-stream",
    )


async def _download_file(url: str) -> bytes:
    """Download file from URL."""
    logger.info("Downloading file from %s", url)
    async with httpx.AsyncClient() as client:
        file_content = await client.get(url)
        return file_content.content


def extract_pdf_with_positions(pdf_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract text and positions from PDF."""
    document_text = ""
    text_positions = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Extract text with character-level data for accurate positioning
            if page.chars:
                # Group characters into lines
                lines = {}
                for char in page.chars:
                    y = round(char["top"])  # Round to group by line
                    if y not in lines:
                        lines[y] = {"chars": [], "x0": char["x0"], "x1": char["x1"]}
                    lines[y]["chars"].append(char["text"])
                    lines[y]["x0"] = min(lines[y]["x0"], char["x0"])
                    lines[y]["x1"] = max(lines[y]["x1"], char["x1"])

                # Get first non-empty line for citation
                sorted_lines = sorted(lines.items())
                for y_pos, line_data in sorted_lines[:5]:  # Check first 5 lines
                    line_text = "".join(line_data["chars"]).strip()
                    if line_text and len(line_text) > 10:  # Skip very short lines
                        text_positions.append(
                            {
                                "text": line_text,
                                "page": page_num,
                                "x0": line_data["x0"],
                                "top": y_pos,
                                "x1": line_data["x1"],
                                "bottom": y_pos + 12,  # Standard line height
                            }
                        )
                        break  # Just get the first meaningful line

            # Also extract full text for context
            page_text = page.extract_text()
            if page_text:
                document_text += page_text + "\n\n"

    return document_text, text_positions


# Files can either be served from a URL...
async def _get_url_pdf_text(
    data: SingleFileReference,
) -> Tuple[str, List[Dict[str, Any]]]:
    file_content = await _download_file(str(data.url))
    return extract_pdf_with_positions(file_content)


# ... or via base64 encoding.
async def _get_base64_pdf_text(
    data: SingleDataContent,
) -> Tuple[str, List[Dict[str, Any]]]:
    file_content = base64.b64decode(data.content)
    return extract_pdf_with_positions(file_content)


async def handle_widget_data(
    data: list[DataContent | DataFileReferences],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Process widget data and extract PDF text with positions if applicable.

    Returns:
        Tuple containing:
        - result_str: Formatted text content from all data sources
        - all_positions: Text position data for PDF citations (empty for non-PDFs)
    """
    result_str = "--- Data ---\n"
    all_positions = []

    for result in data:
        for item in result.items:
            if isinstance(item.data_format, PdfDataFormat):
                result_str += f"===== {item.data_format.filename} =====\n"
                if isinstance(item, SingleDataContent):
                    # Handle the base64 PDF case.
                    text, positions = await _get_base64_pdf_text(item)
                    result_str += text
                    all_positions.extend(positions)
                elif isinstance(item, SingleFileReference):
                    # Handle the URL PDF case.
                    text, positions = await _get_url_pdf_text(item)
                    result_str += text
                    all_positions.extend(positions)
            else:
                # Handle other data formats by just dumping the content as a
                # string.
                result_str += f"{item.content}\n"
            result_str += "------\n"

    return result_str, all_positions
