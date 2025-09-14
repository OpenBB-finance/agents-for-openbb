from typing import AsyncGenerator
from uuid import uuid4
import openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai import get_widget_data, WidgetRequest, message_chunk, cite, citations
from openbb_ai.models import (
    MessageChunkSSE,
    QueryRequest,
    CitationCollectionSSE
)

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
            "vanilla_agent_raw_widget_data_citations": {
                "name": "Vanilla Agent Raw Widget Data Citations",
                "description": "A vanilla agent that automatically retrieves widget data and passes it as raw context to the LLM, and then cites the data that has been retrieved.",
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
            yield get_widget_data(widget_requests)

        # Early exit to retrieve widget data
        async def serialize_widget_events():
            async for event in retrieve_widget_data():
                yield event.model_dump(exclude_none=True)
        
        return EventSourceResponse(
            content=serialize_widget_events(),
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
    citations_list = []
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
            context_str += "Use the following data to answer the question:\n\n"
            result_str = "--- Data ---\n"
            for result in message.data:
                for item in result.items:
                    result_str += f"{item.content}\n"
                    result_str += "------\n"
            context_str += result_str

            # We also need to create citations for the widget data we retrieved.
            for widget_data_request in message.input_arguments["data_sources"]:
                filtered_widgets = list(
                    filter(
                        lambda w: str(w.uuid) == widget_data_request["widget_uuid"],
                        request.widgets.primary,
                    )
                )
                if filtered_widgets:
                    widget = filtered_widgets[0]
                    input_args = widget_data_request["input_args"]
                    
                    # Create extra_details matching UI expectations
                    extra_details = {
                        "Widget Name": widget.name,
                        "Widget Input Arguments": input_args,
                    }
                    
                    citation = cite(
                        widget=widget,
                        input_arguments=input_args,
                        extra_details=extra_details
                    )
                    # Ensure citation has a plain string ID for the UI renderer
                    try:
                        citation.id = str(citation.id)  # type: ignore[attr-defined]
                    except Exception:
                        citation.id = str(uuid4())  # type: ignore[attr-defined]
                    citations_list.append(citation)

    if context_str:
        openai_messages[-1]["content"] += "\n\n" + context_str  # type: ignore

    # Define the execution loop.
    async def execution_loop() -> AsyncGenerator[MessageChunkSSE | CitationCollectionSSE, None]:
        client = openai.AsyncOpenAI()

        # Prime the UI with a minimal chunk so the AI message exists while tokens stream
        yield message_chunk(" ")

        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
        )

        async for event in stream:
            if chunk := event.choices[0].delta.content:
                # Yield typed events like Ada does
                yield message_chunk(chunk)

        # Append citations at the end, after the main content
        if citations_list:
            yield citations(citations_list)

    # Stream the SSEs back to the client exactly like Ada does
    # Let EventSourceResponse serialize the typed events
    async def serialize_events():
        async for event in execution_loop():
            yield event.model_dump(exclude_none=True)
    
    return EventSourceResponse(
        content=serialize_events(),
        media_type="text/event-stream",
    )
