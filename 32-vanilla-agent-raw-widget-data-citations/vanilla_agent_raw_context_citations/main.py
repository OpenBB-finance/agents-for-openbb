from typing import AsyncGenerator
import openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai.models import MessageChunkSSE, QueryRequest
from openbb_ai import get_widget_data, WidgetRequest, message_chunk, cite, citations

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
)


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
                    citations_list.append(
                        cite(
                            widget=filtered_widgets[0],
                            input_arguments=widget_data_request["input_args"],
                            # You can add any extra details you want to the
                            # citation using the `extra_details` argument.
                            extra_details={
                                **widget_data_request["input_args"],
                            },
                        )
                    )

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

        # If we have any citations, let's yield them at the end to the OpenBB Workspace.
        if citations_list:
            yield citations(citations_list).model_dump()

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
