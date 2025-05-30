import json
import logging
from typing import AsyncGenerator
import uuid
import openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from dotenv import load_dotenv
from common.models import (
    CitationCollection,
    CitationCollectionSSE,
    FunctionCallSSE,
    LlmClientFunctionCallResultMessage,
    MessageChunkSSE,
    MessageChunkSSEData,
    QueryRequest,
)
from common.agent import create_citation

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageToolCallParam,
)

from openai.types.chat.chat_completion_message_tool_call_param import Function


from .prompts import render_system_prompt
from .functions import GET_WIDGET_DATA_FUNCTION_DEFINITION, get_widget_data


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
            "simple_copilot_rfc": {
                "name": "Simple Copilot with OpenBB Function Calling",
                "description": "A simple copilot that can answer questions, execute OpenBB function calls, and return reasoning steps.",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {"query": "http://localhost:7777/v1/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,
                    "widget-dashboard-search": True,
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
            content=render_system_prompt(widget_collection=request.widgets),
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
        elif message.role == "tool":
            tool_call_id = str(uuid.uuid4())[:13]
            function = Function(
                name=message.function,
                arguments=json.dumps(message.input_arguments),
            )
            # First the function call itself
            openai_messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCallParam(
                            type="function",
                            id=tool_call_id,
                            function=function,
                        )
                    ],
                )
            )

            # Then the function call result
            result_str = "--- Data ---\n"
            for result in message.data:
                for item in result.items:
                    result_str += f"{item.content}\n"
                    result_str += "------\n"

            openai_messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    content=result_str,
                    tool_call_id=tool_call_id,
                )
            )

    # Create citations for the most recent function call
    citations = []
    last_message = request.messages[-1]
    if isinstance(last_message, LlmClientFunctionCallResultMessage):
        if citation := create_citation(
            function_call_result=last_message, request=request
        ):
            citations.append(citation)

    # Define the execution loop.
    async def execution_loop() -> AsyncGenerator[MessageChunkSSE, None]:
        client = openai.AsyncOpenAI()
        function_name = None
        function_call_arguments = ""
        async for event in await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
            tools=[GET_WIDGET_DATA_FUNCTION_DEFINITION],
        ):
            if event.choices[0].delta.tool_calls:
                if tool_calls := event.choices[0].delta.tool_calls:
                    for tool_call in tool_calls:
                        if function := tool_call.function:
                            # If we don't have a function name yet, let's get
                            # it.  (it won't be available on the completion
                            # chunk, so we grab it now)
                            if not function_name:
                                function_name = function.name
                            # Let's accumulate the streamed tool calls
                            if function.arguments:
                                function_call_arguments += function.arguments

            # If we're done with tool calls, let's call the function.
            if event.choices[0].finish_reason == "tool_calls":
                if function_name == "get_widget_data":
                    async for event in get_widget_data(
                        **json.loads(function_call_arguments), request=request
                    ):
                        yield event.model_dump()
                        if isinstance(event, FunctionCallSSE):
                            # If we return a function call to the front-end, we
                            # need to stop the loop and close the connection.
                            return
                else:
                    raise NotImplementedError(
                        "This example only handles function calling to the front-end."
                    )

            # If we're streaming text chunks, let's yield them to the front-end.
            if event.choices[0].delta.content:
                yield MessageChunkSSE(
                    data=MessageChunkSSEData(delta=event.choices[0].delta.content),
                ).model_dump()

            if event.choices[0].finish_reason == "stop":
                if citations:
                    yield CitationCollectionSSE(
                        data=CitationCollection(citations=citations)
                    ).model_dump()

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
