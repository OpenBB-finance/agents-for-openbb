import json
import logging
import os
import uuid
from pathlib import Path
from typing import AsyncGenerator, Callable

import httpx
from common.agent import get_remote_data, reasoning_step, remote_function_call
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from magentic import (
    AssistantMessage,
    AsyncStreamedStr,
    Chat,
    FunctionCall,
    FunctionResultMessage,
    SystemMessage,
    UserMessage,
)
from openbb_ai.models import (
    DataContent,
    FunctionCallSSE,
    FunctionCallSSEData,
    QueryRequest,
    StatusUpdateSSE,
    Widget,
    WidgetCollection,
)
from sse_starlette.sse import EventSourceResponse

from .prompts import SYSTEM_PROMPT

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


@app.get("/")
async def root():
    """Root endpoint that returns basic API information."""
    return {
        "name": "Portfolio Commentary API",
        "version": "1.0.0",
        "endpoints": ["/v1/query", "/agents.json"],
        "status": "operational",
    }


# Direct web search function that returns a string
async def perplexity_web_search(query: str) -> str:
    """Search the web using Perplexity's API through OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY environment variable is not set"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://pro.openbb.dev",  # Required by OpenRouter
        "X-Title": "OpenBB Terminal Pro",  # Optional but recommended
    }
    data = {
        "model": "perplexity/sonar",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and up-to-date information from the web.",
            },
            {"role": "user", "content": query},
        ],
        "stream": False,  # Not using streaming here - will get complete response
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
    except Exception as e:
        error_message = f"Error searching the web: {str(e)}"
        logger.error(error_message)
        return error_message


# Custom patched version of run_agent that properly handles our perplexity_web_search function
async def custom_run_agent(
    chat: Chat, max_completions: int = 10
) -> AsyncGenerator[dict, None]:
    completion_count = 0
    # We set a limit to avoid infinite loops.
    while completion_count < max_completions:
        completion_count += 1
        chat = await chat.asubmit()
        # Handle a streamed text response.
        if isinstance(chat.last_message.content, AsyncStreamedStr):
            async for chunk in chat.last_message.content:
                yield {
                    "event": "copilotMessageChunk",
                    "data": json.dumps({"delta": chunk}),
                }
            yield {"event": "copilotMessageChunk", "data": json.dumps({"delta": ""})}
            return
        # Handle a function call.
        elif isinstance(chat.last_message.content, FunctionCall):
            # Check if it's our perplexity function
            if chat.last_message.content.function.__name__ == "perplexity_web_search":
                # Special handling for perplexity_web_search
                try:
                    # Print function call details for debugging
                    function_call = chat.last_message.content
                    logger.info(f"Function call details: {dir(function_call)}")
                    logger.info(f"Function call dict: {function_call.__dict__}")

                    # Try to extract the query in multiple ways
                    query = None

                    # Method 1: Try to get from kwargs
                    kwargs = getattr(function_call, "kwargs", {})
                    if kwargs and "query" in kwargs:
                        query = kwargs["query"]
                        logger.info(f"Got query from kwargs: {query}")

                    # Method 2: Try to access arguments through function signature
                    if query is None and hasattr(function_call, "arguments"):
                        arguments = function_call.arguments
                        if isinstance(arguments, dict) and "query" in arguments:
                            query = arguments["query"]
                            logger.info(f"Got query from arguments: {query}")

                    # Method 3: Try to get from args if it exists
                    if query is None and hasattr(function_call, "args"):
                        args = function_call.args
                        if args and len(args) > 0:
                            query = args[0]
                            logger.info(f"Got query from args: {query}")

                    # Final fallback
                    if query is None:
                        logger.warning("Could not extract query from function call")
                        query = "Unable to determine query"

                    # First, yield an info message that we're searching
                    yield reasoning_step(
                        event_type="INFO",
                        message=f"Searching the web for: {query}",
                        details=[],
                    ).model_dump()

                    # Call the function directly
                    result = await perplexity_web_search(query)

                    # Add the result to the chat
                    chat = chat.add_message(
                        FunctionResultMessage(
                            content=result,
                            function_call=chat.last_message.content,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error in perplexity_web_search: {str(e)}", exc_info=True
                    )
                    error_result = f"Error searching the web: {str(e)}"
                    chat = chat.add_message(
                        FunctionResultMessage(
                            content=error_result,
                            function_call=chat.last_message.content,
                        )
                    )
            else:
                # For other functions, use the standard approach
                function_call_result = ""
                try:
                    # Call the function and iterate over its results
                    function_call = chat.last_message.content
                    # Instead of iterating, just call it directly
                    function_result = await function_call()
                    function_call_result = str(function_result)
                except Exception as e:
                    logger.error(f"Error calling function: {str(e)}", exc_info=True)
                    function_call_result = f"Error: {str(e)}"

                # Add the function result to the chat
                chat = chat.add_message(
                    FunctionResultMessage(
                        content=function_call_result,
                        function_call=chat.last_message.content,
                    )
                )
        else:
            # If the last message is not a function call or streamed output,
            # it's probably just regular content - yield it and return
            if hasattr(chat.last_message, "content") and chat.last_message.content:
                yield {
                    "event": "copilotMessageChunk",
                    "data": json.dumps({"delta": str(chat.last_message.content)}),
                }
                yield {
                    "event": "copilotMessageChunk",
                    "data": json.dumps({"delta": ""}),
                }
            else:
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {"message": "Unexpected response type from LLM"}
                    ),
                }
            return


@app.get("/agents.json")
def get_copilot_description():
    """Widgets configuration file for the OpenBB Terminal Pro"""
    return JSONResponse(
        content=json.load(open((Path(__file__).parent.resolve() / "agents.json")))
    )


# Function to handle widget data response formatting
async def handle_widget_data(data: list[DataContent]) -> str:
    result_str = "--- Data ---\n"
    for content in data:
        result_str += f"{content.content}\n"
        result_str += "------\n"
    return result_str


# Function to create the widget data retrieval function
def get_widget_data(widget_collection: WidgetCollection) -> Callable:
    # Combine primary and secondary widgets
    widgets = (
        widget_collection.primary + widget_collection.secondary
        if widget_collection
        else []
    )

    @remote_function_call(
        function="get_widget_data", output_formatter=handle_widget_data
    )
    async def _get_widget_data(
        widget_uuid: str,
    ) -> AsyncGenerator[FunctionCallSSE | StatusUpdateSSE, None]:
        """Retrieve data for a widget by specifying the widget UUID."""

        # Find the widget that matches the UUID
        matching_widgets = list(
            filter(lambda widget: str(widget.uuid) == widget_uuid, widgets)
        )
        widget = matching_widgets[0] if matching_widgets else None

        # If we can't find the widget, report an error
        if not widget:
            yield reasoning_step(
                event_type="ERROR",
                message="Unable to retrieve data for widget (does not exist)",
                details={"widget_uuid": widget_uuid},
            )
            yield f"Unable to retrieve data for widget with UUID: {widget_uuid} (it is not present on the dashboard)"
            return

        # Let the user know we're retrieving data
        yield reasoning_step(
            event_type="INFO",
            message=f"Retrieving data for widget: {widget.name}...",
            details={"widget_uuid": widget_uuid},
        )

        # Request the widget data
        yield get_remote_data(
            widget=widget,
            # Use the current values of widget parameters
            input_arguments={
                param.name: param.current_value for param in widget.params
            },
        )

    return _get_widget_data


# Generate a system prompt that includes widget information
def render_system_prompt(widget_collection: WidgetCollection | None = None) -> str:
    from .prompts import SYSTEM_PROMPT  # Import the base system prompt

    widgets_prompt = "# Available Widgets\n\n"

    # Primary widgets section
    widgets_prompt += "## Primary Widgets (prioritize using these):\n\n"
    for widget in widget_collection.primary if widget_collection else []:
        widgets_prompt += _render_widget(widget)

    # Secondary widgets section
    widgets_prompt += "\n## Secondary Widgets:\n\n"
    for widget in widget_collection.secondary if widget_collection else []:
        widgets_prompt += _render_widget(widget)

    # Append widget information to system prompt
    complete_prompt = f"{SYSTEM_PROMPT}\n\nYou can use the following functions to help you answer the user's query:\n"
    complete_prompt += "- get_widget_data(widget_uuid: str) -> str: Get the data for a widget by specifying its UUID.\n\n"
    complete_prompt += widgets_prompt

    return complete_prompt


# Helper function to format widget information
def _render_widget(widget: Widget) -> str:
    widget_str = ""
    widget_str += (
        f"uuid: {widget.uuid} <-- use this to retrieve the data for the widget\n"
    )
    widget_str += f"name: {widget.name}\n"
    widget_str += f"description: {widget.description}\n"
    widget_str += "parameters:\n"
    for param in widget.params:
        widget_str += f"  {param.name}={param.current_value}\n"
    widget_str += "-------\n"
    return widget_str


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot."""

    # Custom function to process messages that handles the widget response data properly
    async def custom_process_messages(messages, system_prompt, functions):
        chat_messages = [SystemMessage(system_prompt)]

        for message in messages:
            if hasattr(message, "role"):
                if message.role == "human" and hasattr(message, "content"):
                    chat_messages.append(UserMessage(content=message.content))
                elif message.role == "ai" and hasattr(message, "content"):
                    if isinstance(message.content, str):
                        chat_messages.append(AssistantMessage(content=message.content))
                    else:
                        # Just skip function call messages, as they'll be handled by the result
                        pass
                elif message.role == "tool" and message.function == "get_widget_data":
                    # For get_widget_data results, we'll format them directly
                    result_str = "--- Widget Data ---\n"
                    for content in message.data:
                        for item in content.items:
                            result_str += f"{item.content}\n"
                            result_str += "------\n"

                    # Add a user message with the widget data
                    chat_messages.append(
                        UserMessage(content=f"Widget data retrieved: \n{result_str}")
                    )

        return chat_messages

    # Simple, direct approach without complex streaming logic
    async def direct_response():
        try:
            # Create the get_widget_data function if widgets are available
            functions = []
            if request.widgets:
                functions.append(get_widget_data(widget_collection=request.widgets))

            # Get appropriate system prompt with widget information
            system_prompt = (
                render_system_prompt(widget_collection=request.widgets)
                if request.widgets
                else SYSTEM_PROMPT
            )

            # Check if this is a response to a previous function call
            previous_function_call = None
            function_call_result = None

            for message in request.messages:
                # Check if there's a previous function call from the AI
                if (
                    hasattr(message, "role")
                    and message.role == "ai"
                    and hasattr(message, "content")
                ):
                    if not isinstance(message.content, str):
                        # This could be a function call
                        if (
                            hasattr(message.content, "function")
                            and message.content.function == "get_widget_data"
                        ):
                            previous_function_call = message
                    # Also check if it's a string that might be JSON
                    elif isinstance(message.content, str):
                        try:
                            content_obj = json.loads(message.content)
                            if (
                                isinstance(content_obj, dict)
                                and content_obj.get("function") == "get_widget_data"
                            ):
                                previous_function_call = message
                        except (json.JSONDecodeError, TypeError):
                            pass

                # Check if there's a function call result from a tool
                if (
                    hasattr(message, "role")
                    and message.role == "tool"
                    and hasattr(message, "function")
                ):
                    if message.function == "get_widget_data":
                        function_call_result = message

            logger.info(f"Previous function call: {previous_function_call}")
            logger.info(f"Function call result: {function_call_result}")

            # Use our custom processor instead of the standard one
            processed_messages = await custom_process_messages(
                messages=request.messages,
                system_prompt=system_prompt,
                functions=functions,
            )

            # Format messages for OpenAI API
            formatted_messages = []
            for msg in processed_messages:
                if isinstance(msg, SystemMessage):
                    formatted_messages.append(
                        {"role": "system", "content": msg.content}
                    )
                elif isinstance(msg, UserMessage):
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AssistantMessage) and isinstance(msg.content, str):
                    formatted_messages.append(
                        {"role": "assistant", "content": msg.content}
                    )
                elif isinstance(msg, AssistantMessage) and isinstance(
                    msg.content, FunctionCall
                ):
                    # Handle function calls in the AssistantMessage
                    if msg.content.function.__name__ == "_get_widget_data":
                        widget_uuid = msg.content.kwargs.get("widget_uuid", "")
                        formatted_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": str(uuid.uuid4()),
                                        "type": "function",
                                        "function": {
                                            "name": "get_widget_data",
                                            "arguments": json.dumps(
                                                {"widget_uuid": widget_uuid}
                                            ),
                                        },
                                    }
                                ],
                            }
                        )
                elif isinstance(msg, FunctionResultMessage):
                    # Add the result of a function call
                    if (
                        hasattr(msg.function_call, "function")
                        and msg.function_call.function.__name__ == "_get_widget_data"
                    ):
                        formatted_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": next(
                                    (
                                        call.get("id")
                                        for msg_idx, msg_val in enumerate(
                                            formatted_messages
                                        )
                                        for call in msg_val.get("tool_calls", [])
                                        if msg_val.get("role") == "assistant"
                                        and call.get("function", {}).get("name")
                                        == "get_widget_data"
                                    ),
                                    str(uuid.uuid4()),
                                ),
                                "content": msg.content,
                            }
                        )

            # Create tools definition for web search and widget data retrieval
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "perplexity_web_search",
                        "description": "Search the web using Perplexity's API through OpenRouter.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to look up on the web",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]

            # Add get_widget_data tool if widgets are available
            if request.widgets:
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "get_widget_data",
                            "description": "Retrieve data for a widget by specifying the widget UUID.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "widget_uuid": {
                                        "type": "string",
                                        "description": "The UUID of the widget to retrieve data from.",
                                    }
                                },
                                "required": ["widget_uuid"],
                            },
                        },
                    }
                )

            # Get API key
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                yield reasoning_step(
                    event_type="ERROR",
                    message="Missing API key for web search capabilities.",
                    details={"error": "OPENROUTER_API_KEY not set"},
                ).model_dump()
                yield {
                    "event": "error",
                    "data": json.dumps({"message": "OPENROUTER_API_KEY not set"}),
                }
                return

            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://pro.openbb.dev",
                "X-Title": "OpenBB Terminal Pro",
            }

            # Making initial LLM request
            yield reasoning_step(
                event_type="INFO",
                message="Analyzing your query and determining the best approach...",
            ).model_dump()

            # Start with non-streaming call to detect tool calls
            data = {
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": formatted_messages,
                "stream": False,  # Start with non-streaming
                "tools": tools,
                "tool_choice": "auto",
            }

            logger.info(
                f"Making initial request to detect tool calls with messages: {formatted_messages}"
            )
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Initial response: {result}")

                # Check for tool calls
                choices = result.get("choices", [])
                finish_reason = choices[0].get("finish_reason") if choices else None

                # Check if there are tool calls indicated by finish_reason
                if choices and (
                    finish_reason == "tool_calls"
                    or "tool_calls" in choices[0].get("message", {})
                ):
                    # If finish_reason is tool_calls but no tool_calls in message,
                    # we need to make another request to get the tool call information
                    yield reasoning_step(
                        event_type="INFO",
                        message="I need to search for information to answer your question properly.",
                    ).model_dump()

                    if finish_reason == "tool_calls" and "tool_calls" not in choices[
                        0
                    ].get("message", {}):
                        # Request with stream=False and respond_to_tool_calls=true to get tool calls
                        tool_data = {
                            "model": "deepseek/deepseek-chat-v3-0324",
                            "messages": formatted_messages,
                            "stream": False,
                            "tools": tools,
                            "tool_choice": "auto",
                        }
                        logger.info("Making follow-up request to get tool call details")
                        tool_response = await client.post(
                            url, headers=headers, json=tool_data
                        )
                        tool_response.raise_for_status()
                        tool_result = tool_response.json()
                        logger.info(f"Tool call response: {tool_result}")

                        if "tool_calls" in tool_result.get("choices", [{}])[0].get(
                            "message", {}
                        ):
                            tool_calls = tool_result["choices"][0]["message"][
                                "tool_calls"
                            ]
                            logger.info(f"Tool calls detected: {tool_calls}")

                            yield reasoning_step(
                                event_type="INFO",
                                message="Found relevant information sources to check.",
                            ).model_dump()
                        else:
                            logger.warning(
                                "Couldn't retrieve tool calls even after follow-up request"
                            )
                            # Fall back to regular content streaming
                            yield {
                                "event": "copilotMessageChunk",
                                "data": json.dumps(
                                    {
                                        "delta": "I couldn't retrieve the information you requested. Please try asking your question differently."
                                    }
                                ),
                            }
                            yield {
                                "event": "copilotMessageChunk",
                                "data": json.dumps({"delta": ""}),
                            }
                            return
                    else:
                        tool_calls = choices[0]["message"]["tool_calls"]
                        logger.info(f"Tool calls detected: {tool_calls}")

                    # Process each tool call
                    for tool_call in tool_calls:
                        if tool_call["function"]["name"] == "perplexity_web_search":
                            try:
                                # Extract query
                                args = json.loads(tool_call["function"]["arguments"])
                                query = args.get("query", "")
                                logger.info(f"Extracted query: {query}")

                                # Inform user we're searching
                                yield reasoning_step(
                                    event_type="INFO",
                                    message=f"Searching the web for: {query}",
                                    details={"search_query": query},
                                ).model_dump()

                                # Call perplexity
                                yield reasoning_step(
                                    event_type="INFO",
                                    message="Connecting to search service and retrieving results...",
                                ).model_dump()

                                search_result = await perplexity_web_search(query)
                                logger.info(f"Search result: {search_result[:100]}...")

                                yield reasoning_step(
                                    event_type="INFO",
                                    message="Search completed successfully, processing results.",
                                    details={"result_length": len(search_result)},
                                ).model_dump()

                                # Add result to messages
                                new_messages = formatted_messages.copy()
                                new_messages.append(
                                    {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": tool_call["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": "perplexity_web_search",
                                                    "arguments": json.dumps(
                                                        {"query": query}
                                                    ),
                                                },
                                            }
                                        ],
                                    }
                                )
                                new_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call["id"],
                                        "content": search_result,
                                    }
                                )

                                # Get final response with streaming
                                final_data = {
                                    "model": "deepseek/deepseek-chat-v3-0324",
                                    "messages": new_messages,
                                    "stream": True,
                                }

                                yield reasoning_step(
                                    event_type="INFO",
                                    message="Retrieved information from the web, now formulating a response.",
                                ).model_dump()

                                logger.info(
                                    f"Making final streaming request with messages: {new_messages}"
                                )
                                async with client.stream(
                                    "POST", url, headers=headers, json=final_data
                                ) as stream_response:
                                    stream_response.raise_for_status()

                                    async for line in stream_response.aiter_lines():
                                        if not line or not line.startswith("data: "):
                                            continue

                                        line = line[6:].strip()
                                        if line == "[DONE]":
                                            break

                                        try:
                                            chunk = json.loads(line)
                                            content = (
                                                chunk.get("choices", [{}])[0]
                                                .get("delta", {})
                                                .get("content")
                                            )

                                            if content:  # Simplified check, empty strings are falsy
                                                yield {
                                                    "event": "copilotMessageChunk",
                                                    "data": json.dumps(
                                                        {"delta": content}
                                                    ),
                                                }
                                        except json.JSONDecodeError as e:
                                            logger.error(
                                                f"JSON decode error: {e} for line: {line}"
                                            )
                                            yield reasoning_step(
                                                event_type="WARNING",
                                                message="Encountered an issue processing part of the response.",
                                                details={
                                                    "error_type": "JSON decode error"
                                                },
                                            ).model_dump()
                                            continue

                                # Signal end of response
                                yield {
                                    "event": "copilotMessageChunk",
                                    "data": json.dumps({"delta": ""}),
                                }
                                return  # Important to return here to prevent falling through

                            except Exception as e:
                                logger.error(
                                    f"Error processing web search: {str(e)}",
                                    exc_info=True,
                                )
                                yield reasoning_step(
                                    event_type="ERROR",
                                    message="Error occurred while searching the web.",
                                    details={"error": str(e)},
                                ).model_dump()
                                yield {
                                    "event": "error",
                                    "data": json.dumps(
                                        {"message": f"Error with web search: {str(e)}"}
                                    ),
                                }
                                return

                        # Handle widget data retrieval if it's a get_widget_data tool call
                        elif tool_call["function"]["name"] == "get_widget_data":
                            try:
                                # Extract widget UUID
                                args = json.loads(tool_call["function"]["arguments"])
                                widget_uuid = args.get("widget_uuid", "")
                                logger.info(
                                    f"Retrieving data for widget UUID: {widget_uuid}"
                                )

                                # Find the requested widget
                                if not request.widgets:
                                    yield reasoning_step(
                                        event_type="ERROR",
                                        message="No widgets available to retrieve data from.",
                                        details={"widget_uuid": widget_uuid},
                                    ).model_dump()
                                    yield {
                                        "event": "error",
                                        "data": json.dumps(
                                            {"message": "No widgets available"}
                                        ),
                                    }
                                    return

                                all_widgets = []
                                if request.widgets.primary:
                                    all_widgets.extend(request.widgets.primary)
                                if request.widgets.secondary:
                                    all_widgets.extend(request.widgets.secondary)

                                matching_widgets = list(
                                    filter(
                                        lambda widget: str(widget.uuid) == widget_uuid,
                                        all_widgets,
                                    )
                                )
                                widget = (
                                    matching_widgets[0] if matching_widgets else None
                                )

                                if not widget:
                                    yield reasoning_step(
                                        event_type="ERROR",
                                        message=f"Widget with UUID {widget_uuid} not found",
                                        details={"widget_uuid": widget_uuid},
                                    ).model_dump()
                                    yield {
                                        "event": "error",
                                        "data": json.dumps(
                                            {
                                                "message": f"Widget with UUID {widget_uuid} not found"
                                            }
                                        ),
                                    }
                                    return

                                # Inform user we're retrieving widget data
                                yield reasoning_step(
                                    event_type="INFO",
                                    message=f"Retrieving data for widget: {widget.name}",
                                    details={
                                        "widget_uuid": widget_uuid,
                                        "widget_name": widget.name,
                                    },
                                ).model_dump()

                                # Create a FunctionCallSSE object to request the data from the frontend
                                widget_data_request = FunctionCallSSE(
                                    event="copilotFunctionCall",
                                    data=FunctionCallSSEData(
                                        function="get_widget_data",
                                        input_arguments={
                                            "data_sources": [
                                                {
                                                    "origin": widget.origin,
                                                    "id": widget.widget_id,
                                                    "input_args": {
                                                        param.name: param.current_value
                                                        for param in widget.params
                                                    },
                                                }
                                            ]
                                        },
                                        extra_state={
                                            "copilot_function_call_arguments": {
                                                "widget_uuid": widget_uuid
                                            }
                                        },
                                    ),
                                )

                                yield widget_data_request.model_dump()
                                return  # Must return here to allow the frontend to handle the request

                            except Exception as e:
                                logger.error(
                                    f"Error processing widget data request: {str(e)}",
                                    exc_info=True,
                                )
                                yield reasoning_step(
                                    event_type="ERROR",
                                    message="Error occurred while retrieving widget data.",
                                    details={"error": str(e)},
                                ).model_dump()
                                yield {
                                    "event": "error",
                                    "data": json.dumps(
                                        {"message": f"Error with widget data: {str(e)}"}
                                    ),
                                }
                                return

                # No tool calls detected, just stream the normal response
                content = ""
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    logger.info(
                        f"No tool calls, streaming regular content: {content[:100]}..."
                    )

                    yield reasoning_step(
                        event_type="INFO",
                        message="Found the answer directly without needing to search external sources.",
                    ).model_dump()

                # Stream the content in small chunks to simulate streaming
                if not content:
                    logger.warning("No content to stream in the response")
                    yield reasoning_step(
                        event_type="WARNING",
                        message="The model didn't generate any content for your query.",
                    ).model_dump()
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {"message": "No content received from model"}
                        ),
                    }
                    return

                # Stream in larger chunks for better performance
                chunk_size = 500  # Larger chunks for faster streaming
                for i in range(0, len(content), chunk_size):
                    chunk = content[i : i + chunk_size]
                    yield {
                        "event": "copilotMessageChunk",
                        "data": json.dumps({"delta": chunk}),
                    }

                # Signal end of response
                yield {
                    "event": "copilotMessageChunk",
                    "data": json.dumps({"delta": ""}),
                }

        except Exception as e:
            logger.error(f"Error in direct_response: {str(e)}", exc_info=True)
            yield reasoning_step(
                event_type="ERROR",
                message="Encountered an unexpected error while processing your request.",
                details={"error": str(e)},
            ).model_dump()
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    # Use our simpler direct approach
    return EventSourceResponse(
        content=direct_response(),
        media_type="text/event-stream",
    )
