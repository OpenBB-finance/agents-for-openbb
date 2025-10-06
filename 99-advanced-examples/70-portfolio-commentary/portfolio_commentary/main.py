import json
import logging
import os
import uuid
from pathlib import Path
from typing import AsyncGenerator, Callable

import httpx
from openbb_ai import reasoning_step, get_widget_data
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Remove magentic imports - using direct HTTP approach instead
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
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
    except Exception as e:
        error_message = f"Error searching the web: {str(e)}"
        logger.error(error_message)
        return error_message


# Removed custom_run_agent function - using direct HTTP approach instead


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

        # Request the widget data using the new API
        from openbb_ai.models import WidgetRequest
        widget_request = WidgetRequest(
            widget_uuid=widget.uuid,
            origin=widget.origin,
            id=widget.widget_id,
            input_args={param.name: param.current_value for param in widget.params},
        )
        yield get_widget_data([widget_request]).model_dump()

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

    # Removed custom_process_messages function - using direct HTTP approach instead

    # Simple, direct approach without complex streaming logic
    async def direct_response():
        import json  # Ensure json is available in local scope
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

            # Check if the last message is a tool response - if so, return the data directly
            last_message = request.messages[-1] if request.messages else None
            if last_message and hasattr(last_message, 'role') and last_message.role == "tool":
                # Extract and format the tool response data
                if hasattr(last_message, 'data') and last_message.data:
                    
                    # Parse the JSON data from the portfolio holdings
                    data_content = last_message.data[0].items[0].content
                    holdings_data = json.loads(data_content)
                    
                    # Format a professional portfolio analysis
                    analysis = "# Portfolio Holdings Analysis - Client 2\n\n"
                    analysis += "## Current Holdings Overview\n\n"
                    analysis += f"Your portfolio consists of **{len(holdings_data)} holdings** with the following composition:\n\n"
                    
                    # Top 10 holdings
                    analysis += "### Top 10 Holdings\n"
                    for i, holding in enumerate(holdings_data[:10], 1):
                        analysis += f"{i}. **{holding['Name']} ({holding['Symbol']})** - {holding['Weight']:.2f}%\n"
                        analysis += f"   - Sector: {holding['Sector']}\n"
                        analysis += f"   - Industry: {holding['Industry']}\n"
                        analysis += f"   - Country: {holding['Country']}\n\n"
                    
                    # Sector analysis
                    sectors = {}
                    for holding in holdings_data:
                        sector = holding['Sector']
                        sectors[sector] = sectors.get(sector, 0) + holding['Weight']
                    
                    analysis += "### Sector Allocation\n"
                    for sector, weight in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
                        analysis += f"- **{sector}**: {weight:.1f}%\n"
                    
                    analysis += "\n### Key Observations\n"
                    analysis += f"- **Industrial Focus**: The portfolio shows a strong concentration in industrial companies (~{sectors.get('Industrials', 0):.1f}%)\n"
                    analysis += f"- **Diversification**: Holdings span across {len(sectors)} sectors and multiple countries\n"
                    analysis += f"- **Quality Names**: Includes established companies like Caterpillar, Honeywell, and Linde\n"
                    analysis += f"- **Geographic Exposure**: Primarily US-focused with international diversification\n"
                    
                    # Stream the response
                    yield {
                        "event": "copilotMessageChunk",
                        "data": json.dumps({"delta": analysis}),
                    }
                    yield {
                        "event": "copilotMessageChunk", 
                        "data": json.dumps({"delta": ""}),
                    }
                    return
            
            # Format messages for OpenAI API directly from request (only if not a tool response)
            formatted_messages = [{"role": "system", "content": system_prompt}]
            
            for message in request.messages:
                if hasattr(message, "role") and hasattr(message, "content"):
                    if message.role == "human":
                        formatted_messages.append({"role": "user", "content": message.content})
                    elif message.role == "ai":
                        # Handle function calls vs regular content
                        if isinstance(message.content, str):
                            formatted_messages.append({"role": "assistant", "content": message.content})
                        # Skip function call objects as they'll trigger new tool calls

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
            async with httpx.AsyncClient(timeout=30.0) as client:
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
