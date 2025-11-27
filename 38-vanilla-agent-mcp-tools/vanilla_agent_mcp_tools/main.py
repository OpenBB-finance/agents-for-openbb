from typing import AsyncGenerator
import json

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
from openbb_ai import WidgetRequest, get_widget_data, message_chunk
from openbb_ai.models import (
    MessageChunkSSE,
    QueryRequest,
    FunctionCallSSE,
    FunctionCallSSEData,
)
from sse_starlette.sse import EventSourceResponse

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
            "vanilla_agent_mcp": {
                "name": "Vanilla Agent with MCP",
                "description": "A vanilla agent that supports MCP tools and automatically retrieves widget data.",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {"query": "/v1/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,
                    "widget-dashboard-search": False,
                    "mcp-tools": True,
                },
            }
        }
    )


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Copilot with MCP tools support."""


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
        # Note: If we wanted to iterate through the widgets on the dashboard
        # rather than on the widgets on the explicit context
        # then we would need to iterate through request.widgets.secondary
        # and the agents.json would need "widget-dashboard-search": True
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
    system_content = (
        "You are a helpful financial assistant. Your name is 'Vanilla Agent'."
    )

    # Add MCP tools to system prompt if available
    if request.tools:
        system_content += "\n\nYou have access to the following MCP tools:\n"
        for tool in request.tools:
            server_id = getattr(tool, "server_id", "unknown")
            system_content += f"- Tool: {tool.name} (Server ID: {server_id})\n"
            system_content += f"  Description: {tool.description}\n"
            if hasattr(tool, "input_schema") and tool.input_schema:
                system_content += f"  Parameters: {tool.input_schema}\n"
                # Add parameter details to help LLM understand what to pass
                if (
                    isinstance(tool.input_schema, dict)
                    and "properties" in tool.input_schema
                ):
                    system_content += f"  Required parameters: {tool.input_schema.get('properties', {}).keys()}\n"
        system_content += "\nUse the execute_agent_tool function to call these tools. When calling, make sure to:\n"
        system_content += "1. Use the exact Server ID and tool name as shown above\n"
        system_content += "2. Include all required parameters in the 'parameters' field based on the tool's schema\n"
        system_content += "3. After receiving tool results, answer the user's question directly without calling tools again."

    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=system_content,
        )
    ]

    # Prepare MCP function definitions for OpenAI
    functions = []
    if request.tools:
        # Create a single execute_agent_tool function that can call any MCP tool
        functions.append(
            {
                "name": "execute_agent_tool",
                "description": "Execute an MCP tool to retrieve data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "server_id": {
                            "type": "string",
                            "description": "The ID of the MCP server",
                            "enum": list(
                                set(
                                    getattr(tool, "server_id", "unknown")
                                    for tool in request.tools
                                )
                            ),
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "The name of the tool to execute",
                            "enum": [
                                tool.name for tool in request.tools
                            ],  # Use full tool names
                        },
                        "parameters": {
                            "type": "object",
                            "description": "The arguments to pass to the tool. Use the parameter schema defined for each specific tool.",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["server_id", "tool_name", "parameters"],
                },
            }
        )

    # Helper function to truncate content if too long
    def truncate_content(content: str, max_chars: int = 1000000) -> str:
        """Truncate content to avoid hitting OpenAI's token limits."""
        if len(content) > max_chars:
            # Keep first and last portions with ellipsis in middle
            keep_start = max_chars // 2
            keep_end = max_chars // 4
            return f"{content[:keep_start]}\n\n... [Content truncated due to size limits] ...\n\n{content[-keep_end:]}"
        return content

    context_str = ""
    for index, message in enumerate(request.messages):
        if message.role == "human":
            # Truncate human messages if they're too long
            content = truncate_content(message.content)
            openai_messages.append(
                ChatCompletionUserMessageParam(role="user", content=content)
            )
        elif message.role == "ai":
            if isinstance(message.content, str):
                # Truncate AI messages if they're too long
                content = truncate_content(message.content)
                openai_messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=content
                    )
                )
        # We only add the most recent tool call / widget data to context.  We do
        # this **only for this particular example** to prevent
        # previously-retrieved widget data from piling up and exceeding the
        # context limit of the LLM.
        elif message.role == "tool" and index == len(request.messages) - 1:
            context_str += (
                "IMPORTANT: You MUST structure your response exactly as follows:\n\n"
            )
            context_str += "## MCP OUTPUT\n"
            tool_content = ""
            for result in message.data:
                for item in result.items:
                    tool_content += f"{item.content}\n\n"
            # Truncate tool output if too long
            tool_content = truncate_content(tool_content, max_chars=500000)
            context_str += tool_content
            context_str += "## AI OUTPUT\n"
            context_str += "Now provide your analysis and answer to the user's question based on the MCP output above.\n\n"

    if context_str:
        # Truncate the entire context before adding to prevent exceeding limits
        context_str = truncate_content(context_str, max_chars=500000)
        openai_messages[-1]["content"] += "\n\n" + context_str  # type: ignore

    # Define the execution loop with MCP support
    async def execution_loop() -> (
        AsyncGenerator[MessageChunkSSE | FunctionCallSSE, None]
    ):
        client = openai.AsyncOpenAI()

        # Check if the last message contains tool results (from MCP execution)
        last_message = request.messages[-1] if request.messages else None
        if last_message and last_message.role == "tool":
            # We have tool results, continue the conversation with the LLM
            # The tool results are already added to context_str above

            # Use streaming for the final response WITHOUT function calling
            try:
                async for event in await client.chat.completions.create(
                    model="gpt-4o",
                    messages=openai_messages,
                    stream=True,
                    # Don't pass functions here to prevent another tool call
                ):
                    if chunk := event.choices[0].delta.content:
                        yield message_chunk(chunk).model_dump()
            except openai.BadRequestError as e:
                error_msg = f"Error: Content too large for OpenAI API. {str(e)[:200]}"
                yield message_chunk(error_msg).model_dump()
            return

        # Check if we need function calling
        if functions:
            # Use non-streaming for function calls
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=openai_messages,
                    functions=functions,
                    stream=False,
                )
            except openai.BadRequestError as e:
                error_msg = f"Error: Content too large for OpenAI API. {str(e)[:200]}"
                yield message_chunk(error_msg).model_dump()
                return

            message = response.choices[0].message

            # Handle function calls
            if (
                message.function_call
                and message.function_call.name == "execute_agent_tool"
            ):
                try:
                    # Parse function arguments
                    args = json.loads(message.function_call.arguments)

                    server_id = args.get("server_id", "")
                    tool_name = args.get("tool_name", "")
                    parameters = args.get("parameters", {})

                    # Send function call back to frontend for MCP execution
                    function_call_data = FunctionCallSSEData(
                        function="execute_agent_tool",
                        input_arguments={
                            "server_id": server_id,
                            "tool_name": tool_name,  # Send the full tool name as received
                            "parameters": parameters,
                        },
                        extra_state={
                            "copilot_function_call_arguments": {
                                "server_id": server_id,
                                "tool_name": tool_name,
                                "tool_args": parameters,
                                "summary": f"Execute {tool_name} MCP tool",
                            }
                        },
                    )

                    yield FunctionCallSSE(data=function_call_data).model_dump()
                    return  # Return control to frontend

                except json.JSONDecodeError:
                    # Fallback to regular response if function parsing fails
                    pass

            # If no function call or function call failed, stream the response
            if message.content:
                # Stream each character of the content
                for char in message.content:
                    yield message_chunk(char).model_dump()
        else:
            # Regular streaming without function calls
            try:
                async for event in await client.chat.completions.create(
                    model="gpt-4o",
                    messages=openai_messages,
                    stream=True,
                ):
                    if chunk := event.choices[0].delta.content:
                        yield message_chunk(chunk).model_dump()
            except openai.BadRequestError as e:
                error_msg = f"Error: Content too large for OpenAI API. {str(e)[:200]}"
                yield message_chunk(error_msg).model_dump()

    # Stream the SSEs back to the client.
    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
