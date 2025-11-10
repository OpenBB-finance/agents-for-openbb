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

    # Debug: Print all messages to understand the flow
    print(f"[DEBUG] Received request with {len(request.messages)} messages:")
    for i, msg in enumerate(request.messages):
        content_preview = "NO CONTENT"
        if hasattr(msg, "content") and msg.content:
            try:
                if isinstance(msg.content, str):
                    content_preview = msg.content[:100]
                else:
                    content_preview = str(msg.content)[:100]
            except Exception as e:
                content_preview = f"ERROR_READING_CONTENT: {e}"
        print(f"[DEBUG] Message {i}: role='{msg.role}', content='{content_preview}...'")
        if hasattr(msg, "function") and msg.function:
            print(f"[DEBUG] Message {i} has function: {msg.function}")
        if hasattr(msg, "data") and msg.data:
            print(f"[DEBUG] Message {i} has data: {len(msg.data)} items")
            for j, data_item in enumerate(msg.data):
                if hasattr(data_item, "items") and data_item.items:
                    print(f"[DEBUG] Data {j}: {len(data_item.items)} items")
                    for k, item in enumerate(data_item.items):
                        item_content = getattr(item, "content", "NO ITEM CONTENT")
                        print(f"[DEBUG] Item {k}: {item_content[:200]}...")
                else:
                    print(f"[DEBUG] Data {j}: {data_item}")

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
        print(f"[DEBUG] Available MCP tools: {len(request.tools)} tools found")
        system_content += "\n\nYou have access to the following MCP tools:\n"
        for tool in request.tools:
            server_id = getattr(tool, "server_id", "unknown")
            print(f"[DEBUG] Tool: {tool.name}, Server ID: {server_id}")
            print(f"[DEBUG] Tool URL: {getattr(tool, 'url', 'NO_URL')}")
            print(f"[DEBUG] Tool endpoint: {getattr(tool, 'endpoint', 'NO_ENDPOINT')}")
            print(f"[DEBUG] Tool input_schema: {tool.input_schema}")
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
            context_str += (
                "IMPORTANT: You MUST structure your response exactly as follows:\n\n"
            )
            context_str += "## MCP OUTPUT\n"
            for result in message.data:
                for item in result.items:
                    context_str += f"{item.content}\n\n"
            context_str += "## AI OUTPUT\n"
            context_str += "Now provide your analysis and answer to the user's question based on the MCP output above.\n\n"

    if context_str:
        openai_messages[-1]["content"] += "\n\n" + context_str  # type: ignore

    # Debug: Print the final OpenAI messages
    print(f"[DEBUG] Sending {len(openai_messages)} messages to OpenAI:")
    for i, msg in enumerate(openai_messages):
        print(
            f"[DEBUG] OpenAI Message {i}: role='{msg['role']}', content_length={len(str(msg.get('content', '')))}"
        )

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
            print("[DEBUG] Continuing conversation with tool results - NO FUNCTIONS")

            # Use streaming for the final response WITHOUT function calling
            async for event in await client.chat.completions.create(
                model="gpt-4o",
                messages=openai_messages,
                stream=True,
                # Don't pass functions here to prevent another tool call
            ):
                if chunk := event.choices[0].delta.content:
                    yield message_chunk(chunk).model_dump()
            return

        # Check if we need function calling
        if functions:
            # Use non-streaming for function calls
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=openai_messages,
                functions=functions,
                stream=False,
            )

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

                    print(
                        f"[DEBUG] Executing MCP tool: server_id='{server_id}', tool_name='{tool_name}'"
                    )
                    print(f"[DEBUG] Tool parameters: {parameters}")
                    print(f"[DEBUG] Sending to frontend with tool_name: '{tool_name}'")

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
