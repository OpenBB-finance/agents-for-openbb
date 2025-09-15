from typing import AsyncGenerator
import openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai.models import MessageChunkSSE, QueryRequest
from openbb_ai import get_widget_data, WidgetRequest, message_chunk

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
    """Copilot descriptor for OpenBB Workspace."""
    return JSONResponse(
        content={
            "vanilla_agent_dashboard_widgets": {
                "name": "Vanilla Agent Dashboard Widgets",
                "description": "Lists dashboard widgets and retrieves data from the first widget on the dashboard (if any).",
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
    
    # Helper to get all widgets
    def get_all_widgets():
        widgets = []
        if request.widgets:
            if request.widgets.primary:
                widgets.extend(request.widgets.primary)
            if request.widgets.secondary:
                widgets.extend(request.widgets.secondary)
            if request.widgets.extra:
                widgets.extend(request.widgets.extra)
        return widgets

    # Always show widget list and fetch last widget data on human messages
    if request.messages[-1].role == "human":
        all_widgets = get_all_widgets()
        
        # Build widget list to show
        widget_list_msg = ""
        if request.workspace_state and request.workspace_state.current_dashboard_info:
            tabs = request.workspace_state.current_dashboard_info.tabs
            active_tab = request.workspace_state.current_dashboard_info.current_tab_id
            
            if tabs:
                widget_list_msg = "Widgets on your current dashboard:\n\n"
                if active_tab:
                    widget_list_msg += f"Active tab: {active_tab}\n\n"
                
                for tab in tabs:
                    widget_list_msg += f"[Tab: {tab.tab_id}]\n"
                    if tab.widgets:
                        for widget in tab.widgets:
                            widget_list_msg += f"  - {widget.name}\n"
                    else:
                        widget_list_msg += "  (no widgets)\n"
                    widget_list_msg += "\n"
        elif all_widgets:
            widget_list_msg = "Available widgets:\n\n"
            for w in all_widgets:
                widget_list_msg += f"  - {w.name or w.widget_id or 'Unnamed Widget'}\n"
            widget_list_msg += "\n"
        
        # Stream widget list and then fetch last widget data
        async def show_widgets_and_fetch():
            # Show the widget list (first message completes here)
            if widget_list_msg:
                yield message_chunk(widget_list_msg.rstrip()).model_dump()
            
            # Then fetch data from the last widget if available
            if all_widgets:
                last_widget = all_widgets[-1]
                widget_requests = [
                    WidgetRequest(
                        widget=last_widget,
                        input_arguments={
                            param.name: param.current_value for param in last_widget.params
                        },
                    )
                ]
                yield get_widget_data(widget_requests).model_dump()
            elif not widget_list_msg:
                yield message_chunk("No widgets found on your dashboard.").model_dump()
        
        # Return early with widget info and data fetch
        return EventSourceResponse(
            content=show_widgets_and_fetch(),
            media_type="text/event-stream",
        )

    # Check if we just received tool data - if so, show a sample and continue conversation
    if request.messages and request.messages[-1].role == "tool":
        async def show_data_sample():
            # Extract widget name and data
            widget_name = "Unknown Widget"
            all_widgets = get_all_widgets()
            if all_widgets:
                last_widget = all_widgets[-1]
                widget_name = last_widget.name or last_widget.widget_id or 'Unnamed'
            
            # Extract data content
            data_content = ""
            for result in request.messages[-1].data:
                for item in result.items:
                    data_content = item.content
                    break
                if data_content:
                    break
            
            if data_content:
                sample = data_content[:500] + "..." if len(data_content) > 500 else data_content
                yield message_chunk(f"Fetching sample data from last widget: {widget_name}\n\nSample of widget data:\n```\n{sample}\n```").model_dump()
        
        return EventSourceResponse(
            content=show_data_sample(),
            media_type="text/event-stream",
        )
    
    # Format the messages into a list of OpenAI messages
    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful financial assistant. Your name is 'Dashboard Widget Agent'.",
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
        # Add widget data to context if it's a tool message (most recent only)
        elif message.role == "tool" and index == len(request.messages) - 1:
            context_str += "Use the following data to answer the question:\n\n"
            result_str = "--- Data ---\n"
            for result in message.data:
                for item in result.items:
                    result_str += f"{item.content}\n"
                    result_str += "------\n"
            context_str += result_str

    # Build comprehensive widget listing
    widget_list = ""
    all_widgets = get_all_widgets()
    
    # Show widgets organized by tabs if available
    if request.workspace_state and request.workspace_state.current_dashboard_info:
        tabs = request.workspace_state.current_dashboard_info.tabs
        active_tab = request.workspace_state.current_dashboard_info.current_tab_id
        
        if tabs:
            widget_list = "Widgets on your current dashboard:\n\n"
            if active_tab:
                widget_list += f"Active tab: {active_tab}\n\n"
            
            for tab in tabs:
                widget_list += f"[Tab: {tab.tab_id}]\n"
                if tab.widgets:
                    for widget in tab.widgets:
                        widget_list += f"  - {widget.name}\n"
                else:
                    widget_list += "  (no widgets)\n"
                widget_list += "\n"
    elif all_widgets:
        # Fallback: list all widgets if no tab info
        widget_list = "Available widgets:\n\n"
        for w in all_widgets:
            widget_list += f"  - {w.name or w.widget_id or 'Unnamed Widget'}\n"
        widget_list += "\n"

    # Add widget list and data context to the last user message
    if widget_list or context_str:
        full_context = ""
        if widget_list:
            full_context += widget_list
        if context_str:
            if all_widgets:
                last_widget = all_widgets[-1]
                full_context += (f"Data from last widget "
                                f"({last_widget.name or last_widget.widget_id or 'Unnamed'}):\n\n")
            full_context += context_str
        openai_messages[-1]["content"] += "\n\n" + full_context  # type: ignore

    # Define the execution loop to stream LLM response
    async def execution_loop() -> AsyncGenerator[MessageChunkSSE, None]:
        client = openai.AsyncOpenAI()
        async for event in await client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            stream=True,
        ):
            if chunk := event.choices[0].delta.content:
                yield message_chunk(chunk).model_dump()

    # Stream the SSEs back to the client
    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream"
    )
