from typing import AsyncGenerator
from common.agent import (
    get_remote_widget_data,
    reasoning_step,
)
from common.models import (
    QueryRequest,
    DataContent,
    FunctionCallSSE,
    StatusUpdateSSE,
)

from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

GET_WIDGET_DATA_FUNCTION_DEFINITION = ChatCompletionToolParam(
    type="function",
    function=FunctionDefinition(
        name="get_widget_data",
        description="Retrieves data for the given widget.",
        parameters={
            "type": "object",
            "properties": {
                "widget_uuid": {
                    "type": "string",
                    "description": "The UUID of the widget to retrieve data for.",
                },
            },
            "required": [
                "widget_uuid",
            ],
            "additionalProperties": False,
        },
        strict=True,
    ),
)


async def handle_widget_data(data: list[DataContent]) -> str:
    result_str = "--- Data ---\n"
    for result in data:
        for item in result.items:
            result_str += f"{item.content}\n"
            result_str += "------\n"
    return result_str


async def get_widget_data(
    widget_uuid: str,
    request: QueryRequest,  # Must be included as an argument
) -> AsyncGenerator[FunctionCallSSE | StatusUpdateSSE, None]:
    """Retrieve data for a widget by specifying the widget UUID."""

    widgets = (
        request.widgets.primary + request.widgets.secondary if request.widgets else []
    )

    # Get the first widget that matches the UUID (there should be only one).
    matching_widgets = list(
        filter(lambda widget: str(widget.uuid) == widget_uuid, widgets)
    )
    widget = matching_widgets[0] if matching_widgets else None

    # If we're unable to find the widget, let's the the user know.
    if not widget:
        yield reasoning_step(
            event_type="ERROR",
            message="Unable to retrieve data for widget (does not exist)",
            details={"widget_uuid": widget_uuid},
        )
        # And let's also return a message to the LLM.
        yield f"Unable to retrieve data for widget with UUID: {widget_uuid} (it is not present on the dashboard)"  # noqa: E501
        return

    # Let's let the user know that we're retrieving data for the widget.
    yield reasoning_step(
        event_type="INFO",
        message=f"Retrieving data for widget: {widget.name}...",
        details={"widget_uuid": widget_uuid},
    )

    # Let's yield the actual function call to the front-end.
    yield get_remote_widget_data(
        widget=widget,
        # In this example we will just re-use the currently-set values of
        # the widget parameters.
        input_arguments={param.name: param.current_value for param in widget.params},
    )
