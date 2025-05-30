import inspect
import json
import os
from asyncstdlib import tee
from magentic import (
    AsyncStreamedResponse,
    Chat,
    AsyncStreamedStr,
    FunctionCall,
    OpenaiChatModel,
    ParallelFunctionCall,
)
from google import genai
from openbb_ai.models import (
    ClientFunctionCallError,
    QueryRequest,
    Citation,
    CitationCollection,
    CitationCollectionSSE,
    DataContent,
    DataFileReferences,
    DataSourceRequest,
    FunctionCallSSE,
    FunctionCallSSEData,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    MessageChunkSSE,
    MessageChunkSSEData,
    SourceInfo,
    StatusUpdateSSE,
    StatusUpdateSSEData,
    Widget,
)
from magentic import (
    AssistantMessage,
    FunctionResultMessage,
    AnyMessage,
    SystemMessage,
    UserMessage,
)
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterable,
    Literal,
    Protocol,
    cast,
)
import re
from openai import NOT_GIVEN, AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.shared_params import FunctionDefinition as OpenAiFunctionDefinition
from magentic.chat_model.function_schema import FunctionCallFunctionSchema

import logging

logger = logging.getLogger(__name__)


def sanitize_message(message: str) -> str:
    """Sanitize a message by escaping forbidden characters."""
    cleaned_message = re.sub(r"(?<!\{)\{(?!{)", "{{", message)
    cleaned_message = re.sub(r"(?<!\})\}(?!})", "}}", cleaned_message)
    return cleaned_message


def reasoning_step(
    event_type: Literal["INFO", "WARNING", "ERROR"],
    message: str,
    details: dict[str, Any] | None = None,
) -> StatusUpdateSSE:
    return StatusUpdateSSE(
        data=StatusUpdateSSEData(
            eventType=event_type,
            message=message,
            details=[details] if details else [],
        )
    )


class WrappedFunctionProtocol(Protocol):
    async def execute_post_processing(
        self, data: list[DataContent | DataFileReferences | ClientFunctionCallError]
    ) -> str: ...
    def execute_callbacks(
        self,
        function_call_result: LlmClientFunctionCallResultMessage,
        request: QueryRequest,
    ) -> AsyncGenerator[Any, None]: ...

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[FunctionCallSSE | StatusUpdateSSE, None]: ...


def remote_function_call(
    function: Literal["get_widget_data"],
    output_formatter: Callable[..., Awaitable[str]] | None = None,
    callbacks: list[Callable[..., Awaitable[Any]]] | None = None,
) -> Callable:
    if function not in ["get_widget_data"]:
        raise ValueError(
            f"Unsupported function: {function}. Must be 'get_widget_data'."
        )

    def outer_wrapper(func: Callable) -> WrappedFunctionProtocol:
        class InnerWrapper(WrappedFunctionProtocol):
            def __init__(self):
                self.__name__ = func.__name__
                self.__signature__ = self._mask_signature(func)
                self.__doc__ = func.__doc__
                self.local_function = func
                self.function = function
                self.post_process_function = output_formatter
                self.callbacks = callbacks
                self._request = None

            @property
            def request(self) -> QueryRequest:
                return self._request

            @request.setter
            def request(self, request: QueryRequest):
                self._request = request

            def _mask_signature(self, func: Callable):
                """Hide the `request` argument from the signature, since we want
                to pass this in automatically, but not expose it to the LLM."""
                signature = inspect.signature(func)
                masked_params = [
                    p for p in signature.parameters.values() if p.name != "request"
                ]
                return signature.replace(parameters=masked_params)

            async def execute_callbacks(
                self,
                function_call_result: LlmClientFunctionCallResultMessage,
                request: QueryRequest,
            ) -> AsyncGenerator[Any, None]:
                if self.callbacks:
                    for callback in self.callbacks:
                        if inspect.isasyncgenfunction(callback):
                            async for event in callback(function_call_result, request):
                                yield event
                        else:
                            await callback(function_call_result, self.request)

            async def execute_post_processing(
                self,
                data: list[DataContent | DataFileReferences | ClientFunctionCallError],
            ) -> str:
                if self.post_process_function:
                    return await self.post_process_function(data)
                return str(data)

            async def __call__(
                self, *args, **kwargs
            ) -> AsyncGenerator[FunctionCallSSE | StatusUpdateSSE, None]:
                bound_args = self.__signature__.bind(*args, **kwargs).arguments

                async for event in func(*args, request=self._request, **kwargs):
                    if isinstance(event, StatusUpdateSSE):
                        yield event
                    elif isinstance(event, DataSourceRequest):
                        yield FunctionCallSSE(
                            data=FunctionCallSSEData(
                                function=self.function,
                                input_arguments={
                                    "data_sources": [
                                        DataSourceRequest(
                                            widget_uuid=event.widget_uuid,
                                            origin=event.origin,
                                            id=event.id,
                                            input_args=event.input_args,
                                        )
                                    ]
                                },
                                extra_state={
                                    "copilot_function_call_arguments": {
                                        **bound_args,
                                    },
                                    "_locally_bound_function": func.__name__,
                                },
                            )
                        )
                        return
                    else:
                        yield event

        return InnerWrapper()

    return outer_wrapper


def get_remote_widget_data(
    widget: Widget, input_arguments: dict[str, Any]
) -> FunctionCallSSE:
    return FunctionCallSSE(
        data=FunctionCallSSEData(
            function="get_widget_data",
            input_arguments={
                "data_sources": [
                    DataSourceRequest(
                        widget_uuid=str(widget.uuid),
                        origin=widget.origin,
                        id=widget.widget_id,
                        input_args=input_arguments,
                    )
                ],
            },
        )
    )


def get_remote_data(
    widget: Widget,
    input_arguments: dict[str, Any],
) -> DataSourceRequest:
    return DataSourceRequest(
        widget_uuid=str(widget.uuid),
        origin=widget.origin,
        id=widget.widget_id,
        input_args=input_arguments,
    )


def get_wrapped_function(
    function_name: str, functions: list[Any]
) -> WrappedFunctionProtocol:
    matching_local_functions = list(
        filter(
            lambda x: x.__name__ == function_name,
            functions,
        )
    )
    wrapped_function = matching_local_functions[0] if matching_local_functions else None
    if not wrapped_function:
        raise ValueError(f"Local function not found: {function_name}")
    return wrapped_function


async def process_messages(
    system_prompt: str,
    messages: list[LlmClientFunctionCallResultMessage | LlmClientMessage],
) -> list[AnyMessage] | list[ChatCompletionMessageParam]:
    return await _process_messages_openai(system_prompt, messages)


async def _process_messages_openai(
    system_prompt: str,
    messages: list[LlmClientFunctionCallResultMessage | LlmClientMessage],
) -> list[ChatCompletionMessageParam]:
    chat_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt)
    ]
    for message in messages:
        match message:
            case LlmClientMessage(role="human"):
                chat_messages.append(
                    ChatCompletionUserMessageParam(
                        role="user", content=cast(str, message.content)
                    )
                )
            case LlmClientMessage(role="ai") if isinstance(message.content, str):
                chat_messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=message.content
                    )
                )
            case _:
                raise ValueError(f"Unsupported message type: {message}")
    return chat_messages


class GeminiChat:
    def __init__(
        self,
        messages: list[AnyMessage],
        output_types: list[Any] | None = None,  # TODO: Implement this.
        functions: list[Callable] | None = None,
        model: str = "gemini-2.0-flash",
        vertex_ai: bool = False,
        project: str | None = None,
        location: str | None = None,
    ):
        self._messages = messages
        self._last_message: AnyMessage | None = None
        self._output_types = output_types
        self._functions = functions
        self._model = model

        if vertex_ai:
            if not project or not location:
                raise ValueError(
                    "project and location must be provided if vertex_ai is True"
                )
            self._client = genai.Client(
                vertexai=vertex_ai,
                project=project,
                location=location,
            )
        else:
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _get_system_prompt(self, messages: list[AnyMessage]) -> str:
        return next(m for m in messages if isinstance(m, SystemMessage)).content

    async def _convert_messages(
        self, messages: list[AnyMessage]
    ) -> list[genai.types.Content]:
        contents: list[genai.types.Content] = []
        for message in messages:
            if isinstance(message, UserMessage):
                contents.append(
                    genai.types.Content(
                        role="user", parts=[genai.types.Part(text=message.content)]
                    )
                )
            elif isinstance(message, AssistantMessage):
                if isinstance(message.content, str):
                    contents.append(
                        genai.types.Content(
                            role="model", parts=[genai.types.Part(text=message.content)]
                        )
                    )
                elif isinstance(message.content, AsyncStreamedResponse):
                    async for item in message.content:
                        if isinstance(item, FunctionCall):
                            contents.append(
                                genai.types.Content(
                                    role="model",
                                    parts=[
                                        genai.types.Part(
                                            function_call=genai.types.FunctionCall(
                                                name=item.function.__name__,
                                                args=item.arguments,
                                            )
                                        )
                                    ],
                                )
                            )

                elif isinstance(message.content, FunctionCall):
                    contents.append(
                        genai.types.Content(
                            role="model",
                            parts=[
                                genai.types.Part(
                                    function_call=genai.types.FunctionCall(
                                        name=message.content.function.__name__,
                                        args=message.content.arguments,
                                    )
                                )
                            ],
                        )
                    )
                elif isinstance(message.content, ParallelFunctionCall):
                    for function_call in message.content:
                        contents.append(
                            genai.types.Content(
                                role="model",
                                parts=[
                                    genai.types.Part(
                                        function_call=genai.types.FunctionCall(
                                            name=function_call.function.__name__,
                                            args=function_call.arguments,
                                        )
                                    )
                                ],
                            )
                        )
            elif isinstance(message, FunctionResultMessage):
                contents.append(
                    genai.types.Content(
                        role="user",
                        parts=[
                            genai.types.Part(
                                function_response=genai.types.FunctionResponse(
                                    name=message.function_call.function.__name__,
                                    response={"output": message.content},
                                )
                            )
                        ],
                    )
                )
        return contents

    def _get_function(self, function_name: str) -> Callable:
        for function in self._functions if self._functions else []:
            if function.__name__ == function_name:
                return function
        raise ValueError(f"Function not found: {function_name}")

    def add_message(self, message: AnyMessage) -> "GeminiChat":
        self._messages.append(message)
        return self

    def _prepare_tools(
        self, functions: list[Callable] | None
    ) -> list[genai.types.Tool] | None:
        from magentic.chat_model.function_schema import FunctionCallFunctionSchema

        if not functions:
            return None
        function_declarations: list[genai.types.FunctionDeclaration] = []
        for function in functions:
            schema = FunctionCallFunctionSchema(function)
            function_declarations.append(
                genai.types.FunctionDeclaration(
                    name=schema.name,
                    description=schema.description,
                    parameters=genai.types.Schema(**schema.parameters),
                )
            )
        return [genai.types.Tool(function_declarations=function_declarations)]

    async def asubmit(self) -> "GeminiChat":
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=await self._convert_messages(self._messages),  # type: ignore[arg-type]
            config=genai.types.GenerateContentConfig(
                system_instruction=self._get_system_prompt(self._messages),
                tools=self._prepare_tools(self._functions),  # type: ignore[arg-type]
                automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        async def async_streamed_response() -> (
            AsyncGenerator[FunctionCall | AsyncStreamedStr, None]
        ):
            async for event in stream:
                if function_calls := event.function_calls:
                    for function_call in function_calls:
                        yield FunctionCall(
                            function=self._get_function(function_call.name or ""),
                            **(function_call.args or {}),
                        )
                elif text := event.text:

                    async def async_streamed_str() -> AsyncGenerator[str, None]:
                        # Need to field the first chunk (it's not in the stream)
                        yield text
                        async for event in stream:
                            if event.text:
                                yield event.text
                            if (
                                event.candidates
                                and event.candidates[0].finish_reason == "STOP"
                            ):
                                if grounding_metadata := event.candidates[
                                    0
                                ].grounding_metadata:
                                    if (
                                        grounding_chunks
                                        := grounding_metadata.grounding_chunks
                                    ):
                                        yield "\n\nSources:\n"
                                        for grounding_chunk in grounding_chunks:
                                            yield "<br>"
                                            if web := grounding_chunk.web:
                                                yield f"<a href='{web.uri}'>{web.title}</a>"

                    yield AsyncStreamedStr(async_streamed_str())

        self.add_message(
            AssistantMessage(content=AsyncStreamedResponse(async_streamed_response()))
        )
        return self

    @property
    def last_message(self) -> AnyMessage:
        return self._messages[-1]


class OpenRouterChat:
    def __init__(
        self,
        messages: list[AnyMessage],
        output_types: list[Any] | None = None,  # TODO: Implement this.
        functions: list[Callable] | None = None,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        show_reasoning: bool = True,
    ):
        self._messages = messages
        self._model = model
        self._functions = functions
        self._output_types = output_types
        self._api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self._show_reasoning = show_reasoning

    def add_message(self, message: AnyMessage) -> "OpenRouterChat":
        self._messages.append(message)
        return self

    def _get_system_prompt(self) -> str:
        system_message = next(m for m in self._messages if isinstance(m, SystemMessage))
        return system_message.content if system_message else ""

    async def _convert_messages(self) -> list[ChatCompletionMessageParam]:
        converted_messages: list[ChatCompletionMessageParam] = []
        for message in self._messages:
            if isinstance(message, SystemMessage):
                converted_messages.append(
                    ChatCompletionSystemMessageParam(
                        role="system", content=message.content
                    )
                )
            elif isinstance(message, UserMessage):
                converted_messages.append(
                    ChatCompletionUserMessageParam(role="user", content=message.content)
                )
            elif isinstance(message, AssistantMessage):
                if isinstance(message.content, AsyncStreamedResponse):
                    async for item in message.content:
                        if isinstance(item, FunctionCall):
                            converted_messages.append(
                                ChatCompletionAssistantMessageParam(
                                    role="assistant",
                                    tool_calls=[
                                        ChatCompletionMessageToolCallParam(
                                            id=item._unique_id,
                                            type="function",
                                            function={
                                                "name": item.function.__name__,
                                                "arguments": str(item.arguments),
                                            },
                                        )
                                    ],
                                )
                            )
                        if isinstance(item, AsyncStreamedStr):
                            content = ""
                            async for chunk in item:
                                content += chunk
                            converted_messages.append(
                                ChatCompletionAssistantMessageParam(
                                    role="assistant", content=content
                                )
                            )
            elif isinstance(message, FunctionResultMessage):
                converted_messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=message.function_call._unique_id,
                        content=message.content,
                    )
                )
        return converted_messages

    def _prepare_tools(
        self, functions: list[Callable] | None
    ) -> Iterable[ChatCompletionToolParam] | None:
        if not functions:
            return None
        function_declarations: list[ChatCompletionToolParam] = []
        for function in functions:
            schema = FunctionCallFunctionSchema(function)
            tool_definition = ChatCompletionToolParam(
                function=OpenAiFunctionDefinition(
                    name=schema.name,
                    description=schema.description or "",
                    parameters=schema.parameters,
                    strict=True,
                ),
                type="function",
            )
            # This MUST be included to avoid errors.
            tool_definition["function"]["parameters"]["additionalProperties"] = False

            function_declarations.append(tool_definition)
        return function_declarations

    def _get_function(self, function_name: str) -> Callable:
        for function in self._functions if self._functions else []:
            if function.__name__ == function_name:
                return function
        raise ValueError(f"Function not found: {function_name}")

    async def asubmit(self) -> "OpenRouterChat":
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

        stream = await client.chat.completions.create(
            model=self._model,
            messages=await self._convert_messages(),
            tools=self._prepare_tools(self._functions) or NOT_GIVEN,
            stream=True,
        )

        async def async_streamed_response() -> (
            AsyncGenerator[FunctionCall | AsyncStreamedStr | StatusUpdateSSE, None]
        ):
            previous, current = tee(stream, n=2)
            reasoning = ""
            function_arguments = ""
            function_name = None
            async for chunk in previous:
                if (
                    self._show_reasoning
                    and hasattr(chunk.choices[0].delta, "reasoning")
                    and chunk.choices[0].delta.reasoning
                ):
                    if not reasoning:
                        yield reasoning_step(
                            event_type="INFO",
                            message="Model is reasoning...",
                        )
                    reasoning += chunk.choices[0].delta.reasoning
                elif chunk.choices[0].delta.content:
                    if self._show_reasoning:
                        if reasoning:
                            yield reasoning_step(
                                event_type="INFO",
                                message="Reasoning complete",
                                details={"Reasoning": reasoning},
                            )
                            reasoning = ""

                    async def async_streamed_str() -> AsyncGenerator[str, None]:
                        async for chunk in current:
                            if delta := chunk.choices[0].delta:
                                if delta.content:
                                    yield delta.content

                    yield AsyncStreamedStr(async_streamed_str())
                elif chunk.choices[0].delta.tool_calls:
                    if function := chunk.choices[0].delta.tool_calls[0].function:
                        if not function_name:
                            function_name = function.name
                        if function.arguments:
                            function_arguments += function.arguments

                if chunk.choices[0].finish_reason == "tool_calls":
                    yield FunctionCall(
                        function=self._get_function(function_name or ""),
                        **(json.loads(function_arguments) or {}),
                    )
                    function_name = None
                    function_arguments = ""

        self.add_message(
            AssistantMessage(content=AsyncStreamedResponse(async_streamed_response()))  # type: ignore
        )
        return self

    @property
    def last_message(self) -> AnyMessage:
        return self._messages[-1]


class OpenBBAgent:
    def __init__(
        self,
        query_request: QueryRequest,
        system_prompt: str,
        functions: list[Callable] | None = None,
        chat_class: type[Chat] | type[GeminiChat] | type[OpenRouterChat] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ):
        self.request = query_request
        self.widgets = query_request.widgets
        self.system_prompt = system_prompt
        self.functions = functions
        self.chat_class = chat_class or Chat
        self._model: str | OpenaiChatModel | None = model
        self._chat: Chat | GeminiChat | OpenRouterChat | None = None
        self._citations: CitationCollection | None = None
        self._messages: list[AnyMessage] = []
        self._kwargs = kwargs

        if isinstance(self.chat_class, GeminiChat):
            self._model = self._model or "gemini-2.0-flash"
        elif isinstance(self.chat_class, Chat):
            self._model = OpenaiChatModel(model=self._model or "gpt-4o")

    async def run(self, max_completions: int = 10) -> AsyncGenerator[dict, None]:
        self._messages = await self._handle_request()
        self._citations = await self._handle_callbacks()

        self._chat = self.chat_class(
            messages=self._messages,
            output_types=[AsyncStreamedResponse],
            functions=self.functions if self.functions else None,
            model=self._model,  # type: ignore[arg-type]
            **self._kwargs,
        )
        async for event in self._execute(max_completions=max_completions):
            yield event.model_dump()

        if self._citations.citations:
            yield CitationCollectionSSE(data=self._citations).model_dump()

    async def _handle_callbacks(self) -> CitationCollection:
        if not self.functions:
            return CitationCollection(citations=[])
        citations: list[Citation] = []

        if isinstance(self.request.messages[-1], LlmClientFunctionCallResultMessage):
            wrapped_function = get_wrapped_function(
                function_name=self.request.messages[-1].function,
                functions=self.functions,
            )
            async for event in wrapped_function.execute_callbacks(  # type: ignore
                request=self.request, function_call_result=self.request.messages[-1]
            ):
                if isinstance(event, Citation):
                    citations.append(event)
        return CitationCollection(citations=citations)

    async def _handle_request(self) -> list[AnyMessage]:
        chat_messages: list[AnyMessage] = [SystemMessage(self.system_prompt)]
        for message in self.request.messages:
            match message:
                case LlmClientMessage(role="human"):
                    chat_messages.append(UserMessage(content=message.content))
                case LlmClientMessage(role="ai") if isinstance(message.content, str):
                    chat_messages.append(AssistantMessage(content=message.content))
                case LlmClientMessage(role="ai") if isinstance(
                    message.content, LlmClientFunctionCall
                ):
                    # Everything is handled in the function call result message.
                    pass
                case LlmClientFunctionCallResultMessage(role="tool"):
                    if not self.functions:
                        continue
                    wrapped_function = get_wrapped_function(
                        function_name=message.extra_state.get(
                            "_locally_bound_function", ""
                        ),
                        functions=self.functions,
                    )

                    function_call: FunctionCall = FunctionCall(
                        function=wrapped_function,
                        **message.extra_state.get(
                            "copilot_function_call_arguments", {}
                        ),
                    )
                    chat_messages.append(AssistantMessage(function_call))

                    chat_messages.append(
                        FunctionResultMessage(
                            content=await wrapped_function.execute_post_processing(
                                message.data
                            ),
                            function_call=function_call,
                        )
                    )
                case _:
                    raise ValueError(f"Unsupported message type: {message}")
        return chat_messages

    async def _handle_text_stream(
        self, stream: AsyncStreamedStr
    ) -> AsyncGenerator[MessageChunkSSE, None]:
        self._chat = cast(Chat | GeminiChat, self._chat)
        async for chunk in stream:
            yield MessageChunkSSE(data=MessageChunkSSEData(delta=chunk))

    async def _handle_function_call(
        self, function_call: FunctionCall
    ) -> AsyncGenerator[FunctionCallSSE | StatusUpdateSSE, None]:
        self._chat = cast(Chat | GeminiChat, self._chat)
        function_call_result: str = ""

        if not isinstance(self._chat.last_message, AssistantMessage):
            raise ValueError("Last message is not an assistant message")

        # We sneak in the request as extra state.
        function_call.function.request = self.request

        # Execute the function.
        logger.info(
            f"Executing function: {function_call.function.__name__} with arguments: {function_call.arguments}"
        )
        async for event in function_call():
            # Yield reasoning steps.
            if isinstance(event, StatusUpdateSSE):
                yield event
            # Or an SSE to execute a function on the client-side.
            elif isinstance(event, FunctionCallSSE):
                yield event
                return
            # Otherwise, append to the function call result.
            else:
                function_call_result += str(event)
        self._chat = self._chat.add_message(
            FunctionResultMessage(
                content=function_call_result,
                function_call=function_call,
            )
        )

    async def _execute(
        self, max_completions: int
    ) -> AsyncGenerator[MessageChunkSSE | FunctionCallSSE | StatusUpdateSSE, None]:
        completion_count = 0
        # We set a limit to avoid infinite loops.
        while completion_count < max_completions:
            completion_count += 1
            # TODO: Use a protocol for this.
            self._chat = await cast(Chat | GeminiChat, self._chat).asubmit()
            # Handle a streamed text response.
            event: MessageChunkSSE | FunctionCallSSE | StatusUpdateSSE | None = None

            if isinstance(self._chat.last_message.content, AsyncStreamedResponse):
                async for item in self._chat.last_message.content:
                    if isinstance(item, StatusUpdateSSE):
                        yield item
                    elif isinstance(item, AsyncStreamedStr):
                        async for event in self._handle_text_stream(item):
                            yield event
                        return
                    elif isinstance(item, FunctionCall):
                        async for event in self._handle_function_call(item):
                            yield event
                            if isinstance(event, FunctionCallSSE):
                                return


def create_citation(
    function_call_result: LlmClientFunctionCallResultMessage,
    request: QueryRequest,
) -> Citation | None:
    data_source_requests = [
        DataSourceRequest(**data_source)
        for data_source in function_call_result.input_arguments.get("data_sources", [])
    ]
    all_widgets = (
        request.widgets.primary + request.widgets.secondary if request.widgets else []
    )

    for data_source_request in data_source_requests:
        widget = next(
            (
                w
                for w in all_widgets
                if str(w.uuid) == str(data_source_request.widget_uuid)
            ),
            None,
        )
        if not widget:
            logger.warning(
                f"Widget not found while trying to create citation: {data_source_request.widget_uuid}"
            )
            continue
        else:
            return Citation(
                source_info=SourceInfo(
                    type="widget",
                    origin=widget.origin,
                    widget_id=widget.widget_id,
                    metadata={
                        "input_args": data_source_request.input_args,
                    },
                ),
                details=[
                    {
                        "Widget Origin": widget.origin,
                        "Widget Name": widget.name,
                        "Widget ID": widget.widget_id,
                        **data_source_request.input_args,
                    }
                ],
            )
    return None
