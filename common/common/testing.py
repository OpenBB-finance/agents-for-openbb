from ast import literal_eval
from pydantic import BaseModel


class CopilotEvent(BaseModel):
    event_type: str
    content: str | dict


class CopilotResponse:
    def __init__(self, event_stream: str):
        self.events: list = []
        self.index = 0
        self.event_stream = event_stream
        self.parse_event_stream()

    def parse_event_stream(self):
        captured_message_chunks = ""
        event_name = ""
        lines = self.event_stream.split("\n")
        for line in lines:
            if line.startswith("event:"):
                event_type = line.split("event:")[1].strip()
            if event_type == "copilotMessageChunk" and line.startswith("data:"):
                event_name = "copilotMessageChunk"
                data_payload = line.split("data:")[1].strip()
                data_dict_ = literal_eval(data_payload)
                captured_message_chunks += data_dict_["delta"]
            elif event_type == "copilotFunctionCall" and line.startswith("data:"):
                event_name = "copilotFunctionCall"
                data_payload = line.split("data:")[1].strip()
                data_dict_ = literal_eval(data_payload)
                self.events.append(
                    CopilotEvent(event_type=event_name, content=data_dict_)
                )
            elif event_type == "copilotStatusUpdate" and line.startswith("data:"):
                event_name = "copilotStatusUpdate"
                data_payload = line.split("data:")[1].strip()
                data_dict_ = literal_eval(data_payload)
                self.events.append(
                    CopilotEvent(event_type=event_name, content=data_dict_)
                )

        self.events.append(
            CopilotEvent(event_type="copilotMessage", content=captured_message_chunks)
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.events):
            event = self.events[self.index]
            self.index += 1
            return event
        else:
            raise StopIteration

    def starts_with(self, event_type: str, content_contains: str):
        self.index = 0
        assert self.events[self.index].event_type == event_type
        assert content_contains in str(self.events[self.index].content)
        return self

    def then(self, event_type: str, content_contains: str):
        self.index += 1
        assert self.events[self.index].event_type == event_type
        assert content_contains in str(self.events[self.index].content)
        return self

    def and_(self, content_contains: str):
        assert content_contains in str(self.events[self.index].content)
        return self

    def and_not(self, content_contains: str):
        assert content_contains not in str(self.events[self.index].content)
        return self

    def then_not(self, event_type: str, content_contains: str):
        self.index += 1
        assert self.events[self.index].event_type != event_type
        assert content_contains not in str(self.events[self.index].content)
        return self

    def then_ignore(self):
        self.index += 1
        return self

    def ends_with(self, event_type: str, content_contains: str):
        self.index = len(self.events) - 1
        assert self.events[self.index].event_type == event_type
        assert content_contains in str(self.events[self.index].content)
        return self

    def ends_with_not(self, event_type: str, content_contains: str):
        self.index = len(self.events) - 1
        assert self.events[self.index].event_type == event_type
        assert content_contains not in str(self.events[self.index].content)
        return self

    def has_any(self, event_type: str, content_contains: str):
        assert any(
            event_type == event.event_type and content_contains in str(event.content)
            for event in self.events
        )
        return self

    def has_all(self, copilot_events: list[CopilotEvent]):
        assert all(
            copilot_event.event_type == event.event_type
            and copilot_event.content == event.content
            for copilot_event in copilot_events
            for event in self.events
        )
        return self


def capture_stream_response(event_stream: str) -> tuple[str, str]:
    if "copilotFunctionCall" in event_stream:
        event_name = "copilotFunctionCall"
        event_stream = event_stream.split("\n")
        data_payload = event_stream[1].split("data:")[-1].strip()
        return event_name, data_payload

    captured_stream = ""
    event_name = ""
    lines = event_stream.split("\n")
    for line in lines:
        if line.startswith("event:"):
            event_type = line.split("event:")[1].strip()
        if event_type == "copilotMessageChunk" and line.startswith("data:"):
            event_name = "copilotMessageChunk"
            data_payload = line.split("data:")[1].strip()
            data_dict_ = literal_eval(data_payload)
            captured_stream += data_dict_["delta"]
    return event_name, captured_stream
