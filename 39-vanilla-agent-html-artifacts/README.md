# Example agent that produces HTML artifacts

This is an example agent, powered by OpenAI, that can produce HTML artifacts as part of its response. These HTML artifacts are rendered inline in the OpenBB Workspace copilot chat, and users can create dashboard widgets from them.

## Features

- **Inline HTML rendering**: HTML artifacts are displayed directly in the chat
- **Widget creation**: Users can click the widget icon to add HTML artifacts to their dashboard
- **XSS protection**: The frontend sanitizes all HTML with DOMPurify before rendering

## Getting started

### Prerequisites

- Python 3.11+
- Poetry for dependency management
- OpenAI API key

### Installation and Running

1. Set the OpenAI API key as an environment variable:

```sh
export OPENAI_API_KEY=<your-api-key>
```

2. Install dependencies:

```sh
poetry install --no-root
```

3. Start the API server:

```sh
cd 39-vanilla-agent-html-artifacts
poetry run uvicorn vanilla_agent_html.main:app --port 7777 --reload
```

4. Add the agent to OpenBB Workspace:
   - Go to Copilot settings
   - Add custom copilot with URL: `http://localhost:7777`

### HTML Artifact Format

HTML artifacts are sent as SSE events with the following format:

```python
{
    "event": "copilotMessageArtifact",
    "data": {
        "type": "html",
        "uuid": "unique-id",
        "name": "artifact_name",
        "description": "A description of the artifact",
        "content": "<div>Your HTML content here</div>"
    }
}
```

### Security Considerations

- Avoid including `<script>` tags - they will be stripped by the frontend
- Inline styles are supported but `on*` event handlers are stripped
- The following tags are forbidden: `script`, `style`, `iframe`, `form`, `object`, `embed`

### Example HTML Templates

The agent includes several example templates:
- **Dashboard Card**: A gradient portfolio summary card
- **Metric Cards**: Key metrics displayed in a grid
- **Alert Box**: A styled notification/alert component

## SDK Note

The `openbb-ai` SDK's `ClientArtifact` model currently only supports `type: Literal["text", "table", "chart"]`. This example manually constructs the SSE event to demonstrate HTML artifacts. The SDK should be updated to include `"html"` as a supported type.
