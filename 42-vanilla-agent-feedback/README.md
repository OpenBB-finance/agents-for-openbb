# 42 - Vanilla Agent Feedback

A vanilla agent that demonstrates how to receive and persist user feedback (thumbs up/down) from OpenBB Workspace.

## What it does

This agent exposes a `/v1/feedback` endpoint that the Workspace frontend calls when a user gives thumbs up/down on an AI response. Feedback is persisted to a local `feedback.json` file.

The agent declares `"feedback": true` in its `agents.json` features, which signals to the Workspace that this agent supports receiving feedback directly (instead of it going to analytics).

## Feedback payload

```json
{
  "vote": "thumbs_up",
  "tags": ["Not factually correct / Hallucinations / Inaccurate"],
  "user_comment": "Optional additional comment",
  "ai_response": "The AI response that was rated",
  "user_prompt": "The user's original prompt",
  "trace_id": "request-trace-id"
}
```

## Running

```bash
cd 42-vanilla-agent-feedback
poetry run uvicorn vanilla_agent_feedback.main:app --port 7777 --reload
```

Then add `http://localhost:7777` as a custom agent in OpenBB Workspace.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/agents.json` | Agent manifest |
| POST | `/v1/query` | Chat query (basic LLM passthrough) |
| POST | `/v1/feedback` | Receive user feedback |

## Testing

```bash
poetry run pytest 42-vanilla-agent-feedback/tests
```
