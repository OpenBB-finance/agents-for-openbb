# Vanilla Agent with Dynamic Skill Loading

This example shows the smallest useful version of dynamic skill loading for an
OpenBB agent.

It follows the same two-step handshake as the MCP example, but only for a
single skill-loading function:

1. The model sees a lightweight `skills_catalog`
2. The model requests `get_skill_content`
3. The client/frontend loads the skill and sends it back as a tool result
4. The agent answers using the loaded skill instructions

Unlike the MCP example, this agent does not do generic tool selection. It only
supports loading one skill and then answering.

## Getting started

### Prerequisites

- Poetry
- `OPENAI_API_KEY`

### Installation and running

```sh
poetry install --no-root
cd 41-vanilla-agent-dynamic-skill
poetry run uvicorn vanilla_agent_dynamic_skill.main:app --host 127.0.0.1 --port 8003 --reload
```

Or run it directly:

```sh
cd 41-vanilla-agent-dynamic-skill
poetry run python -m vanilla_agent_dynamic_skill.main
```

This example now defaults to `127.0.0.1:8003` when run as a module so it lines
up with a common local OpenBB agent URL. If your OpenBB custom agent is
configured to a different port, the server must be started on that same port.

### Testing

```sh
pytest tests
```

### Debugging local connectivity

Check these URLs directly in your browser or with `curl`:

```sh
curl http://127.0.0.1:8003/health
curl http://127.0.0.1:8003/agents.json
```

If OpenBB shows `ERR_CONNECTION_REFUSED`, it means the browser could not connect
to the server at all. The most common causes are:

- the agent process is not running
- the agent is running on a different port than the one configured in OpenBB
- the process crashed before binding the port

Once it is running, `/health` and `/agents.json` should respond directly.

### Request shape

Initial request with a skill catalog:

```json
{
  "messages": [
    {
      "role": "human",
      "content": "Use the financial-analysis skill to review AAPL."
    }
  ],
  "skills_catalog": [
    {
      "slug": "financial-analysis",
      "description": "Analyze company financials and earnings",
      "updatedAt": "2026-03-22T12:00:00Z"
    }
  ]
}
```

Follow-up request after the client loads the skill:

```json
{
  "messages": [
    {
      "role": "human",
      "content": "Use the financial-analysis skill to review AAPL."
    },
    {
      "role": "tool",
      "function": "get_skill_content",
      "input_arguments": {
        "slug": "financial-analysis"
      },
      "data": [
        {
          "status": "success",
          "data": {
            "skill": {
              "slug": "financial-analysis",
              "description": "Analyze company financials and earnings",
              "contentMarkdown": "# Financial Analysis\n\nFocus on revenue growth, margins, and guidance."
            }
          }
        }
      ]
    }
  ],
  "skills_catalog": [
    {
      "slug": "financial-analysis",
      "description": "Analyze company financials and earnings",
      "updatedAt": "2026-03-22T12:00:00Z"
    }
  ]
}
```

## Notes

- This example intentionally supports only one loaded skill per request flow.
- If multiple `selected_skills` are provided, the first one is used.
- No widget, MCP, or other tool integrations are included.
