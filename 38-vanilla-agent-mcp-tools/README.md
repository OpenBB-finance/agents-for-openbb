# Vanilla Agent with MCP Tools

This agent demonstrates how to integrate and use Model Context Protocol (MCP) tools within an OpenBB agent, enabling dynamic tool execution for enhanced functionality.

## What it does

This agent:
- Detects and lists available MCP tools from connected MCP servers
- Executes MCP tools with proper parameter handling
- Formats responses clearly separating MCP tool output from AI analysis
- Supports real-time tool execution through the OpenBB frontend

## Key Features

### MCP Tool Integration
- **Automatic Tool Detection**: Discovers MCP tools from connected servers
- **Parameter Schema Support**: Validates and passes parameters based on tool schemas
- **Server Management**: Handles multiple MCP servers with proper server ID mapping
- **Error Handling**: Graceful fallbacks when tools are unavailable or fail

### Response Formatting
All responses with MCP tool data are structured as:
```
## MCP OUTPUT
[Raw data from MCP tool execution]

## AI OUTPUT
[AI analysis and interpretation of the tool results]
```

This clear separation helps users distinguish between actual tool data and AI-generated insights.

## Supported MCP Tools

This example works with any MCP tools, but includes specific support for:
- **OpenBB Docs Tools**: For querying OpenBB documentation
- **Data Retrieval Tools**: For fetching structured data
- **Analysis Tools**: For processing and analyzing information

## How MCP Integration Works

1. **Tool Discovery**: Agent detects available MCP tools from connected servers
2. **Schema Parsing**: Reads tool input schemas to understand required parameters
3. **Function Definition**: Creates OpenAI function definitions for each tool
4. **Execution Flow**: 
   - User requests action requiring external data
   - Agent calls appropriate MCP tool with parameters
   - Frontend executes tool on MCP server
   - Results returned and formatted for user
5. **Response Generation**: AI analyzes tool output and provides insights

## Getting Started

### Prerequisites

- Poetry for dependency management
- OpenAI API key
- MCP server with tools configured in OpenBB Terminal Pro

### Installation and Running

1. Clone this repository to your local machine.

2. Set the OpenAI API key as an environment variable:

    ``` sh
    # in .zshrc or .bashrc
    export OPENAI_API_KEY=<your-api-key>
    ```

3. Install dependencies:

``` sh
poetry install --no-root
```

4. Start the API server:

``` sh
cd 38-vanilla-agent-mcp-tools
poetry run uvicorn vanilla_agent_mcp_tools.main:app --port 7777 --reload
```

### Testing MCP Integration

To test MCP tool functionality:

1. Ensure you have MCP tools configured in OpenBB Terminal Pro
2. Ask the agent: "What MCP tools do you have available?"
3. Request tool execution: "Use [tool_name] to search for [query]"
4. Observe the structured output with MCP OUTPUT and AI OUTPUT sections

### Running Tests

The agent includes tests for MCP tool integration:

```sh
pytest tests
```

### Configuration

The agent automatically configures MCP tool support through:
- **Feature Flag**: `"mcp-tools": true` in the agent configuration
- **Tool Detection**: Automatic discovery from `request.tools`
- **Parameter Mapping**: Dynamic schema-based parameter validation

## API Documentation

Access the interactive API documentation at: http://localhost:7777/docs

## Example Usage

```
User: "What MCP tools do you have available?"
Agent: Lists all detected MCP tools with descriptions and parameters

User: "Use the docs tool to find information about widgets"
Agent: 
## MCP OUTPUT
[Documentation search results from MCP tool]

## AI OUTPUT
Based on the documentation search, here's what I found about widgets...
```

## Architecture

- **FastAPI Backend**: Handles requests and MCP tool orchestration
- **OpenAI Integration**: Uses function calling for tool selection
- **MCP Protocol**: Standard protocol for external tool communication
- **Structured Output**: Clear separation of tool data and AI analysis

## Debugging

The agent includes comprehensive debugging output:
- Tool discovery and server mapping
- Parameter validation and transmission
- Response processing and formatting

Enable debug mode by checking server logs for `[DEBUG]` messages.