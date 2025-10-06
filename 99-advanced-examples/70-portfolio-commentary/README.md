# Portfolio Commentary Agent

A sophisticated AI-powered portfolio analysis agent built for OpenBB Workspace. This agent provides intelligent commentary and insights on portfolio performance, allocation, and risk metrics by analyzing widget data from your dashboard.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Poetry for dependency management (or pip)
- Docker (optional, for containerized deployment)
- OpenRouter API key for AI model access

### Environment Setup

1. **Clone and navigate to the project:**
```bash
cd 99-advanced-examples/70-portfolio-commentary
```

2. **Create environment file:**
Create a `.env` file with the following variables:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=deepseek/deepseek-chat-v3-0324
OPENROUTER_MODEL_PERPLEXITY=perplexity/sonar
```

### Local Development

**Option 1: Using Poetry**
```bash
# Install dependencies
poetry install --no-root

# Run the server
poetry run uvicorn portfolio_commentary.main:app --reload --port 7777
```

**Option 2: Using pip**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn python-dotenv httpx sse-starlette openbb-ai

# Run the server
uvicorn portfolio_commentary.main:app --reload --port 7777
```

### Docker Deployment

```bash
# Build the image
docker build -t portfolio-commentary .

# Run the container
docker run -p 7777:7777 --env-file .env portfolio-commentary
```

## Configuration

### OpenBB Workspace Integration

The agent automatically registers with OpenBB Workspace through the `/agents.json` endpoint. It supports:

- **Streaming responses** for real-time analysis
- **Widget dashboard selection** for data source integration
- **Perplexity search feature** as an optional enhancement

### Feature Toggles

**Perplexity Search** (`perplexity-search`):
- Enables real-time web search capabilities
- Provides current market context and news
- Uses Perplexity model via OpenRouter
- Default: Disabled

## Usage

### 1. **Add Portfolio Widgets**
- Add relevant widgets to your OpenBB Workspace dashboard
- Common widgets: Performance charts, Holdings tables, Allocation breakdowns, Risk metrics

### 2. **Select Widget Data**
- Use the widget selection feature in OpenBB Workspace
- The agent will automatically detect and process the selected widget data

### 3. **Enable Features (Optional)**
- Toggle "Perplexity for Search" for enhanced market context
- This adds real-time research capabilities to your analysis

### 4. **Request Analysis**
- Ask questions about your portfolio
- Examples:
  - "Provide a commentary on my portfolio performance"
  - "What are the key risks in my current allocation?"
  - "How has my portfolio performed relative to benchmarks?"

## API Endpoints

- **GET `/agents.json`**: Agent configuration for OpenBB Workspace
- **POST `/v1/query`**: Main query endpoint for portfolio analysis
- **GET `/docs`**: Interactive API documentation (FastAPI Swagger UI)

## Technical Details

### AI Models
- **Default**: DeepSeek Chat v3 (cost-effective, high-quality analysis)
- **Enhanced**: Perplexity Sonar (when search feature is enabled)
- **Streaming**: Real-time response delivery via Server-Sent Events

### Dependencies
- **FastAPI**: Web framework and API server
- **OpenBB AI**: Widget data integration and processing
- **OpenRouter**: AI model access and management
- **httpx**: HTTP client for external API calls

### Dynamic Prompt System
The agent uses intelligent prompt generation that adapts based on:
- Available widget data and context
- Enabled features (Perplexity search)
- Analysis requirements and user requests

## Development

### Local Testing
```bash
# Access API documentation
http://localhost:7777/docs

# Test agent configuration
curl http://localhost:7777/agents.json
```

### Debugging
The application logs detailed information about:
- Widget data processing
- AI model selection (DeepSeek vs Perplexity)
- API calls and responses
- Error handling and fallbacks

## License

This project is licensed under the MIT License.