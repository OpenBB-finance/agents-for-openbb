# Vanilla Agent – Fundamental Analysis

This example demonstrates how to build a simple, tool-driven financial agent that uses the OpenBB SDK to perform fundamental analysis.

## What this example demonstrates

Unlike other examples that simulate reasoning, this agent:
1. **Uses Real Data**: Calls `obb.equity.fundamental.metrics` to fetch live financial data.
2. **Real Reasoning Steps**: Yields a `reasoning_step` event to the UI only when it actually executes a tool.
3. **Safe Pattern**: Separates data fetching (Python) from explanation (LLM) to reduce hallucination.

## How it works

The agent follows this loop:
1. Receives a user query (e.g., "Analyze Apple's fundamentals").
2. Determines if it needs to call a tool.
3. Yields a "Thinking..." event to the OpenBB Workspace.
4. Executes the OpenBB function locally.
5. Feeds the data back to the LLM to generate the final response.

## Example Queries

- "What is the P/E ratio and ROE for NVDA?"
- "Analyze the financial health of Tesla."
- "Get fundamental metrics for AAPL."

## Notes

- This agent uses `yfinance` as a data provider for demonstration purposes.
- This code is for educational purposes and does not constitute investment advice.