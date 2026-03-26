"""LLM agent with tool-calling for French territorial intelligence."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from dataclasses import asdict

from openai import AsyncOpenAI
from openbb_ai import message_chunk
from openbb_ai.models import QueryRequest

from french_territorial_intelligence.crossref import (
    TerritoryProfile,
    build_profile,
    compare_profiles,
)
from french_territorial_intelligence.sources.registry import registry

# Side-effect imports: register data sources
import french_territorial_intelligence.sources.dvf  # noqa: F401
import french_territorial_intelligence.sources.entreprises  # noqa: F401
import french_territorial_intelligence.sources.geo  # noqa: F401

SYSTEM_PROMPT = """You are the French Territorial Intelligence agent.
You analyze French territories by cross-referencing open data: business registries,
real estate transactions, and demographic data.

When given a territory profile, you synthesize the data into clear, actionable insights.
Focus on:
- Economic dynamics (enterprise density, sector composition, growth signals)
- Real estate market (price levels, transaction volume, property types)
- Cross-referenced indicators (affordability vs. economic activity, density ratios)
- Concrete comparisons when relevant

Always cite specific numbers. Be concise and analytical. Answer in English.
When data is missing or unavailable, state it clearly rather than guessing.

You have access to these tools:
- analyze_territory(city_name): Full territorial profile for a French city
- compare_territories(city_a, city_b): Side-by-side comparison of two cities
- search_enterprises(query): Search French business registry
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_territory",
            "description": "Get a full cross-referenced profile of a French territory (city or department)",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "Name of the French city to analyze (e.g. 'Lyon', 'Marseille')",
                    }
                },
                "required": ["city_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_territories",
            "description": "Compare two French territories side by side",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_a": {"type": "string", "description": "First city name"},
                    "city_b": {"type": "string", "description": "Second city name"},
                },
                "required": ["city_a", "city_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_enterprises",
            "description": "Search the French business registry (SIRENE) by keyword",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
]


async def _build_territory_profile(city_name: str) -> TerritoryProfile | None:
    """Fetch all sources and build cross-referenced profile."""
    geo_source = registry.get("geo")
    if not geo_source:
        return None

    communes = await geo_source.search(city_name)
    if not communes:
        return None

    commune = communes[0]
    code = commune["code"]

    tasks: dict[str, object] = {}
    if ent_source := registry.get("entreprises"):
        tasks["ent"] = ent_source.fetch_territory(code)
    if dvf_source := registry.get("dvf"):
        tasks["dvf"] = dvf_source.fetch_territory(code)

    results: dict[str, dict] = {}
    if tasks:
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for key, result in zip(tasks.keys(), gathered):
            if not isinstance(result, Exception):
                results[key] = result

    return build_profile(
        geo=commune,
        entreprises=results.get("ent"),
        dvf=results.get("dvf"),
    )


async def _handle_tool_call(name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "analyze_territory":
        profile = await _build_territory_profile(args["city_name"])
        if profile:
            return json.dumps(asdict(profile), ensure_ascii=False)
        return json.dumps({"error": f"City '{args['city_name']}' not found"})

    if name == "compare_territories":
        p_a = await _build_territory_profile(args["city_a"])
        p_b = await _build_territory_profile(args["city_b"])
        if p_a and p_b:
            comparison = compare_profiles(p_a, p_b)
            comparison["profile_a"] = asdict(p_a)
            comparison["profile_b"] = asdict(p_b)
            return json.dumps(comparison, ensure_ascii=False)
        missing = args["city_a"] if not p_a else args["city_b"]
        return json.dumps({"error": f"City '{missing}' not found"})

    if name == "search_enterprises":
        ent = registry.get("entreprises")
        if ent:
            results = await ent.search(args["query"])
            return json.dumps(results, ensure_ascii=False)
        return json.dumps({"error": "Enterprise source unavailable"})

    return json.dumps({"error": f"Unknown tool: {name}"})


async def stream_response(request: QueryRequest) -> AsyncGenerator[dict, None]:
    """Stream an LLM response with tool calling for territorial analysis."""
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.messages:
        role = "user" if msg.role == "human" else "assistant"
        if isinstance(msg.content, str):
            messages.append({"role": role, "content": msg.content})

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
        temperature=0.3,
    )
    choice = response.choices[0]

    if choice.message.tool_calls:
        messages.append(choice.message.model_dump())
        for tool_call in choice.message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = await _handle_tool_call(tool_call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield message_chunk(delta.content).model_dump()
    elif choice.message.content:
        yield message_chunk(choice.message.content).model_dump()
