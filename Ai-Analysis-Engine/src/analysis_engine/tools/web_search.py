import json
import os
from dataclasses import dataclass
from analysis_engine.tools.base import ToolResult

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    score: float


def web_search(query: str, max_results: int = 5) -> ToolResult:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return ToolResult(
            "web_search",
            json.dumps({
                "error": "TAVILY_API_KEY not set — web search unavailable.",
                "tip": "Add TAVILY_API_KEY to your .env file to enable web grounding.",
            })
        )
    try:
        from tavily import Tavily
        client = Tavily(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer = False,
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "url":     r.get("url", ""),
                "title":   r.get("title", ""),
                "snippet": r.get("content", "")[:400],  # truncate long snippets
                "score":   round(r.get("score", 0.0), 3),
            })
        return ToolResult(
            "web_search",
            json.dumps({
                "query":   query,
                "results": results,
                "count":   len(results),
            }, indent=2)
        )
    except Exception as e:
        return ToolResult(
            "web_search",
            json.dumps({"error": f"Web search failed: {e}"})
        )


WEB_SEARCH_TOOL = {
    "type": "function",
    "function":{
        "name": "web_Search",
        "descrption":(
             "Search the web to find external context for a claim that references "
            "industry trends, market conditions, or other external facts. "
            "Do NOT use this to verify internal data numbers — use recompute_* tools for those. "
            "Only use for claims like 'consistent with industry trends' or 'reflects a market-wide pattern'."
        ),
        "parameters":{
            "type": "object",
            "properties": {
                "query":{
                    "type": "string",
                    "description": "Specific search query. Be precise — include dates, industry, region if relevant."
                }
            },
            "required": ["query"]
        }
    }
}