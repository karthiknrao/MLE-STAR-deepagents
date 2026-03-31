"""Web search tool using SearXNG for MLE-STAR."""

import logging
import requests
from langchain.tools import tool

logger = logging.getLogger("mle-star.tools.web_search")

# Module-level state, initialized by configure()
_state = {
    "base_url": "http://localhost:8888",
    "num_results": 10,
}


def configure(config: dict) -> None:
    """Initialize with runtime context."""
    _state.update(config)
    logger.info(f"[web_search] Configured with base_url={_state['base_url']}, num_results={_state['num_results']}")


@tool
def web_search(query: str, num_results: int = 0) -> str:
    """Search the web using SearXNG to find ML models, techniques, and solutions.

    Returns formatted search results with titles, links, and snippets.
    Use this to find state-of-the-art models for the competition/task.

    Args:
        query: Search query string.
        num_results: Number of results to return (0 = use default from config).

    Returns:
        Formatted search results as a string.
    """
    n = num_results or _state.get("num_results", 10)
    base_url = _state.get("base_url", "http://localhost:8888")

    logger.info("[web_search] Searching for: '%s' (num_results=%d)", query, n)

    try:
        resp = requests.get(
            f"{base_url}/search",
            params={
                "q": query,
                "format": "json",
                "num_results": n,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        #logger.info(f"SearXNG results : {str(data)}")
    except Exception as e:
        logger.error("[web_search] SearXNG search failed: %s", e)
        return f"Search failed: {e}"

    results = data.get("results", [])
    if not results:
        logger.warning("[web_search] No results found for query: '%s'", query)
        return "No results found."

    formatted = []
    for i, r in enumerate(results[:n], 1):
        title = r.get("title", "No title")
        link = r.get("url", "")
        snippet = r.get("content", "")
        formatted.append(f"{i}. {title}\n   Link: {link}\n   {snippet}")

    logger.info("[web_search] Found %d results for: '%s'", len(results[:n]), query)
    for i, r in enumerate(results[:5], 1):
        logger.info("[web_search]   Result %d: %s — %s", i, r.get("title", "No title")[:80], r.get("url", "")[:80])

    return "\n\n".join(formatted)
