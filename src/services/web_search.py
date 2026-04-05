"""Web search service.

Provides search results for the external verification router and
general web search when the user enables the toggle.

Uses DuckDuckGo lite as the zero-config default.
If api_key is set, overrides with the configured provider.
"""
from __future__ import annotations
import re
import httpx
from typing import Optional

# Configurable API key — if set, overrides the default DuckDuckGo search
# Future: support Perplexity, Brave, Google AI, SearXNG, etc.
_api_key: Optional[str] = None
_api_provider: str = "duckduckgo"  # "perplexity" | "brave" | "google" | "duckduckgo"

_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0), follow_redirects=True)


def configure(api_key: Optional[str] = None, provider: str = "duckduckgo"):
    """Set the search API key and provider. Called from app startup or settings."""
    global _api_key, _api_provider
    _api_key = api_key
    _api_provider = provider


async def search(query: str, top_n: int = 5) -> list[dict]:
    """Search the web. Returns list of {title, url, snippet}.

    Uses configured provider if api_key is set, otherwise DuckDuckGo.
    """
    if _api_key:
        return await _search_with_api(query, top_n)
    return await _search_duckduckgo(query, top_n)


async def _search_duckduckgo(query: str, top_n: int = 5) -> list[dict]:
    """DuckDuckGo HTML search — no API key required."""
    try:
        resp = await _client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        resp.raise_for_status()
        html = resp.text

        results = []
        # DDG HTML results: class="result__a" for links, class="result__snippet" for snippets
        links = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        snippets = re.findall(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )

        for i, (url, title_html) in enumerate(links[:top_n]):
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
            # DDG wraps URLs in a redirect — extract actual URL
            actual_url = url
            uddg = re.search(r'uddg=([^&]+)', url)
            if uddg:
                from urllib.parse import unquote
                actual_url = unquote(uddg.group(1))
            results.append({
                "title": title,
                "url": actual_url,
                "snippet": snippet[:300],
            })

        return results if results else [{"title": "No results", "url": "", "snippet": f"No results for: {query}"}]
    except Exception as e:
        return [{"title": "Search error", "url": "", "snippet": str(e)}]


async def _search_with_api(query: str, top_n: int = 5) -> list[dict]:
    """Route to configured search provider."""
    if _api_provider == "serper":
        return await _search_serper(query, top_n)
    if _api_provider == "google":
        return await _search_google_gemini(query, top_n)
    if _api_provider == "openai":
        return await _search_openai(query, top_n)
    if _api_provider == "perplexity":
        return await _search_perplexity(query, top_n)
    if _api_provider == "brave":
        return await _search_brave(query, top_n)
    return await _search_duckduckgo(query, top_n)


async def _search_google_gemini(query: str, top_n: int = 5) -> list[dict]:
    """Google Gemini API with grounding via Google Search.

    Uses the Gemini generateContent endpoint with google_search tool.
    Free tier: 15 requests/minute, 1500/day.
    """
    try:
        resp = await _client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={_api_key}",
            json={
                "contents": [{"parts": [{"text": query}]}],
                "tools": [{"google_search": {}}],
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        # Extract grounding metadata
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            # Get the text response (grounded)
            parts = content.get("parts", [])
            text = " ".join(p.get("text", "") for p in parts)

            # Get grounding sources
            grounding = candidates[0].get("groundingMetadata", {})
            chunks = grounding.get("groundingChunks", [])
            for chunk in chunks[:top_n]:
                web = chunk.get("web", {})
                results.append({
                    "title": web.get("title", ""),
                    "url": web.get("uri", ""),
                    "snippet": "",
                })

            # If no structured chunks, return the grounded text as one result
            if not results and text:
                results.append({
                    "title": "Gemini grounded response",
                    "url": "",
                    "snippet": text[:500],
                })

            # Add search queries used
            queries = grounding.get("webSearchQueries", [])
            if queries and not results:
                results.append({
                    "title": f"Searched: {', '.join(queries[:3])}",
                    "url": "",
                    "snippet": text[:300] if text else "No grounding results",
                })

        return results if results else [{"title": "No results", "url": "", "snippet": "Gemini returned no grounded content"}]
    except Exception as e:
        return [{"title": "Google search error", "url": "", "snippet": str(e)[:300]}]


async def _search_openai(query: str, top_n: int = 5) -> list[dict]:
    """OpenAI API with web search tool (GPT-4o/4o-mini).

    Uses the responses API with web_search_preview tool for grounded results.
    Falls back to chat completions with a search-oriented prompt if needed.
    """
    try:
        # Try the responses API with web search tool first
        resp = await _client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {_api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "tools": [{"type": "web_search_preview"}],
                "input": f"Search the web and return factual results for: {query}",
            },
            timeout=20.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for item in data.get("output", []):
                if item.get("type") == "web_search_call":
                    continue
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        text = part.get("text", "")
                        # Extract citations if present
                        annotations = part.get("annotations", [])
                        for ann in annotations[:top_n]:
                            if ann.get("type") == "url_citation":
                                results.append({
                                    "title": ann.get("title", ""),
                                    "url": ann.get("url", ""),
                                    "snippet": text[ann.get("start_index", 0):ann.get("end_index", 100)][:300],
                                })
                        if not annotations and text:
                            results.append({"title": "OpenAI response", "url": "", "snippet": text[:500]})
            return results if results else [{"title": "No results", "url": "", "snippet": "OpenAI returned no grounded content"}]

        # Fallback to chat completions with search instruction
        resp2 = await _client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {_api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a web search assistant. Return factual, cited information."},
                    {"role": "user", "content": query},
                ],
                "max_tokens": 500,
            },
            timeout=20.0,
        )
        resp2.raise_for_status()
        data2 = resp2.json()
        text = data2["choices"][0]["message"]["content"]
        return [{"title": "OpenAI response", "url": "", "snippet": text[:500]}]

    except Exception as e:
        return [{"title": "OpenAI search error", "url": "", "snippet": str(e)[:300]}]


async def _search_serper(query: str, top_n: int = 5) -> list[dict]:
    """Serper.dev — Google Search API with knowledge graph + featured snippets.

    Free tier: 2,500 queries. Returns structured JSON including
    knowledge graph, featured snippets, People Also Ask, and organic results.
    """
    try:
        resp = await _client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": _api_key, "Content-Type": "application/json"},
            json={"q": query, "num": top_n},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []

        # Knowledge Graph — structured entity data (highest value)
        kg = data.get("knowledgeGraph", {})
        if kg:
            desc = kg.get("description", "")
            attrs = " | ".join(f"{k}: {v}" for k, v in kg.get("attributes", {}).items())
            results.append({
                "title": f"[Knowledge Graph] {kg.get('title', '')}",
                "url": kg.get("descriptionLink", ""),
                "snippet": f"{desc} {attrs}"[:400],
            })

        # Featured Snippet / Answer Box — Google's extracted answer
        answer = data.get("answerBox", {})
        if answer:
            results.append({
                "title": f"[Answer] {answer.get('title', '')}",
                "url": answer.get("link", ""),
                "snippet": answer.get("snippet", answer.get("answer", ""))[:400],
            })

        # People Also Ask — related questions with answers
        paa = data.get("peopleAlsoAsk", [])
        for item in paa[:2]:
            results.append({
                "title": f"[PAA] {item.get('question', '')}",
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")[:300],
            })

        # Organic results — standard search links
        for item in data.get("organic", [])[:top_n]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")[:300],
            })

        return results[:top_n + 3] if results else [{"title": "No results", "url": "", "snippet": f"No Serper results for: {query}"}]
    except Exception as e:
        return [{"title": "Serper error", "url": "", "snippet": str(e)[:300]}]


async def _search_perplexity(query: str, top_n: int = 5) -> list[dict]:
    """Perplexity API — stubbed for future implementation."""
    return [{"title": "Perplexity stub", "url": "", "snippet": "Not yet implemented"}]


async def _search_brave(query: str, top_n: int = 5) -> list[dict]:
    """Brave Search API — stubbed for future implementation."""
    return [{"title": "Brave stub", "url": "", "snippet": "Not yet implemented"}]
