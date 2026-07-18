from langchain_core.tools import tool

@tool
def search_web_mock(query: str) -> str:
    """Mock search tool to search the web for information.
    
    Args:
        query: Search query string.
    """
    return f"Search result for '{query}': No recent web information found. MOCKED RESULT."
