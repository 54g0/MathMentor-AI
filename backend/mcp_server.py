from fastmcp import FastMCP
from langchain_tavily import TavilySearch
from typing import Optional
import os
from KB_setup import KB_setup,Vector_store
from dotenv import load_dotenv
load_dotenv()
mcp = FastMCP("Server")

@mcp.tool
def retrieve_data(query: str) -> str:
    """Retrieve relevant data from the knowledge base."""
    results = Vector_store.similarity_search(query, k=3)
    if not results:
        return "No documents found for your query."
    text = ""
    for result in results:
        text += f"{result.page_content}\n"
    return text if text else "No relevant information found."

@mcp.tool
def web_search(query:str)->str:
    """Perform a web search to gather information."""
    tavily_search = TavilySearch(max_results=3,include_raw_content=True,search_depth="advanced",topic="general")
    results = tavily_search.invoke({"query":query})
    return "\n".join([result['content'] for result in results['results']])



if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8001, path="/mcp")
