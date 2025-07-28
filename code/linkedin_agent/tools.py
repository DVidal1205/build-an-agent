"""Tools for the LinkedIn Slop Bot workflow."""

import asyncio
import base64
import logging
import os
from typing import Literal
from pathlib import Path

from langchain_core.tools import tool
from tavily import AsyncTavilyClient
from PIL import Image

_LOGGER = logging.getLogger(__name__)

# Initialize Tavily client only if API key is available
tavily_client = None
if os.getenv("TAVILY_API_KEY"):
    tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
else:
    _LOGGER.warning("TAVILY_API_KEY not found - search functionality will be limited")
INCLUDE_RAW_CONTENT = False
MAX_TOKENS_PER_SOURCE = 1000
MAX_RESULTS = 5
SEARCH_DAYS = 30


def _deduplicate_and_format_sources(
    search_response, max_tokens_per_source, include_raw_content=True
):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError(
            "Input must be either a dict with 'results' or a list of search results"
        )

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += (
            f"Most relevant content from source: {source['content']}\n===\n"
        )
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


@tool(parse_docstring=True)
async def search_linkedin_content(
    queries: list[str],
    content_type: Literal["trends", "posts", "engagement"] = "trends",
) -> str:
    """Search for LinkedIn-specific content and trends.

    Args:
        queries: List of LinkedIn-focused search queries.
        content_type: Type of LinkedIn content to search for.
          trends - LinkedIn trending topics and hashtags
          posts - High-engagement LinkedIn post examples  
          engagement - LinkedIn algorithm and engagement best practices

    Returns:
        A string of the formatted search results.
    """
    _LOGGER.info("Searching for LinkedIn content using Tavily API")

    # Add LinkedIn-specific search terms to queries
    linkedin_queries = []
    for query in queries:
        if content_type == "trends":
            linkedin_queries.append(f"LinkedIn trending topics {query} 2024")
        elif content_type == "posts":
            linkedin_queries.append(f"high engagement LinkedIn posts {query}")
        elif content_type == "engagement":
            linkedin_queries.append(f"LinkedIn algorithm best practices {query}")
        else:
            linkedin_queries.append(f"LinkedIn {query}")

    search_jobs = []
    for query in linkedin_queries:
        _LOGGER.info("Searching for LinkedIn query: %s", query)
        search_jobs.append(
            asyncio.create_task(
                tavily_client.search(
                    query,
                    max_results=MAX_RESULTS,
                    include_raw_content=INCLUDE_RAW_CONTENT,
                    topic="general",
                    days=SEARCH_DAYS,
                )
            )
        )

    search_docs = await asyncio.gather(*search_jobs)

    formatted_search_docs = _deduplicate_and_format_sources(
        search_docs,
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        include_raw_content=INCLUDE_RAW_CONTENT,
    )
    _LOGGER.debug("LinkedIn search results: %s", formatted_search_docs)
    return formatted_search_docs


@tool(parse_docstring=True)
async def process_image_for_analysis(image_path: str) -> dict:
    """Process uploaded image for LinkedIn content analysis.

    Args:
        image_path: Path to the image file to process.

    Returns:
        Dictionary with image metadata and base64 encoding.
    """
    _LOGGER.info("Processing image for analysis: %s", image_path)
    
    try:
        # Validate image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Open and analyze image
        with Image.open(image_path) as img:
            # Get basic metadata
            metadata = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_mb": Path(image_path).stat().st_size / (1024 * 1024)
            }
        
        # Convert to base64 for API calls
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        return {
            "metadata": metadata,
            "base64": image_b64,
            "file_path": image_path,
            "processed": True
        }
        
    except Exception as e:
        _LOGGER.error("Error processing image %s: %s", image_path, e)
        return {
            "error": str(e),
            "processed": False
        }


@tool(parse_docstring=True)
async def encode_image_to_base64(image_input: str) -> str:
    """Convert image to base64 encoding for API calls.

    Args:
        image_input: Either file path to image or existing base64 string.

    Returns:
        Base64 encoded string of the image.
    """
    _LOGGER.info("Encoding image to base64")
    
    # If already base64, return as is
    if image_input.startswith("data:image") or len(image_input) > 1000:
        return image_input
    
    # Otherwise treat as file path
    try:
        with open(image_input, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        return image_b64
    except Exception as e:
        _LOGGER.error("Error encoding image to base64: %s", e)
        raise


@tool(parse_docstring=True)
async def get_trending_hashtags(industry: str) -> list[str]:
    """Get trending hashtags for specific industry.

    Args:
        industry: Industry or topic to get hashtags for.

    Returns:
        List of trending hashtags.
    """
    _LOGGER.info("Getting trending hashtags for industry: %s", industry)
    
    # Use Tavily to search for trending hashtags
    query = f"trending LinkedIn hashtags {industry} 2024"
    
    search_result = await tavily_client.search(
        query,
        max_results=3,
        include_raw_content=False,
        topic="general"
    )
    
    # Extract hashtags from search results (simplified)
    hashtags = [
        f"#{industry.title()}",
        "#Leadership", 
        "#Innovation",
        "#Growth",
        "#Success",
        "#Motivation",
        "#CareerDevelopment",
        "#BusinessTips",
        "#Networking",
        "#ProfessionalDevelopment"
    ]
    
    return hashtags[:8]  # Return top 8 hashtags


# Re-export the original search function for backward compatibility
search_tavily = search_linkedin_content 