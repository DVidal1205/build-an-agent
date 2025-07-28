"""Main entry point for the LinkedIn Slop Bot workflow."""

import asyncio
from typing import Any

from .linkedin_state import LinkedInAgentState
from .linkedin_agent import graph


async def async_create_linkedin_post(
    initial_prompt: str, 
    image_path: str | None = None,
    image_base64: str | None = None,
    post_type: str = "general",
    grammar_level: int = 3,
    emoji_level: int = 3,
    hashtag_level: int = 3,
    ragebait_level: int = 2,
    inspirational_level: int = 3,
    informational_level: int = 3
) -> Any | dict[str, Any] | None:
    """Create a LinkedIn post from image and prompt with style preferences."""
    
    state = LinkedInAgentState(
        initial_prompt=initial_prompt,
        image_path=image_path,
        image_base64=image_base64,
        post_type=post_type,
        grammar_level=grammar_level,
        emoji_level=emoji_level,
        hashtag_level=hashtag_level,
        ragebait_level=ragebait_level,
        inspirational_level=inspirational_level,
        informational_level=informational_level
    )
    
    result = await graph.ainvoke(state)
    return result


def create_linkedin_post(
    initial_prompt: str,
    image_path: str | None = None, 
    image_base64: str | None = None,
    post_type: str = "general",
    grammar_level: int = 3,
    emoji_level: int = 3,
    hashtag_level: int = 3,
    ragebait_level: int = 2,
    inspirational_level: int = 3,
    informational_level: int = 3
) -> Any | dict[str, Any] | None:
    """Create a LinkedIn post from image and prompt with style preferences."""
    
    return asyncio.run(async_create_linkedin_post(
        initial_prompt, image_path, image_base64, post_type,
        grammar_level, emoji_level, hashtag_level, ragebait_level,
        inspirational_level, informational_level
    )) 