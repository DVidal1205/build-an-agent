"""
Questionnaire Agent for collecting LinkedIn style preferences.
"""

import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from .linkedin_state import LinkedInAgentState

_LOGGER = logging.getLogger(__name__)


async def questionnaire_agent(state: LinkedInAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Collect style preferences for LinkedIn post generation."""
    
    start_time = time.time()
    _LOGGER.info("📋 Starting style questionnaire...")
    
    # If all parameters are already set (programmatically), skip collection
    if all([
        state.grammar_level != 3,  # Check if changed from default
        state.emoji_level != 3,
        state.hashtag_level != 3,
        state.ragebait_level != 2,
        state.inspirational_level != 3,
        state.informational_level != 3
    ]):
        _LOGGER.info("✅ Style parameters already provided - skipping questionnaire")
        total_time = time.time() - start_time
        return {"messages": []}
    
    # For now, use default "balanced slop" parameters since this is non-interactive
    # In a real implementation, this would collect user input
    style_params = {
        "grammar_level": state.grammar_level,
        "emoji_level": state.emoji_level, 
        "hashtag_level": state.hashtag_level,
        "ragebait_level": state.ragebait_level,
        "inspirational_level": state.inspirational_level,
        "informational_level": state.informational_level
    }
    
    total_time = time.time() - start_time
    _LOGGER.info(f"✅ Style questionnaire completed in {total_time:.3f}s")
    _LOGGER.info(f"📊 Style parameters: Grammar={style_params['grammar_level']}, Emojis={style_params['emoji_level']}, Hashtags={style_params['hashtag_level']}, Ragebait={style_params['ragebait_level']}, Inspirational={style_params['inspirational_level']}, Informational={style_params['informational_level']}")
    
    return {
        "grammar_level": style_params["grammar_level"],
        "emoji_level": style_params["emoji_level"],
        "hashtag_level": style_params["hashtag_level"], 
        "ragebait_level": style_params["ragebait_level"],
        "inspirational_level": style_params["inspirational_level"],
        "informational_level": style_params["informational_level"],
        "messages": []
    }


def get_style_description(state: LinkedInAgentState) -> str:
    """Generate a description of the style preferences for use in prompts."""
    
    style_descriptions = []
    
    # Grammar level
    if state.grammar_level <= 2:
        style_descriptions.append("Use casual, imperfect grammar with typos and informal language")
    elif state.grammar_level >= 4:
        style_descriptions.append("Use proper grammar, punctuation, and professional language")
    else:
        style_descriptions.append("Use mostly correct grammar with some casual informality")
    
    # Emoji level  
    if state.emoji_level <= 2:
        style_descriptions.append("Minimal emojis (0-2 per post)")
    elif state.emoji_level >= 4:
        style_descriptions.append("Heavy emoji usage (8-15 emojis throughout)")
    else:
        style_descriptions.append("Moderate emoji usage (3-6 emojis)")
    
    # Hashtag level
    if state.hashtag_level <= 2:
        style_descriptions.append("Few hashtags (3-5)")
    elif state.hashtag_level >= 4:
        style_descriptions.append("Many hashtags (12-20)")
    else:
        style_descriptions.append("Moderate hashtags (6-10)")
    
    # Ragebait level
    if state.ragebait_level <= 2:
        style_descriptions.append("Humble and modest tone, focus on others")
    elif state.ragebait_level >= 4:
        style_descriptions.append("Highly egotistical, self-promotional, controversial takes")
    else:
        style_descriptions.append("Balanced self-promotion with some humility")
    
    # Inspirational level
    if state.inspirational_level <= 2:
        style_descriptions.append("Minimal motivational content, more practical/realistic")
    elif state.inspirational_level >= 4:
        style_descriptions.append("Highly motivational, uplifting, dream-big messaging")
    else:
        style_descriptions.append("Moderately inspirational with practical advice")
    
    # Informational level
    if state.informational_level <= 2:
        style_descriptions.append("Light on facts/data, more opinion and story-based")
    elif state.informational_level >= 4:
        style_descriptions.append("Rich in industry insights, data, and educational content")
    else:
        style_descriptions.append("Balanced mix of information and personal perspective")
    
    return "; ".join(style_descriptions)


def get_style_examples(state: LinkedInAgentState) -> str:
    """Get style examples based on the parameters."""
    
    examples = []
    
    # Grammar examples
    if state.grammar_level <= 2:
        examples.append("Grammar: 'ur right, its crazy how ai is changing everything lol'")
    elif state.grammar_level >= 4:
        examples.append("Grammar: 'You are absolutely correct. It is remarkable how artificial intelligence is transforming every industry.'")
    
    # Emoji examples
    if state.emoji_level <= 2:
        examples.append("Emojis: 'Just shipped our new feature 🚀'")
    elif state.emoji_level >= 4:
        examples.append("Emojis: 'Just shipped our new feature! 🚀💻✨ So excited! 🎉🔥💯 Team work! 👥💪🌟'")
    
    # Ragebait examples
    if state.ragebait_level <= 2:
        examples.append("Tone: 'I was lucky to learn from amazing mentors...'")
    elif state.ragebait_level >= 4:
        examples.append("Tone: 'I'm probably one of the few people who truly understands this industry...'")
    
    return " | ".join(examples) 