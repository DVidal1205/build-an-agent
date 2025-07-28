"""
Image Context Analyzer Agent for LinkedIn content creation.
"""

import asyncio
import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .linkedin_state import LinkedInAgentState
from .tools import encode_image_to_base64

_LOGGER = logging.getLogger(__name__)
_MAX_LLM_RETRIES = 3

# Primary model: Use the faster 11B vision model (less rate limited)
vision_model = ChatNVIDIA(model="meta/llama-3.2-11b-vision-instruct")

# Backup models if needed
backup_vision_model = ChatNVIDIA(model="meta/llama-3.2-90b-vision-instruct")
vila_vision_model = ChatNVIDIA(model="nvidia/vila")


async def image_context_analyzer(state: LinkedInAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Analyze uploaded image to extract context and visual elements."""
    
    start_time = time.time()
    _LOGGER.info("🖼️  Starting image context analysis...")
    
    analysis_prompt = """
    Analyze this image for LinkedIn content creation. Provide a detailed analysis:
    
    🎯 VISUAL ELEMENTS:
    - Main subjects/people (describe their appearance, expressions, activities)
    - Objects and props visible
    - Text or signage present
    - Color palette and visual style
    
    🌍 SETTING & CONTEXT:
    - Location/environment (office, home, outdoor, etc.)
    - Time of day/lighting
    - Professional vs personal setting
    - Industry context clues
    
    😊 EMOTIONAL TONE:
    - Mood conveyed (professional, casual, celebratory, reflective)
    - Energy level (high, calm, intense)
    - Authenticity level (staged vs candid)
    
    💼 LINKEDIN ANGLES:
    - Career development opportunities
    - Leadership/teamwork themes
    - Industry insights possible
    - Personal branding potential
    - Motivational messaging angles
    - Humble-brag opportunities
    
    🎭 SLOP POTENTIAL:
    - Relatability factors
    - Vulnerability moments
    - "Behind the scenes" elements
    - Transformation/before-after potential
    
    Format as structured analysis with clear categories.
    """
    
    try:
        # Process image with base64 encoding for API
        _LOGGER.info("⚙️  Converting image to base64...")
        encode_start = time.time()
        image_b64 = await encode_image_to_base64.ainvoke(state.image_path or state.image_base64)
        encode_time = time.time() - encode_start
        _LOGGER.info(f"✅ Image encoded in {encode_time:.2f}s")
        
        _LOGGER.info("🚀 Calling Llama 3.2 Vision (11B) for image analysis...")
        api_start = time.time()
        
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                _LOGGER.info(f"📡 API call attempt {attempt + 1}/{_MAX_LLM_RETRIES}")
                
                # Use the simpler image format from NVIDIA sample
                content_with_image = f'{analysis_prompt} <img src="data:image/jpeg;base64,{image_b64}" />'
                
                response = await vision_model.ainvoke([
                    {
                        "role": "user", 
                        "content": content_with_image
                    }
                ], config)
                
                api_time = time.time() - api_start
                _LOGGER.info(f"✅ Llama 11B Vision API responded in {api_time:.2f}s")
                
                if response and response.content:
                    total_time = time.time() - start_time
                    _LOGGER.info(f"🎯 Image analysis completed in {total_time:.2f}s total")
                    
                    # Extract visual elements (simplified parsing)
                    visual_elements = []
                    content = str(response.content)
                    
                    # Simple extraction of elements mentioned in the analysis
                    if "people" in content.lower():
                        visual_elements.append("people")
                    if "office" in content.lower():
                        visual_elements.append("office")
                    if "outdoor" in content.lower():
                        visual_elements.append("outdoor")
                    if "text" in content.lower():
                        visual_elements.append("text")
                    
                    return {
                        "image_description": content,
                        "visual_elements": visual_elements,
                        "messages": [response]
                    }
                    
            except Exception as e:
                error_msg = str(e)
                _LOGGER.warning(f"⚠️  Attempt {attempt + 1} failed: {error_msg}")
                
                # If rate limited, wait longer between retries
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    if attempt < _MAX_LLM_RETRIES - 1:  # Don't wait after the last attempt
                        wait_time = (attempt + 1) * 5  # 5s, 10s, 15s delays
                        _LOGGER.info(f"⏳ Rate limited, waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                
                if attempt == _MAX_LLM_RETRIES - 1:
                    raise
                    
        raise RuntimeError(f"Failed to analyze image after {_MAX_LLM_RETRIES} attempts")
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = str(e)
        _LOGGER.error(f"❌ Error in image context analysis after {total_time:.2f}s: {error_msg}")
        
        # Improved fallback analysis that explicitly states what happened
        if "429" in error_msg or "Too Many Requests" in error_msg:
            fallback_description = f"""
🚨 IMAGE ANALYSIS FAILED - RATE LIMITED 🚨

The AI vision model hit rate limits while trying to analyze the uploaded image.

Based on the user's prompt: "{state.initial_prompt}"

SUGGESTED LINKEDIN ANGLES:
- Create content about overcoming technical obstacles
- Share a story about persistence and problem-solving  
- Discuss how setbacks can lead to insights
- Make this a "behind the scenes" authentic moment
- Turn this into a lesson about embracing imperfection

CONTENT DIRECTION: Create an engaging LinkedIn post that acknowledges the technical issue but turns it into a motivational story about persistence, authenticity, or learning from unexpected challenges. Make it feel genuine and relatable while incorporating typical LinkedIn "slop" elements.
"""
        else:
            fallback_description = f"""
🚨 IMAGE ANALYSIS FAILED - TECHNICAL ERROR 🚨

Unable to analyze the uploaded image due to technical issues: {error_msg}

User's original prompt: "{state.initial_prompt}"

CONTENT DIRECTION: Create LinkedIn content that relates to the user's prompt but acknowledges that the image couldn't be processed. Turn this into an authentic moment about dealing with technology challenges in a professional context.
"""
        
        return {
            "image_description": fallback_description,
            "visual_elements": ["technical_issue"],
            "messages": []
        } 