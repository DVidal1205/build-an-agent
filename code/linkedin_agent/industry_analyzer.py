"""
Industry Analyzer Agent for determining the correct industry context.
"""

import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .linkedin_state import LinkedInAgentState

_LOGGER = logging.getLogger(__name__)
_MAX_LLM_RETRIES = 3

# Use Llama 3.3 70B for industry analysis
text_model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.3)


async def industry_analyzer_agent(state: LinkedInAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Analyze the image and prompt to determine the industry context."""
    
    start_time = time.time()
    _LOGGER.info("🏭 Starting industry analysis...")
    
    industry_prompt = f"""
    Based on the following information, determine the PRIMARY industry this LinkedIn post should target:
    
    IMAGE DESCRIPTION:
    {state.image_description or "No image provided"}
    
    USER PROMPT:
    {state.initial_prompt}
    
    POST TYPE:
    {state.post_type}
    
    INDUSTRY OPTIONS:
    - software - Software development, programming, AI/ML, startups, SaaS, tech companies (NVIDIA, Google, etc.), gaming/entertainment tech
    - finance - Banking, investment, fintech, accounting, trading, insurance  
    - healthcare - Medicine, nursing, pharma, biotech, medical devices
    - marketing - Digital marketing, advertising, branding, social media, content
    - consulting - Management consulting, strategy, business advisory
    - education - Teaching, training, e-learning, academic, educational tech
    - manufacturing - Industrial, automotive, aerospace, supply chain
    - retail - E-commerce, consumer goods, fashion, food & beverage
    - real_estate - Property, construction, architecture, urban planning
    - energy - Oil & gas, renewable energy, utilities, sustainability
    - media - Journalism, entertainment, publishing, broadcasting
    - nonprofit - NGO, social impact, volunteering, community work
    - general_business - Generic business content, leadership, entrepreneurship (ONLY if none of the above clearly apply)
    
    INDUSTRY DETECTION HINTS:
    - NVIDIA, gaming, Fortnite, AI/ML, programming, coding, DevOps, cloud, containers → software
    - Banking, investing, trading, financial planning → finance
    - Doctors, nurses, medical, healthcare, pharma → healthcare
    - Marketing campaigns, branding, advertising → marketing
    
    INSTRUCTIONS:
    1. Choose the SINGLE most relevant industry from the list above
    2. Consider both the image content AND the user's prompt carefully
    3. Look for specific company names, technologies, or industry keywords
    4. If multiple industries apply, pick the most prominent one
    5. Use "general_business" ONLY if no specific industry clearly fits
    6. Respond with ONLY the industry name (e.g., "software", "finance", etc.)
    
    INDUSTRY:"""
    
    try:
        _LOGGER.info("🤖 Calling Llama 3.3 70B for industry analysis...")
        
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                response = await text_model.ainvoke([
                    {"role": "system", "content": "You are an expert at categorizing business content by industry. Always respond with exactly one industry name from the provided list. Pay special attention to company names and technical keywords."},
                    {"role": "user", "content": industry_prompt}
                ], config)
                
                if response and response.content:
                    industry = str(response.content).strip().lower()
                    
                    # Clean up the response to ensure it's just the industry name
                    industry = industry.replace("industry:", "").strip()
                    industry = industry.split()[0] if industry.split() else "general_business"
                    
                    # Validate it's a known industry
                    valid_industries = [
                        "software", "finance", "healthcare", "marketing", 
                        "consulting", "education", "manufacturing", "retail", 
                        "real_estate", "energy", "media", "nonprofit", "general_business"
                    ]
                    
                    if industry not in valid_industries:
                        industry = "general_business"
                    
                    total_time = time.time() - start_time
                    _LOGGER.info(f"✅ Industry determined: '{industry}' in {total_time:.2f}s")
                    
                    return {
                        "industry": industry,
                        "messages": [response]
                    }
                    
            except Exception as e:
                _LOGGER.warning(f"⚠️  Industry analysis attempt {attempt + 1} failed: {e}")
                if attempt == _MAX_LLM_RETRIES - 1:
                    raise
                    
        raise RuntimeError(f"Failed to determine industry after {_MAX_LLM_RETRIES} attempts")
        
    except Exception as e:
        total_time = time.time() - start_time
        _LOGGER.error(f"❌ Error in industry analysis after {total_time:.2f}s: {e}")
        
        # Smart fallback based on keywords in the prompt
        prompt_lower = state.initial_prompt.lower()
        image_desc_lower = (state.image_description or "").lower()
        
        # Check for obvious industry indicators
        if any(keyword in prompt_lower or keyword in image_desc_lower for keyword in 
               ["nvidia", "software", "ai", "ml", "coding", "programming", "tech", "gaming", "fortnite", "dev", "cloud"]):
            fallback_industry = "software"
        elif any(keyword in prompt_lower for keyword in 
                ["finance", "banking", "investment", "trading", "money"]):
            fallback_industry = "finance"
        elif any(keyword in prompt_lower for keyword in 
                ["marketing", "advertising", "brand", "campaign"]):
            fallback_industry = "marketing"
        else:
            fallback_industry = "general_business"
        
        _LOGGER.info(f"🔄 Using keyword-based fallback: '{fallback_industry}'")
        
        return {
            "industry": fallback_industry,
            "messages": []
        } 