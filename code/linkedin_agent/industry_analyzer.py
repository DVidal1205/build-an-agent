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
    
    You are an industry classification agent. Based on user content (text, images, or metadata), you must categorize the content into a single most relevant industry.

AVAILABLE INDUSTRY OPTIONS:
- software: Software development, web/app development, DevOps, AI/ML, cybersecurity, tech startups, SaaS, open source, gaming tech, entertainment tech (e.g., NVIDIA, Google, Epic Games)
- finance: Banking, investing, venture capital, private equity, accounting, fintech, crypto/blockchain, insurance, hedge funds, trading
- healthcare: Medicine, nursing, pharmaceuticals, biotech, medical research, public health, medical devices, hospitals
- marketing: Digital marketing, SEO/SEM, social media strategy, influencer campaigns, advertising, branding, content creation
- consulting: Management consulting, business strategy, operations, organizational design, transformation, business advisory
- education: K–12, higher education, teaching, tutoring, curriculum design, e-learning, edtech platforms, academic research
- manufacturing: Industrial production, mechanical/automotive/aerospace engineering, robotics, automation, logistics, supply chain
- retail: E-commerce, merchandising, fashion, consumer goods, food & beverage, brick-and-mortar retail
- real_estate: Property development, architecture, urban planning, commercial/residential real estate, construction, zoning
- energy: Renewable energy, oil & gas, power generation, utilities, sustainability, environmental engineering
- media: Journalism, film, TV, publishing, radio, podcasting, video production, entertainment industry
- nonprofit: NGOs, philanthropy, social impact, humanitarian work, community organizing, volunteering
- general_business: Business leadership, entrepreneurship, operations, management, productivity, professional development (use **only** when none of the above clearly apply)

INDUSTRY DETECTION HINTS:
- Keywords like Python, GitHub, DevOps, cloud, Kubernetes, LLMs, NVIDIA, open-source, AI, Fortnite, startup → software
- Banking, stocks, VC, ROI, financial planning, crypto, Bloomberg, Robinhood → finance
- Doctor, nurse, patient, clinic, medical devices, CRISPR, biotech, Moderna → healthcare
- Campaign, Instagram, branding, storytelling, Meta Ads, influencers → marketing
- McKinsey, Bain, "business problem", market analysis, SWOT, strategy roadmap → consulting
- Teacher, student, university, Canvas, edtech, MOOC, academia, thesis → education
- Robotics, manufacturing line, industrial automation, Boeing, Tesla factory → manufacturing
- Amazon, Shopify, DTC, clothing brand, snack launch, fashion line → retail
- Zillow, construction site, home flipping, smart cities, CAD, Revit → real_estate
- Solar panels, wind farm, EV charging, net-zero, Exxon, green tech → energy
- Hollywood, YouTube, documentary, podcast, screenwriting, broadcast → media
- Red Cross, volunteering, impact report, NGO, community event → nonprofit

INSTRUCTIONS:
1. Choose **one and only one** industry from the list above.
2. Consider **both textual and visual context** if provided (e.g., company names, tools, lingo, visuals).
3. Be strict — prefer **specific** industries over general ones.
4. If multiple industries seem relevant, select the **most dominant** one based on context.
5. Use **"general_business" only as a last resort**, and only if **none** of the others clearly apply.
6. Output only the **industry name** as listed (e.g., `software`, `finance`).

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