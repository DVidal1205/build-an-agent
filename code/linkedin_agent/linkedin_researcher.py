"""
LinkedIn Research Agent for trending topics and hashtags.
"""

import json
import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .linkedin_state import LinkedInAgentState
from .tools import search_linkedin_content

_LOGGER = logging.getLogger(__name__)
_MAX_LLM_RETRIES = 3

# Use Llama 3.3 70B for processing research results
text_model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.3)


async def linkedin_research_agent(state: LinkedInAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Research trending topics and hashtags for the determined industry."""
    
    start_time = time.time()
    industry = state.industry or "general_business"
    _LOGGER.info(f"🔍 Starting LinkedIn research for '{industry}' industry...")
    
    try:
        # Create targeted search queries
        search_queries = [
            f"{industry} LinkedIn trending topics 2025",
            f"popular {industry} hashtags LinkedIn",
            f"viral {industry} content LinkedIn"
        ]
        
        _LOGGER.info(f"🌐 Searching with queries: {search_queries}")
        
        # Search using existing Tavily integration - fix the tool call
        search_results = await search_linkedin_content.ainvoke({
            "queries": search_queries,
            "content_type": "trends"
        })
        
        search_time = time.time() - start_time
        _LOGGER.info(f"✅ Search completed in {search_time:.2f}s")
        
        # Process the search results to extract topics and hashtags
        if search_results:
            processed_results = await process_search_results(search_results, industry, config)
            
            total_time = time.time() - start_time
            _LOGGER.info(f"🎯 Research completed in {total_time:.2f}s total")
            
            return {
                "trending_topics": processed_results.get("topics", []),
                "trending_hashtags": processed_results.get("hashtags", []),
                "research_results": search_results,
                "messages": []
            }
        else:
            # Fallback when search fails
            _LOGGER.warning("⚠️  Search failed, using fallback trends")
            fallback = get_fallback_trends(industry)
            
            return {
                "trending_topics": fallback["topics"],
                "trending_hashtags": fallback["hashtags"], 
                "research_results": "Search failed - using fallback trends",
                "messages": []
            }
            
    except Exception as e:
        total_time = time.time() - start_time
        _LOGGER.error(f"❌ Error in LinkedIn research after {total_time:.2f}s: {e}")
        
        # Fallback when everything fails - use the built-in industry trends
        _LOGGER.info("🔄 Using fallback industry trends...")
        fallback = get_fallback_trends(industry)
        return {
            "trending_topics": fallback["topics"],
            "trending_hashtags": fallback["hashtags"],
            "research_results": f"Research failed: {e} - Using fallback industry trends",
            "messages": []
        }


async def process_search_results(search_results: str, industry: str, config: RunnableConfig) -> dict[str, list]:
    """Extract trending topics and hashtags from search results."""
    
    _LOGGER.info("⚙️  Processing search results...")
    
    processing_prompt = f"""
    Analyze these LinkedIn research results for the {industry} industry and extract:
    
    SEARCH RESULTS:
    {search_results}
    
    Extract and return ONLY:
    1. TRENDING TOPICS: 5-8 current trending topics/themes in {industry}
    2. TRENDING HASHTAGS: 8-12 popular hashtags (include the # symbol)
    
    Focus on:
    - Recent trends (2024-2025)
    - High engagement topics
    - Industry-specific themes
    - Professional but engaging hashtags
    
    Format your response as JSON:
    {{
        "topics": ["topic1", "topic2", "topic3", ...],
        "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3", ...]
    }}
    """
    
    try:
        response = await text_model.ainvoke([
            {"role": "system", "content": "You are an expert at analyzing LinkedIn trends. Always respond with valid JSON only."},
            {"role": "user", "content": processing_prompt}
        ], config)
        
        if response and response.content:
            # Try to parse JSON response
            content = str(response.content).strip()
            
            # Clean up common JSON formatting issues
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            try:
                parsed = json.loads(content)
                
                # Validate structure
                if "topics" in parsed and "hashtags" in parsed:
                    # Ensure hashtags have # symbol
                    hashtags = []
                    for tag in parsed["hashtags"]:
                        if not tag.startswith("#"):
                            tag = "#" + tag
                        hashtags.append(tag)
                    
                    return {
                        "topics": parsed["topics"][:8],  # Limit to 8 topics
                        "hashtags": hashtags[:12]  # Limit to 12 hashtags
                    }
                    
            except json.JSONDecodeError:
                _LOGGER.warning("⚠️  Failed to parse JSON response, using fallback")
                
    except Exception as e:
        _LOGGER.warning(f"⚠️  Error processing search results: {e}")
    
    # Fallback if processing fails
    return get_fallback_trends(industry)


def get_fallback_trends(industry: str) -> dict[str, list]:
    """Provide fallback trending topics and hashtags when research fails."""
    
    industry_trends = {
        "software": {
            "topics": ["AI/ML adoption", "DevOps transformation", "Cloud migration", "Cybersecurity", "Remote development", "Open source"],
            "hashtags": ["#AI", "#MachineLearning", "#CloudComputing", "#DevOps", "#Cybersecurity", "#TechInnovation", "#SoftwareDevelopment", "#RemoteWork"]
        },
        "tech": {
            "topics": ["AI/ML adoption", "DevOps transformation", "Cloud migration", "Cybersecurity", "Remote development", "Open source"],
            "hashtags": ["#AI", "#MachineLearning", "#CloudComputing", "#DevOps", "#Cybersecurity", "#TechInnovation", "#SoftwareDevelopment", "#RemoteWork"]
        },
        "finance": {
            "topics": ["Fintech innovation", "Digital banking", "Cryptocurrency", "ESG investing", "Risk management", "Financial planning"],
            "hashtags": ["#Fintech", "#DigitalBanking", "#Investing", "#FinancialPlanning", "#ESG", "#RiskManagement", "#Finance", "#Banking"]
        },
        "marketing": {
            "topics": ["Content marketing", "Social media strategy", "AI in marketing", "Customer experience", "Brand storytelling", "Performance marketing"],
            "hashtags": ["#DigitalMarketing", "#ContentMarketing", "#SocialMedia", "#MarketingStrategy", "#BrandBuilding", "#CustomerExperience", "#MarTech", "#Growth"]
        },
        "healthcare": {
            "topics": ["Digital health", "Telemedicine", "Healthcare AI", "Patient experience", "Medical innovation", "Health equity"],
            "hashtags": ["#Healthcare", "#DigitalHealth", "#Telemedicine", "#HealthTech", "#PatientCare", "#MedicalInnovation", "#HealthEquity", "#Wellness"]
        },
        "general_business": {
            "topics": ["Leadership development", "Workplace culture", "Digital transformation", "Entrepreneurship", "Team building", "Innovation"],
            "hashtags": ["#Leadership", "#Innovation", "#BusinessGrowth", "#Entrepreneurship", "#TeamWork", "#Success", "#ProfessionalDevelopment", "#Business"]
        }
    }
    
    return industry_trends.get(industry, industry_trends["general_business"]) 