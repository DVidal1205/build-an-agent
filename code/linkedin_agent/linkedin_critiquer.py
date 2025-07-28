"""
LinkedIn Critiquer Agent for evaluating and improving content quality.
"""

import json
import logging
import re
import time
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .linkedin_state import LinkedInAgentState

_LOGGER = logging.getLogger(__name__)
_MAX_LLM_RETRIES = 3

# Use Llama 3.3 70B for content critique
text_model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.3)


async def linkedin_critiquer_agent(state: LinkedInAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Critique LinkedIn post and provide specific improvement suggestions."""
    
    start_time = time.time()
    _LOGGER.info("🔍 Starting LinkedIn post critique...")
    
    if not state.post_drafts:
        _LOGGER.warning("⚠️  No post drafts to critique")
        return {"critique_feedback": ["No content to critique"], "messages": []}
    
    # Get the latest draft for critique
    latest_draft = state.post_drafts[-1]
    draft_number = len(state.post_drafts)
    
    critique_prompt = f"""
    Evaluate this LinkedIn post for maximum engagement potential:
    
    POST TO CRITIQUE (Draft #{draft_number}):
    {latest_draft}
    
    CONTEXT:
    - Industry: {state.industry or 'general_business'}
    - Image content: {state.image_description or 'No image'}
    - User intent: {state.initial_prompt}
    - Available trending topics: {', '.join(state.trending_topics[:3]) if state.trending_topics else 'None'}
    
    EVALUATION FRAMEWORK:
    Rate each category from 1-10 and provide specific feedback:
    
    📊 ENGAGEMENT POTENTIAL (1-10):
    - Hook strength: Does the opening grab attention within 2 seconds?
    - Curiosity gap: Does it make people want to read more?
    - Emotional resonance: Will people feel something (inspired, angry, nostalgic)?
    - Comment-bait effectiveness: Does it naturally encourage responses?
    - Share-worthiness: Would people share this with their network?
    
    🎭 SLOP AUTHENTICITY (1-10):
    - Humble bragging level: Right balance of achievement + modesty?
    - Vulnerability authenticity: Genuine vs performative sharing?
    - Buzzword usage: Professional enough but not too corporate?
    - Emoji/formatting: Authentic LinkedIn "sloppiness"?
    - Relatability factor: Can average person connect with this?
    
    🎯 ALGORITHM OPTIMIZATION (1-10):
    - Post length: Optimal for LinkedIn algorithm (150-300 words)?
    - Question placement: Strategic questions to drive comments?
    - Hashtag strategy: Right mix and quantity (8-15 hashtags)?
    - Time-sensitive relevance: Current trends incorporated?
    - Image integration: Does it reference the visual content effectively?
    
    📈 IMPROVEMENT SUGGESTIONS:
    Provide 3-5 specific, actionable improvements for the next draft:
    
    RESPONSE FORMAT (JSON):
    {{
        "scores": {{
            "engagement_potential": X,
            "slop_authenticity": X,
            "algorithm_optimization": X
        }},
        "overall_score": X.X,
        "strengths": ["strength1", "strength2", "strength3"],
        "weaknesses": ["weakness1", "weakness2", "weakness3"],
        "improvements": [
            "Specific improvement 1",
            "Specific improvement 2", 
            "Specific improvement 3"
        ],
        "verdict": "CONTINUE" or "APPROVED",
        "reasoning": "Brief explanation of verdict"
    }}
    
    Be constructively critical - we want to create highly engaging content!
    """
    
    try:
        _LOGGER.info(f"🤖 Calling Llama 3.3 70B for critique of draft #{draft_number}...")
        
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                response = await text_model.ainvoke([
                    {"role": "system", "content": "You are an expert LinkedIn content strategist who evaluates posts for maximum engagement. You understand the LinkedIn algorithm and what makes content go viral. Always respond with valid JSON and be constructively critical to help improve content quality."},
                    {"role": "user", "content": critique_prompt}
                ], config)
                
                if response and response.content:
                    critique_text = str(response.content).strip()
                    
                    # Parse the JSON response
                    critique_data = parse_critique_response(critique_text)
                    
                    if critique_data:
                        total_time = time.time() - start_time
                        overall_score = critique_data.get("overall_score", 0)
                        verdict = critique_data.get("verdict", "CONTINUE")
                        
                        _LOGGER.info(f"✅ Critique completed in {total_time:.2f}s")
                        _LOGGER.info(f"📊 Overall score: {overall_score}/10 - Verdict: {verdict}")
                        
                        # Format the critique for storage
                        formatted_critique = format_critique_feedback(critique_data, draft_number)
                        
                        return {
                            "critique_feedback": state.critique_feedback + [formatted_critique],
                            "messages": [response]
                        }
                    else:
                        _LOGGER.warning(f"⚠️  Failed to parse critique response on attempt {attempt + 1}")
                        
            except Exception as e:
                _LOGGER.warning(f"⚠️  Critique attempt {attempt + 1} failed: {e}")
                if attempt == _MAX_LLM_RETRIES - 1:
                    raise
                    
        raise RuntimeError(f"Failed to generate critique after {_MAX_LLM_RETRIES} attempts")
        
    except Exception as e:
        total_time = time.time() - start_time
        _LOGGER.error(f"❌ Error in LinkedIn critique after {total_time:.2f}s: {e}")
        
        # Fallback critique
        fallback_critique = f"""
DRAFT #{draft_number} CRITIQUE (Fallback):

❌ CRITIQUE FAILED: {str(e)}

⚠️  AUTOMATIC APPROVAL: Moving to final formatting due to critique system error.

SCORES (Estimated):
- Engagement: 7/10
- Authenticity: 7/10  
- Optimization: 7/10
- Overall: 7.0/10

VERDICT: APPROVED (Fallback)
"""
        
        return {
            "critique_feedback": state.critique_feedback + [fallback_critique],
            "messages": []
        }


def parse_critique_response(critique_text: str) -> Dict[str, Any] | None:
    """Parse the JSON critique response from the LLM."""
    
    try:
        # Clean up common JSON formatting issues
        if critique_text.startswith("```json"):
            critique_text = critique_text.replace("```json", "").replace("```", "").strip()
        elif critique_text.startswith("```"):
            critique_text = critique_text.replace("```", "").strip()
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', critique_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            critique_data = json.loads(json_str)
            
            # Validate required fields
            if "scores" in critique_data and "verdict" in critique_data:
                # Calculate overall score if not provided
                if "overall_score" not in critique_data:
                    scores = critique_data["scores"]
                    critique_data["overall_score"] = sum(scores.values()) / len(scores)
                
                return critique_data
                
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        _LOGGER.warning(f"⚠️  JSON parsing error: {e}")
    
    return None


def format_critique_feedback(critique_data: Dict[str, Any], draft_number: int) -> str:
    """Format critique data into readable feedback."""
    
    scores = critique_data.get("scores", {})
    overall = critique_data.get("overall_score", 0)
    verdict = critique_data.get("verdict", "CONTINUE")
    
    feedback = f"""
DRAFT #{draft_number} CRITIQUE:

📊 SCORES:
- Engagement Potential: {scores.get('engagement_potential', 0)}/10
- Slop Authenticity: {scores.get('slop_authenticity', 0)}/10
- Algorithm Optimization: {scores.get('algorithm_optimization', 0)}/10
- Overall Score: {overall:.1f}/10

✅ STRENGTHS:
{format_list(critique_data.get('strengths', []))}

❌ AREAS FOR IMPROVEMENT:
{format_list(critique_data.get('weaknesses', []))}

🔧 SPECIFIC IMPROVEMENTS FOR NEXT DRAFT:
{format_list(critique_data.get('improvements', []))}

🎯 VERDICT: {verdict}
💭 REASONING: {critique_data.get('reasoning', 'No reasoning provided')}
"""
    
    return feedback.strip()


def format_list(items: list) -> str:
    """Format a list of items with bullet points."""
    if not items:
        return "- None provided"
    return "\n".join(f"- {item}" for item in items)


def should_continue_refining(state: LinkedInAgentState) -> str:
    """Determine if the post needs more refinement based on critique feedback."""
    
    # Safety: Max iterations to prevent infinite loops
    MAX_ITERATIONS = 5
    if len(state.post_drafts) >= MAX_ITERATIONS:
        _LOGGER.info(f"🛑 Max iterations ({MAX_ITERATIONS}) reached - moving to formatter")
        return "finish"
    
    # If no critique feedback, continue (shouldn't happen but safety check)
    if not state.critique_feedback:
        _LOGGER.info("⚠️  No critique feedback - continuing to avoid infinite loop")
        return "continue"
    
    # Parse the latest critique to get scores and verdict
    latest_critique = state.critique_feedback[-1]
    
    # Look for verdict in the critique
    if "VERDICT: APPROVED" in latest_critique:
        _LOGGER.info("✅ Critique approved the content - moving to formatter")
        return "finish"
    
    # Look for high scores (7+ on all categories means good enough)
    score_pattern = r"Overall Score: (\d+\.?\d*)/10"
    score_match = re.search(score_pattern, latest_critique)
    
    if score_match:
        overall_score = float(score_match.group(1))
        if overall_score >= 7.5:
            _LOGGER.info(f"🎯 High overall score ({overall_score}/10) - moving to formatter")
            return "finish"
        else:
            _LOGGER.info(f"📈 Score needs improvement ({overall_score}/10) - continuing refinement")
            return "continue"
    
    # Default: continue refining if we can't determine quality
    _LOGGER.info("🔄 Continuing refinement - score unclear")
    return "continue" 