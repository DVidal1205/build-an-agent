"""
LinkedIn Author Agent for generating "slop" content.
"""

import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .linkedin_state import LinkedInAgentState
from .prompts import linkedin_author_prompt, SLOP_CHARACTERISTICS
from .questionnaire_agent import get_style_description, get_style_examples

_LOGGER = logging.getLogger(__name__)
_MAX_LLM_RETRIES = 3

# Primary model: Llama 3.3 70B for text generation
text_model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.7)


async def linkedin_author_agent(state: LinkedInAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Generate LinkedIn post content with intentional 'slop' characteristics."""
    
    start_time = time.time()
    draft_number = len(state.post_drafts) + 1
    is_revision = len(state.post_drafts) > 0
    
    if is_revision:
        _LOGGER.info(f"✍️  Revising LinkedIn content (Draft #{draft_number})...")
    else:
        _LOGGER.info("✍️  Starting LinkedIn content generation...")
    
    # Format trending topics and hashtags for inclusion
    trending_info = ""
    if state.trending_topics:
        trending_info += f"\nTRENDING TOPICS: {', '.join(state.trending_topics)}"
    if state.trending_hashtags:
        trending_info += f"\nTRENDING HASHTAGS: {', '.join(state.trending_hashtags)}"
    
    # Get personalized style preferences
    style_description = get_style_description(state)
    style_examples = get_style_examples(state)
    
    # Prepare context for the author prompt
    context = {
        "image_description": state.image_description or "No image provided",
        "initial_prompt": state.initial_prompt,
        "industry": state.industry or "general_business",
        "trending_info": trending_info,
        "research_results": state.research_results or "No research conducted",
        "user_responses": str(state.user_responses) if state.user_responses else "No user preferences specified",
        "slop_characteristics": SLOP_CHARACTERISTICS,
        "style_description": style_description,
        "style_examples": style_examples,
        "style_scores": f"Grammar={state.grammar_level}/5, Emojis={state.emoji_level}/5, Hashtags={state.hashtag_level}/5, Ragebait={state.ragebait_level}/5, Inspirational={state.inspirational_level}/5, Informational={state.informational_level}/5"
    }
    
    # Build prompt based on whether this is a revision or initial draft
    if is_revision:
        enhanced_prompt = build_revision_prompt(context, state, draft_number)
    else:
        enhanced_prompt = build_initial_prompt(context)
    
    # Format the prompt
    _LOGGER.info(f"⚙️  Formatting prompt for Llama 3.3 70B (Draft #{draft_number}) with style preferences...")
    prompt_start = time.time()
    formatted_prompt = enhanced_prompt
    prompt_time = time.time() - prompt_start
    _LOGGER.info(f"✅ Prompt formatted in {prompt_time:.3f}s")
    
    try:
        _LOGGER.info(f"🚀 Calling Llama 3.3 70B for LinkedIn content generation (Draft #{draft_number})...")
        api_start = time.time()
        
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                _LOGGER.info(f"📡 API call attempt {attempt + 1}/{_MAX_LLM_RETRIES}")
                
                system_prompt = "You are an expert LinkedIn content creator specializing in engaging 'slop' content that maximizes engagement while feeling authentic. You MUST always reference the provided image content in your posts, even if it seems unrelated to the topic. Find creative ways to connect images to business lessons. Follow the specific style preferences provided by the user exactly."
                
                if is_revision:
                    system_prompt += " You are revising content based on expert feedback. Incorporate the specific improvements while maintaining the authentic LinkedIn 'slop' style and user's style preferences."
                
                response = await text_model.ainvoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt}
                ], config)
                
                api_time = time.time() - api_start
                _LOGGER.info(f"✅ Llama API responded in {api_time:.2f}s")
                
                if response and response.content:
                    post_content = str(response.content).strip()
                    total_time = time.time() - start_time
                    _LOGGER.info(f"🎯 Content generation completed in {total_time:.2f}s total (Draft #{draft_number})")
                    
                    # Add to post drafts
                    updated_drafts = state.post_drafts + [post_content]
                    
                    return {
                        "post_drafts": updated_drafts,
                        "messages": [response]
                    }
                    
            except Exception as e:
                _LOGGER.warning(f"⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt == _MAX_LLM_RETRIES - 1:
                    raise
                    
        raise RuntimeError(f"Failed to generate content after {_MAX_LLM_RETRIES} attempts")
        
    except Exception as e:
        total_time = time.time() - start_time
        _LOGGER.error(f"❌ Error in LinkedIn content generation after {total_time:.2f}s: {e}")
        
        # Fallback content generation with image reference
        fallback_hashtags = state.trending_hashtags[:8] if state.trending_hashtags else ["#LinkedIn", "#Business", "#Growth", "#Success"]
        
        image_reference = ""
        if state.image_description and state.image_description != "No image provided":
            image_reference = f"Seeing {state.image_description.lower()[:50]}... got me thinking: "
        
        fallback_post = f"""
🔥 Hot take: {image_reference}Sometimes technology doesn't work as expected...

Today I was trying to {state.initial_prompt.lower()}...

But here's what I learned: Even when tools fail, the human element shines through! 💪

The real LinkedIn lesson? Authenticity beats perfection every time.

What's your biggest tech fail that turned into a win? Share below! 👇

{' '.join(fallback_hashtags)}
"""
        
        return {
            "post_drafts": state.post_drafts + [fallback_post],
            "messages": []
        }


def build_initial_prompt(context: dict) -> str:
    """Build the prompt for the initial draft with style preferences."""
    
    return f"""
    Create a LinkedIn post with these elements:
    
    MANDATORY REQUIREMENTS:
    🖼️ IMAGE CONTENT (MUST REFERENCE): {context['image_description']}
    📝 USER REQUEST: {context['initial_prompt']}
    🏭 INDUSTRY CONTEXT: {context['industry']}
    
    TRENDING INFORMATION TO INCORPORATE:
    {context['trending_info']}
    
    PERSONALIZED STYLE PREFERENCES (FOLLOW EXACTLY):
    📊 Style Scores: {context['style_scores']}
    📝 Style Guidelines: {context['style_description']}
    💡 Style Examples: {context['style_examples']}
    
    CRITICAL INSTRUCTION: You MUST mention or reference the image content in your LinkedIn post. Even if the image seems unrelated to the topic, find a creative way to connect it to your message. This is mandatory - do not ignore the image!
    
    Post Structure:
    1. Hook (controversial or relatable opening that somehow connects to the image)
    2. Personal story/anecdote that references what you see in the image
    3. Business lesson/insight (incorporate trending topics naturally + image connection)
    4. Call to action/engagement question
    5. Hashtags (use the number specified by hashtag_level: 1-2=few(3-5), 3=moderate(6-10), 4-5=many(12-20))
    
    EXAMPLES OF IMAGE INTEGRATION:
    - If image shows people/characters: "Saw this image of [describe briefly] and it reminded me of..."
    - If image shows objects/scenes: "Looking at [describe image element], it got me thinking about..."
    - If image is unrelated: "I know this might seem random, but [describe image] actually taught me something about [topic]..."
    
    STYLE REQUIREMENTS:
    - Grammar: Follow the grammar_level setting exactly
    - Emojis: Use the exact number range specified by emoji_level
    - Hashtags: Use the exact number range specified by hashtag_level
    - Ragebait: Match the egotistical/self-absorbed level specified
    - Inspirational: Match the motivational content level specified
    - Informational: Match the educational/data content level specified
    
    IMPORTANT:
    - ALWAYS start by acknowledging what you see in the image
    - Naturally weave in 1-2 trending topics from the list provided
    - Use primarily the trending hashtags provided (they're current and popular)
    - Make it authentic LinkedIn "slop" but customized to the user's style preferences
    - Match the industry context ({context['industry']})
    - The image reference should feel natural, not forced
    
    Write the complete LinkedIn post as your response. Remember: IMAGE MENTION IS MANDATORY and STYLE PREFERENCES MUST BE FOLLOWED EXACTLY!
    """


def build_revision_prompt(context: dict, state: LinkedInAgentState, draft_number: int) -> str:
    """Build the prompt for revising content based on critique feedback with style preferences."""
    
    previous_draft = state.post_drafts[-1] if state.post_drafts else "No previous draft"
    latest_critique = state.critique_feedback[-1] if state.critique_feedback else "No feedback available"
    
    return f"""
    REVISE this LinkedIn post based on expert critique feedback:
    
    PREVIOUS DRAFT #{len(state.post_drafts)}:
    {previous_draft}
    
    EXPERT CRITIQUE FEEDBACK:
    {latest_critique}
    
    CONTEXT (MAINTAIN):
    🖼️ IMAGE CONTENT (MUST STILL REFERENCE): {context['image_description']}
    📝 USER REQUEST: {context['initial_prompt']}
    🏭 INDUSTRY CONTEXT: {context['industry']}
    
    TRENDING INFORMATION TO INCORPORATE:
    {context['trending_info']}
    
    PERSONALIZED STYLE PREFERENCES (FOLLOW EXACTLY):
    📊 Style Scores: {context['style_scores']}
    📝 Style Guidelines: {context['style_description']}
    💡 Style Examples: {context['style_examples']}
    
    REVISION INSTRUCTIONS:
    1. Address the specific improvements mentioned in the critique
    2. Fix any weaknesses identified in the feedback
    3. Enhance the strengths that were praised
    4. STILL MAINTAIN the image reference requirement
    5. STILL FOLLOW the user's style preferences exactly
    6. Keep the authentic LinkedIn "slop" style
    7. Improve engagement potential, authenticity, and algorithm optimization
    
    STYLE REQUIREMENTS (DO NOT CHANGE):
    - Grammar: Maintain grammar_level={state.grammar_level}/5
    - Emojis: Maintain emoji_level={state.emoji_level}/5
    - Hashtags: Maintain hashtag_level={state.hashtag_level}/5
    - Ragebait: Maintain ragebait_level={state.ragebait_level}/5
    - Inspirational: Maintain inspirational_level={state.inspirational_level}/5
    - Informational: Maintain informational_level={state.informational_level}/5
    
    CRITICAL: 
    - You are IMPROVING the previous draft, not starting over
    - Incorporate the specific suggestions from the critique
    - Make it more engaging while staying authentic
    - Image mention is still MANDATORY
    - Style preferences are still MANDATORY and must not change
    - Aim for higher scores in: engagement, authenticity, optimization
    
    Write the IMPROVED LinkedIn post as your response (Draft #{draft_number}):
    """


async def simple_markdown_formatter(state: LinkedInAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Format the final LinkedIn post as markdown with refinement and style summary."""
    
    start_time = time.time()
    _LOGGER.info("📝 Formatting LinkedIn post as markdown...")
    
    if not state.post_drafts:
        final_post = "No content generated."
    else:
        # Use the latest draft
        latest_draft = state.post_drafts[-1]
        total_drafts = len(state.post_drafts)
        
        # Include metadata about the research and refinement process
        research_summary = ""
        if state.industry:
            research_summary += f"**Industry**: {state.industry.title()}\n"
        
        # Add image description to research summary
        if state.image_description and state.image_description != "No image provided":
            research_summary += f"**Image Content**: {state.image_description[:100]}{'...' if len(state.image_description) > 100 else ''}\n"
        
        if state.trending_topics:
            research_summary += f"**Trending Topics Used**: {', '.join(state.trending_topics[:3])}\n"
        if state.trending_hashtags:
            research_summary += f"**Trending Hashtags Available**: {len(state.trending_hashtags)} found\n"
        
        # Add style preferences summary
        style_summary = f"""**Style Preferences Applied**:
- Grammar Level: {state.grammar_level}/5 ({['Terrible', 'Poor', 'Casual', 'Good', 'Proper'][state.grammar_level-1]})
- Emoji Level: {state.emoji_level}/5 ({['Minimal', 'Few', 'Moderate', 'Many', 'Maximum'][state.emoji_level-1]})
- Hashtag Level: {state.hashtag_level}/5 ({['Few', 'Some', 'Moderate', 'Many', 'Maximum'][state.hashtag_level-1]})
- Ragebait Level: {state.ragebait_level}/5 ({['Humble', 'Modest', 'Balanced', 'Confident', 'Egotistical'][state.ragebait_level-1]})
- Inspirational Level: {state.inspirational_level}/5 ({['Realistic', 'Practical', 'Balanced', 'Motivational', 'Dream-big'][state.inspirational_level-1]})
- Informational Level: {state.informational_level}/5 ({['Opinion-based', 'Story-focused', 'Balanced', 'Educational', 'Data-rich'][state.informational_level-1]})
"""
        
        # Add refinement process summary
        refinement_summary = ""
        if total_drafts > 1:
            refinement_summary += f"**Refinement Process**: {total_drafts} drafts created through critique feedback\n"
            
            # Extract final scores if available
            if state.critique_feedback:
                latest_critique = state.critique_feedback[-1]
                if "Overall Score:" in latest_critique:
                    import re
                    score_match = re.search(r"Overall Score: (\d+\.?\d*)/10", latest_critique)
                    if score_match:
                        final_score = score_match.group(1)
                        refinement_summary += f"**Final Quality Score**: {final_score}/10\n"
        else:
            refinement_summary += f"**Refinement Process**: Single draft (no critique loop)\n"
        
        # Simple markdown formatting
        final_post = f"""# LinkedIn Post

{latest_draft}

---

## Research Summary
{research_summary}

## Style Configuration
{style_summary}

## Refinement Summary
{refinement_summary}

**Generated by LinkedIn Slop Bot** 🤖
*Ready to copy and paste to LinkedIn!*
"""
    
    format_time = time.time() - start_time
    _LOGGER.info(f"✅ Markdown formatting completed in {format_time:.3f}s")
    
    return {
        "final_post": final_post,
        "messages": []
    } 