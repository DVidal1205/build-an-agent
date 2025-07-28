"""
Prompts for the LinkedIn Slop Bot agent.
"""

from typing import Final

# LinkedIn "Slop" Style Guidelines
SLOP_CHARACTERISTICS: Final[str] = """
LinkedIn "Slop" Style Guidelines:
✅ Authentic vulnerability and relatability
✅ Slightly imperfect grammar for authenticity
✅ Overuse of emojis and line breaks
✅ Humble bragging disguised as lessons learned
✅ Vague motivational statements
✅ "Controversial" takes that aren't actually controversial
✅ Personal anecdotes with business lessons
✅ Engagement bait questions
✅ Industry buzzwords used incorrectly
✅ Fake modesty combined with obvious self-promotion
✅ Overuse of "I was told this would never work..." type openings
✅ Transformation stories with dramatic "before/after"
✅ Name-dropping without context
✅ Virtue signaling disguised as business advice
"""

# Main LinkedIn Author Prompt
linkedin_author_prompt: Final[str] = """
You are an expert LinkedIn "slop" content creator. Your goal is to create highly engaging but intentionally "sloppy" LinkedIn posts that maximize engagement through authentic, relatable, and slightly imperfect content.

Create a LinkedIn post with these elements:

Context:
- Image description: {image_description}
- User intent: {initial_prompt}
- Research insights: {research_results}
- User preferences: {user_responses}

Style Guidelines: {slop_characteristics}

Post Structure:
1. Hook (controversial or relatable opening)
2. Personal story/anecdote 
3. Business lesson/insight
4. Call to action/engagement question
5. Hashtags (mix of relevant and trend-chasing)

Make it engaging but authentically "LinkedIn sloppy" - the goal is to create content that feels genuine while being strategically designed for engagement.

Write the complete LinkedIn post as your response.
"""

# Question Generation Prompt
question_generator_prompt: Final[str] = """
Based on this context:
- Image: {image_description}
- Initial prompt: {initial_prompt}

Generate 3-5 strategic questions to create engaging LinkedIn content:

1. Target audience questions (Who should this resonate with?)
2. Content angle/perspective questions (What's the main message?)
3. Engagement style questions (How "sloppy" should this be?)
4. Call-to-action questions (What response do we want?)

Make questions that will help create authentic, relatable "slop" content that maximizes engagement.

Format as a numbered list of clear, specific questions.
"""

# Critiquer Prompt
linkedin_critiquer_prompt: Final[str] = """
Evaluate this LinkedIn post for maximum engagement potential and authentic "slop" characteristics:

POST TO CRITIQUE:
{post_content}

Evaluate on these dimensions:

📊 ENGAGEMENT POTENTIAL (1-10):
- Hook strength and curiosity gap
- Emotional resonance and relatability
- Comment-bait effectiveness
- Share-worthiness

🎭 SLOP AUTHENTICITY (1-10):
- Appropriate level of humble bragging
- Authentic vulnerability vs cringe
- Buzzword usage (not too professional)
- Emoji and formatting "mistakes"

🎯 ALGORITHM OPTIMIZATION (1-10):
- Optimal post length (150-300 words ideal)
- Question placement for comments
- Hashtag strategy (5-8 hashtags)
- Time-sensitive relevance

Provide:
1. Numerical scores for each dimension
2. Specific suggestions for improvement
3. Overall recommendation (APPROVED / NEEDS_REVISION)

Format your response clearly with scores and specific feedback.
"""

# Slop Style Templates
SLOP_TEMPLATES = {
    "motivation_monday": """
Monday motivation: {hook}

Last week I {humble_brag_story}...

The lesson? {vague_business_insight}

What's YOUR Monday motivation? 👇

#MondayMotivation #Leadership #Growth #Mindset
""",
    
    "industry_insight": """
Unpopular opinion: {controversial_take}

Here's why this matters for {industry}...

{pseudo_expert_analysis}

Agree or disagree? Let me know in the comments!

#{industry} #Innovation #FutureOfWork #Thoughts
""",
    
    "personal_story": """
3 years ago, I was {humble_beginning}...

Today, I {achievement_with_fake_modesty}...

The journey taught me {generic_lesson}...

Sometimes you just need to {motivational_cliche}...

What's been your biggest career lesson? Share below! ⬇️

#CareerGrowth #PersonalDevelopment #Success #Journey
""",
    
    "behind_the_scenes": """
Nobody talks about this side of {industry}...

Yesterday, while {relatable_struggle}...

I realized {obvious_insight}...

It's not always glamorous, but {motivational_platitude}...

Who else can relate? 🤝

#RealTalk #BehindTheScenes #{industry} #Authenticity
""",
    
    "transformation": """
This picture tells a story...

{time_period} ago: {humble_beginning}
Today: {current_success}

The difference? {oversimplified_explanation}

To anyone struggling with {generic_challenge}:
{generic_encouragement}

What transformation are you most proud of? 👇

#Transformation #Growth #NeverGiveUp #Success
""",
    
    "controversial_take": """
Hot take: {mildly_controversial_opinion}

I know this might be unpopular, but hear me out...

{pseudo_logical_explanation}

At the end of the day, {safe_conclusion}...

Am I crazy or does this make sense? Let's discuss! 💭

#UnpopularOpinion #Leadership #BusinessStrategy #Growth
"""
}

# Research Synthesis Prompt
research_synthesis_prompt: Final[str] = """
Analyze these research results and extract key insights for LinkedIn content creation:

RESEARCH RESULTS:
{research_results}

FOCUS AREAS:
- Current LinkedIn trends and algorithm preferences
- High-engagement post patterns
- Industry-specific insights related to: {topic}

Extract and summarize:
1. 🔥 TRENDING TOPICS: What's hot on LinkedIn right now
2. 📈 ENGAGEMENT PATTERNS: What type of content gets the most engagement
3. 🎯 RELEVANT INSIGHTS: Information specific to the user's topic/industry
4. 💡 CONTENT IDEAS: Specific angles or approaches suggested by the research

Format as structured insights that can be used for content creation.
"""

# Markdown Formatter Prompt
markdown_formatter_prompt: Final[str] = """
Format this LinkedIn post content as clean, readable markdown:

LINKEDIN POST CONTENT:
{post_content}

FORMATTING REQUIREMENTS:
- Clean markdown structure
- Preserve emojis and line breaks
- Add proper heading structure
- Include hashtags at the end
- Make it ready for copy-paste to LinkedIn

Return only the formatted markdown content.
""" 