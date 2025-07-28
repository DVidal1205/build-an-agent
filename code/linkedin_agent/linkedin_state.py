"""LinkedIn Agent State model for the LinkedIn Slop Bot."""

from typing import Annotated, Any, Sequence
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class LinkedInAgentState(BaseModel):
    """State model for the LinkedIn Slop Bot agent workflow."""
    
    # Input
    image_path: str | None = None
    image_base64: str | None = None
    initial_prompt: str
    
    # Style Questionnaire (1-5 scales)
    grammar_level: int = 3  # 1=terrible, 5=proper
    emoji_level: int = 3    # 1=little, 5=a lot  
    hashtag_level: int = 3  # 1=little, 5=a lot
    ragebait_level: int = 2 # 1=little, 5=lot (egotistical/self-absorbed)
    inspirational_level: int = 3  # 1-5
    informational_level: int = 3  # 1-5
    
    # Image Analysis
    image_description: str | None = None
    visual_elements: list[str] = []
    
    # Industry Analysis
    industry: str | None = None  # Determined industry (software, finance, etc.)
    
    # Question Phase
    clarifying_questions: list[str] = []
    user_responses: dict[str, str] = {}
    
    # Research Phase
    trending_topics: list[str] = []
    trending_hashtags: list[str] = []
    research_results: str | None = None
    
    # Content Creation
    post_drafts: list[str] = []
    critique_feedback: list[str] = []
    final_post: str | None = None
    
    # Metadata
    target_engagement_style: str = "relatable_slop"
    post_type: str = "general"  # general, motivation, industry_insight, personal_story
    
    # Messages for LangGraph workflow
    messages: Annotated[Sequence[Any], add_messages] = []
    
    class Config:
        arbitrary_types_allowed = True 