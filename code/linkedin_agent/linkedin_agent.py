"""
Main LinkedIn Agent orchestrator using LangGraph with refinement loop.
"""

import logging
from langgraph.graph import END, START, StateGraph

from .linkedin_state import LinkedInAgentState
from .questionnaire_agent import questionnaire_agent
from .image_analyzer import image_context_analyzer
from .industry_analyzer import industry_analyzer_agent
from .linkedin_researcher import linkedin_research_agent
from .linkedin_author import linkedin_author_agent, simple_markdown_formatter
from .linkedin_critiquer import linkedin_critiquer_agent, should_continue_refining

_LOGGER = logging.getLogger(__name__)


def should_skip_image_analysis(state: LinkedInAgentState) -> str:
    """Determine if we should skip image analysis after questionnaire."""
    if state.image_path or state.image_base64:
        return "analyze_image"
    else:
        return "analyze_industry"


# Create the workflow with refinement loop and questionnaire
workflow = StateGraph(LinkedInAgentState)

# Add nodes for complete workflow with questionnaire first
workflow.add_node("questionnaire", questionnaire_agent)
workflow.add_node("image_analyzer", image_context_analyzer)
workflow.add_node("industry_analyzer", industry_analyzer_agent)
workflow.add_node("researcher", linkedin_research_agent)
workflow.add_node("author", linkedin_author_agent)
workflow.add_node("critiquer", linkedin_critiquer_agent)
workflow.add_node("formatter", simple_markdown_formatter)

# Set up the workflow edges starting with questionnaire
workflow.add_edge(START, "questionnaire")

workflow.add_conditional_edges(
    "questionnaire",
    should_skip_image_analysis,
    {
        "analyze_image": "image_analyzer",
        "analyze_industry": "industry_analyzer"
    }
)

# Main workflow: Questionnaire → Image → Industry → Research → Author → Critiquer → (Loop) → Formatter
workflow.add_edge("image_analyzer", "industry_analyzer")
workflow.add_edge("industry_analyzer", "researcher")
workflow.add_edge("researcher", "author")
workflow.add_edge("author", "critiquer")

# REFINEMENT LOOP: Critiquer decides whether to continue refining or finish
workflow.add_conditional_edges(
    "critiquer",
    should_continue_refining,
    {
        "continue": "author",  # Loop back to author for another iteration
        "finish": "formatter"  # Move to final formatting
    }
)

workflow.add_edge("formatter", END)

# Compile the graph
graph = workflow.compile()

_LOGGER.info("LinkedIn Slop Bot workflow compiled successfully with questionnaire and critique refinement loop") 