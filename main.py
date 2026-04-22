import os
import sys
from typing import TypedDict, List, Dict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from tools.review_loader import load_sample_reviews
from agents.sentiment_agent import SentimentAgent
from agents.summarizer_agent import SummarizerAgent

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------

load_dotenv()


def validate_environment() -> str:
    """
    Validate required environment variables for the selected LLM provider.

    Returns:
        str: The active provider name in uppercase.

    Raises:
        SystemExit: If required API keys are missing or invalid.
    """
    provider = os.getenv("ACTIVE_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or not api_key.strip():
            print("ERROR: GROQ_API_KEY not found or empty in .env file")
            print("Get your free key at: https://console.groq.com")
            sys.exit(1)

    elif provider == "zhipu":
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key or not api_key.strip():
            print("ERROR: ZHIPU_API_KEY not found or empty in .env file")
            print("Get your free key at: https://bigmodel.cn")
            sys.exit(1)

    else:
        print(f"ERROR: Unknown provider '{provider}'. Use 'groq' or 'zhipu'")
        sys.exit(1)

    print(f"INFO: Using {provider.upper()} provider (free tier)")
    return provider.upper()


# -----------------------------------------------------------------------------
# Graph State Definition
# -----------------------------------------------------------------------------

class AgentState(TypedDict):
    """State object passed between workflow nodes."""
    reviews: List[Dict[str, str]]
    analyses: List[Dict]
    summary: Dict


# -----------------------------------------------------------------------------
# Workflow Node Functions
# -----------------------------------------------------------------------------

def load_reviews(state: AgentState) -> AgentState:
    """
    Load sample reviews using the review loader tool.

    Args:
        state (AgentState): Current workflow state.

    Returns:
        AgentState: Updated state with reviews populated.
    """
    print("INFO: Loading sample reviews...")

    try:
        reviews = load_sample_reviews.invoke({})
        if not reviews:
            raise ValueError("No reviews returned from loader")

        state["reviews"] = reviews

    except Exception as e:
        print(f"ERROR: Failed to load reviews: {e}")
        state["reviews"] = []

    return state


def analyze_sentiment(state: AgentState) -> AgentState:
    """
    Run sentiment analysis on loaded reviews.

    Args:
        state (AgentState): Current workflow state with reviews.

    Returns:
        AgentState: Updated state with sentiment analyses populated.
    """
    print("INFO: Analyzing sentiment (this may take a moment)...")

    if not state.get("reviews"):
        print("WARNING: No reviews to analyze. Skipping sentiment analysis.")
        state["analyses"] = []
        return state

    try:
        agent = SentimentAgent()
        state["analyses"] = agent.analyze(state["reviews"])

    except Exception as e:
        print(f"ERROR: Sentiment analysis failed: {e}")
        state["analyses"] = []

    return state


def generate_summary(state: AgentState) -> AgentState:
    """
    Generate final summary from sentiment analyses.

    Args:
        state (AgentState): Current workflow state with analyses.

    Returns:
        AgentState: Updated state with summary populated.
    """
    print("INFO: Generating insights summary...")

    if not state.get("analyses"):
        print("WARNING: No analyses available. Generating empty summary.")
        state["summary"] = _get_empty_summary()
        return state

    try:
        agent = SummarizerAgent()
        state["summary"] = agent.summarize(state["analyses"])

    except Exception as e:
        print(f"ERROR: Summary generation failed: {e}")
        state["summary"] = _get_empty_summary()

    return state


def _get_empty_summary() -> Dict:
    """Return an empty summary structure for fallback scenarios."""
    return {
        "overall_sentiment_distribution": {},
        "top_positive_topics": [],
        "top_negative_topics": [],
        "recommended_actions": ["Unable to generate summary - check logs"]
    }


# -----------------------------------------------------------------------------
# Workflow Construction
# -----------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Returns:
        StateGraph: Compiled workflow graph.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("load", load_reviews)
    workflow.add_node("analyze", analyze_sentiment)
    workflow.add_node("summarize", generate_summary)

    workflow.set_entry_point("load")
    workflow.add_edge("load", "analyze")
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", "end")

    workflow.add_node("end", lambda state: state)  # Terminal node

    return workflow.compile()


# -----------------------------------------------------------------------------
# Output Formatting
# -----------------------------------------------------------------------------

def display_results(final_state: AgentState) -> None:
    """
    Display analysis results in a formatted console output.

    Args:
        final_state (AgentState): Final workflow state with all results.
    """
    separator = "=" * 60
    thin_separator = "-" * 60

    print("\n" + separator)
    print("RETAIL CUSTOMER FEEDBACK ANALYZER (Zero-Cost Edition)")
    print(separator)

    analyses = final_state.get("analyses", [])
    if not analyses:
        print("\nWARNING: No analysis results to display.")
        return

    print("\nINDIVIDUAL REVIEW ANALYSIS:")
    for idx, analysis in enumerate(analyses, 1):
        sentiment = analysis.get("sentiment", "unknown").upper()
        confidence = analysis.get("confidence", 0.0)
        key_phrases = analysis.get("key_phrases", [])
        rating = analysis.get("original_rating", "N/A")

        print(f"\n  Review {idx}:")
        print(f"    - Sentiment: {sentiment} (confidence: {confidence})")
        print(f"    - Key phrases: {', '.join(key_phrases) if key_phrases else 'none'}")
        print(f"    - Original rating: {rating} stars")

    print("\n" + thin_separator)
    print("AGGREGATED INSIGHTS")
    print(thin_separator)

    summary = final_state.get("summary", {})
    distribution = summary.get("overall_sentiment_distribution", {})
    positive_topics = summary.get("top_positive_topics", [])
    negative_topics = summary.get("top_negative_topics", [])
    actions = summary.get("recommended_actions", [])

    print(f"\n  - Sentiment distribution: {distribution}")

    print("\n  - Top positive topics:")
    for topic in positive_topics:
        print(f"    + {topic}")

    print("\n  - Top negative topics:")
    for topic in negative_topics:
        print(f"    - {topic}")

    print("\n  - Recommended actions:")
    for i, action in enumerate(actions, 1):
        print(f"    {i}. {action}")

    print("\n" + separator)
    print("Analysis complete using 100% free LLM API")
    print(separator)


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the customer feedback analyzer."""
    try:
        validate_environment()
    except SystemExit:
        sys.exit(1)

    workflow = build_workflow()
    initial_state: AgentState = {
        "reviews": [],
        "analyses": [],
        "summary": {}
    }

    try:
        final_state = workflow.invoke(initial_state)
        display_results(final_state)

    except Exception as e:
        print(f"\nERROR: Workflow execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()