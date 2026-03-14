import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from tools.review_loader import load_sample_reviews
from agents.sentiment_agent import SentimentAgent
from agents.summarizer_agent import SummarizerAgent

load_dotenv()

# Verify provider is set
provider = os.getenv("ACTIVE_PROVIDER", "groq")
if provider == "groq" and not os.getenv("GROQ_API_KEY"):
    print("⚠️  GROQ_API_KEY not found in .env file")
    print("Get your free key at: https://console.groq.com")
    exit(1)
elif provider == "zhipu" and not os.getenv("ZHIPU_API_KEY"):
    print("⚠️  ZHIPU_API_KEY not found in .env file")
    print("Get your free key at: https://bigmodel.cn")
    exit(1)

print(f"✅ Using {provider.upper()} provider (100% free)")

# Define graph state
class AgentState(TypedDict):
    reviews: List[Dict[str, str]]
    analyses: List[Dict]
    summary: Dict

# Node functions
def load_reviews(state: AgentState) -> AgentState:
    """Load reviews using the tool."""
    print("📥 Loading sample reviews...")
    state["reviews"] = load_sample_reviews.invoke({})
    return state

def analyze_sentiment(state: AgentState) -> AgentState:
    """Run sentiment analysis on reviews."""
    print("🔍 Analyzing sentiment (this may take a moment)...")
    agent = SentimentAgent()
    state["analyses"] = agent.analyze(state["reviews"])
    return state

def generate_summary(state: AgentState) -> AgentState:
    """Generate final summary."""
    print("📊 Generating insights summary...")
    agent = SummarizerAgent()
    state["summary"] = agent.summarize(state["analyses"])
    return state

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("load", load_reviews)
workflow.add_node("analyze", analyze_sentiment)
workflow.add_node("summarize", generate_summary)

workflow.set_entry_point("load")
workflow.add_edge("load", "analyze")
workflow.add_edge("analyze", "summarize")
workflow.add_edge("summarize", END)

app = workflow.compile()

# Run
if __name__ == "__main__":
    initial_state = {"reviews": [], "analyses": [], "summary": {}}
    
    print("\n" + "="*60)
    print("🛍️  RETAIL CUSTOMER FEEDBACK ANALYZER (Zero-Cost Edition)")
    print("="*60)
    
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)
    
    print("\n📝 Individual Review Analysis:")
    for idx, a in enumerate(final_state["analyses"], 1):
        print(f"\n  Review {idx}:")
        print(f"    • Sentiment: {a['sentiment'].upper()} (confidence: {a['confidence']})")
        print(f"    • Key phrases: {', '.join(a['key_phrases'])}")
        print(f"    • Original rating: {a['original_rating']}★")
    
    print("\n" + "-"*60)
    print("📈 AGGREGATED INSIGHTS")
    print("-"*60)
    
    summary = final_state["summary"]
    
    dist = summary.get('overall_sentiment_distribution', {})
    print(f"\n  • Sentiment distribution: {dist}")
    
    print("\n  • Top positive topics:")
    for topic in summary.get('top_positive_topics', []):
        print(f"    ✓ {topic}")
    
    print("\n  • Top negative topics:")
    for topic in summary.get('top_negative_topics', []):
        print(f"    ✗ {topic}")
    
    print("\n  💡 Recommended actions:")
    for i, action in enumerate(summary.get("recommended_actions", []), 1):
        print(f"    {i}. {action}")
    
    print("\n" + "="*60)
    print("✅ Analysis complete using 100% free LLM API")
    print("="*60)