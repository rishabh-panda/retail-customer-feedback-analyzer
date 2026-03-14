from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict
import json
from config.providers import get_free_llm

class SummarizerAgent:
    def __init__(self):
        self.llm = get_free_llm(temperature=0.2)
        
    def summarize(self, analyses: List[Dict]) -> Dict:
        """Generate actionable insights from sentiment analyses using free LLM."""
        system_prompt = SystemMessage(
            content="You are a retail analyst. Based on the sentiment analyses of customer reviews, "
                    "produce a JSON summary with keys: 'overall_sentiment_distribution' (object with counts), "
                    "'top_positive_topics' (list of 3 topics), 'top_negative_topics' (list of 3 topics), "
                    "'recommended_actions' (list of 3 actionable suggestions). "
                    "Return ONLY valid JSON, no other text."
        )
        
        human_prompt = HumanMessage(
            content=f"Sentiment analyses:\n{json.dumps(analyses, indent=2)}"
        )
        
        try:
            response = self.llm.invoke([system_prompt, human_prompt])
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            print(f"Error generating summary: {e}")
            return {
                "overall_sentiment_distribution": {},
                "top_positive_topics": [],
                "top_negative_topics": [],
                "recommended_actions": ["Unable to generate summary - check API connection"]
            }