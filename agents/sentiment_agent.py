from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict
import json
from config.providers import get_free_llm

class SentimentAgent:
    def __init__(self):
        self.llm = get_free_llm(temperature=0)
        
    def analyze(self, reviews: List[Dict[str, str]]) -> List[Dict]:
        """Analyze sentiment of each review using free LLM."""
        system_prompt = SystemMessage(
            content="You are a sentiment analyst. Given a customer review and its rating, "
                    "output a JSON object with keys: 'sentiment' (positive/negative/neutral), "
                    "'confidence' (0-1), and 'key_phrases' (list of 2-3 short phrases). "
                    "Return ONLY valid JSON, no other text."
        )
        
        results = []
        for review in reviews:
            human_prompt = HumanMessage(
                content=f"Review: {review['review']}\nRating: {review['rating']}"
            )
            try:
                response = self.llm.invoke([system_prompt, human_prompt])
                # Clean response to ensure valid JSON
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                
                parsed = json.loads(content.strip())
                parsed["original_rating"] = review["rating"]
                parsed["review"] = review["review"]
                results.append(parsed)
            except Exception as e:
                print(f"Error processing review: {e}")
                results.append({
                    "sentiment": "unknown",
                    "confidence": 0.0,
                    "key_phrases": [],
                    "original_rating": review["rating"],
                    "review": review["review"]
                })
        return results