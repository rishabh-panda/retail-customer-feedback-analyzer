import json
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import OutputParserException

from config.providers import get_free_llm


class SentimentAgent:
    """Agent responsible for analyzing sentiment of customer reviews."""

    def __init__(self, temperature: float = 0.0):
        """
        Initialize the sentiment analysis agent.

        Args:
            temperature (float): Model temperature (0.0 = deterministic).

        Raises:
            ValueError: If LLM initialization fails due to configuration issues.
            ConnectionError: If provider API cannot be reached.
        """
        self.temperature = max(0.0, min(1.0, float(temperature)))
        self.llm = get_free_llm(temperature=self.temperature)

    def analyze(self, reviews: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for each review using the configured LLM.

        Args:
            reviews (List[Dict[str, str]]): List of review dicts with keys:
                - 'review': The review text
                - 'rating': Numerical or string rating

        Returns:
            List[Dict[str, Any]]: List of enriched review dicts containing:
                - 'sentiment': 'positive', 'negative', 'neutral', or 'unknown'
                - 'confidence': Float between 0.0 and 1.0
                - 'key_phrases': List of 2-3 extracted phrases
                - 'original_rating': Original rating from input
                - 'review': Original review text

        Raises:
            ValueError: If input reviews list is empty or malformed.
            TypeError: If review entries lack required keys.
        """
        if not reviews:
            raise ValueError("Reviews list cannot be empty.")

        system_prompt = SystemMessage(
            content=(
                "You are a sentiment analyst. Given a customer review and its rating, "
                "output a JSON object with keys: 'sentiment' (positive/negative/neutral), "
                "'confidence' (0-1), and 'key_phrases' (list of 2-3 short phrases). "
                "Return ONLY valid JSON, no other text."
            )
        )

        results = []

        for idx, review in enumerate(reviews):
            if not isinstance(review, dict):
                raise TypeError(f"Review at index {idx} is not a dictionary: {type(review)}")

            review_text = review.get('review', '').strip()
            rating = review.get('rating', '')

            if not review_text:
                print(f"Warning: Empty review text at index {idx}, skipping sentiment analysis.")
                results.append(self._create_fallback_result(review_text, rating))
                continue

            human_prompt = HumanMessage(
                content=f"Review: {review_text}\nRating: {rating}"
            )

            try:
                response = self.llm.invoke([system_prompt, human_prompt])

                if not response or not hasattr(response, 'content'):
                    raise OutputParserException("LLM returned empty or malformed response.")

                raw_content = response.content.strip()

                if not raw_content:
                    raise OutputParserException("LLM returned empty content.")

                parsed = self._extract_and_parse_json(raw_content)

                # Validate parsed structure
                required_keys = {'sentiment', 'confidence', 'key_phrases'}
                if not required_keys.issubset(parsed.keys()):
                    missing = required_keys - parsed.keys()
                    raise OutputParserException(f"Missing required keys in response: {missing}")

                parsed["original_rating"] = rating
                parsed["review"] = review_text
                results.append(parsed)

            except json.JSONDecodeError as json_err:
                print(f"JSON decode error for review index {idx}: {json_err}")
                results.append(self._create_fallback_result(review_text, rating, raw_content[:100]))

            except OutputParserException as parse_err:
                print(f"Parsing error for review index {idx}: {parse_err}")
                results.append(self._create_fallback_result(review_text, rating))

            except (ConnectionError, TimeoutError) as net_err:
                print(f"Network error for review index {idx}: {net_err}")
                results.append(self._create_fallback_result(review_text, rating))

            except Exception as unexpected_err:
                print(f"Unexpected error for review index {idx}: {type(unexpected_err).__name__}: {unexpected_err}")
                results.append(self._create_fallback_result(review_text, rating))

        return results

    def _extract_and_parse_json(self, raw_content: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response that may contain markdown formatting.

        Args:
            raw_content (str): Raw response content from LLM.

        Returns:
            Dict[str, Any]: Parsed JSON dictionary.

        Raises:
            OutputParserException: If no valid JSON can be extracted.
        """
        cleaned = raw_content.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        if not cleaned:
            raise OutputParserException("No JSON content found after cleaning.")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise OutputParserException(f"Failed to parse JSON: {e}") from e

    def _create_fallback_result(
        self,
        review_text: str,
        rating: Any,
        error_context: str = ""
    ) -> Dict[str, Any]:
        """
        Create a fallback result when sentiment analysis fails.

        Args:
            review_text (str): Original review text.
            rating (Any): Original rating value.
            error_context (str): Optional context about the failure.

        Returns:
            Dict[str, Any]: Fallback result with safe defaults.
        """
        return {
            "sentiment": "unknown",
            "confidence": 0.0,
            "key_phrases": [],
            "original_rating": rating,
            "review": review_text,
            "_error_context": error_context if error_context else None
        }

    def __repr__(self) -> str:
        """Developer-friendly representation of the agent."""
        return f"SentimentAgent(temperature={self.temperature})"