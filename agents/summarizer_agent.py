import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import OutputParserException

from config.providers import get_free_llm


class SummarizerAgent:
    """Agent responsible for generating summary insights from sentiment analyses."""

    def __init__(self, temperature: float = 0.2):
        """
        Initialize the summarizer agent.

        Args:
            temperature (float): Model temperature (0.2 = balanced creativity).
                                 Clamped between 0.0 and 1.0.

        Raises:
            ValueError: If LLM initialization fails.
            ConnectionError: If provider API cannot be reached.
        """
        self.temperature = max(0.0, min(1.0, float(temperature)))
        self.llm = get_free_llm(temperature=self.temperature)

    def summarize(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate actionable summary insights from sentiment analysis results.

        Args:
            analyses (List[Dict[str, Any]]): List of sentiment analysis results,
                                             each containing at minimum:
                - 'sentiment': positive/negative/neutral
                - 'confidence': float
                - 'key_phrases': list of strings
                - 'original_rating': rating value
                - 'review': original review text

        Returns:
            Dict[str, Any]: Summary containing:
                - 'overall_sentiment_distribution': Dict with sentiment counts
                - 'top_positive_topics': List of 3 positive topics
                - 'top_negative_topics': List of 3 negative topics
                - 'recommended_actions': List of 3 actionable suggestions

        Raises:
            ValueError: If analyses list is empty or malformed.
            TypeError: If analyses entries are not dictionaries.
        """
        if not analyses:
            raise ValueError("Analyses list cannot be empty.")

        if not all(isinstance(item, dict) for item in analyses):
            raise TypeError("All items in analyses list must be dictionaries.")

        system_prompt = SystemMessage(
            content=(
                "You are a retail analyst. Based on the sentiment analyses of customer reviews, "
                "produce a JSON summary with keys: 'overall_sentiment_distribution' (object with counts), "
                "'top_positive_topics' (list of 3 topics), 'top_negative_topics' (list of 3 topics), "
                "'recommended_actions' (list of 3 actionable suggestions). "
                "Return ONLY valid JSON, no other text."
            )
        )

        # Prepare a clean, serializable subset of data for the prompt
        sanitized_analyses = self._sanitize_for_prompt(analyses)

        try:
            serialized_input = json.dumps(sanitized_analyses, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as serialization_err:
            raise ValueError(f"Failed to serialize analyses data: {serialization_err}") from serialization_err

        human_prompt = HumanMessage(
            content=f"Sentiment analyses:\n{serialized_input}"
        )

        try:
            response = self.llm.invoke([system_prompt, human_prompt])

            if not response or not hasattr(response, 'content'):
                raise OutputParserException("LLM returned empty or malformed response.")

            raw_content = response.content.strip()

            if not raw_content:
                raise OutputParserException("LLM returned empty content.")

            parsed_summary = self._extract_and_parse_json(raw_content)

            # Validate expected structure
            expected_keys = {
                'overall_sentiment_distribution',
                'top_positive_topics',
                'top_negative_topics',
                'recommended_actions'
            }

            missing_keys = expected_keys - parsed_summary.keys()
            if missing_keys:
                raise OutputParserException(f"Missing expected keys in response: {missing_keys}")

            # Validate data types
            if not isinstance(parsed_summary.get('overall_sentiment_distribution'), dict):
                parsed_summary['overall_sentiment_distribution'] = {}

            if not isinstance(parsed_summary.get('top_positive_topics'), list):
                parsed_summary['top_positive_topics'] = []

            if not isinstance(parsed_summary.get('top_negative_topics'), list):
                parsed_summary['top_negative_topics'] = []

            if not isinstance(parsed_summary.get('recommended_actions'), list):
                parsed_summary['recommended_actions'] = []

            return parsed_summary

        except json.JSONDecodeError as json_err:
            print(f"JSON decode error in summary: {json_err}")
            return self._create_fallback_summary(
                f"JSON parsing failed: {json_err}"
            )

        except OutputParserException as parse_err:
            print(f"Parsing error in summary: {parse_err}")
            return self._create_fallback_summary(str(parse_err))

        except (ConnectionError, TimeoutError) as net_err:
            print(f"Network error in summary generation: {net_err}")
            return self._create_fallback_summary(f"Network error: {net_err}")

        except Exception as unexpected_err:
            print(f"Unexpected error in summary generation: {type(unexpected_err).__name__}: {unexpected_err}")
            return self._create_fallback_summary(f"Unexpected error: {unexpected_err}")

    def _sanitize_for_prompt(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract only essential fields to keep prompt size manageable.

        Args:
            analyses (List[Dict[str, Any]]): Full analysis results.

        Returns:
            List[Dict[str, Any]]: Sanitized analyses with minimal necessary data.
        """
        sanitized = []

        for item in analyses:
            sanitized_item = {
                'sentiment': item.get('sentiment', 'unknown'),
                'confidence': item.get('confidence', 0.0),
                'key_phrases': item.get('key_phrases', [])[:5],  # Limit to 5 phrases
                'original_rating': item.get('original_rating', 'N/A')
            }

            # Truncate review text if too long (prevents token overflow)
            review_text = item.get('review', '')
            if isinstance(review_text, str) and len(review_text) > 500:
                review_text = review_text[:497] + '...'

            sanitized_item['review'] = review_text
            sanitized.append(sanitized_item)

        return sanitized

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
        elif cleaned.startswith("```"):
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

    def _create_fallback_summary(self, error_reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a safe fallback summary when generation fails.

        Args:
            error_reason (Optional[str]): Reason for fallback (for debugging).

        Returns:
            Dict[str, Any]: Fallback summary with safe defaults.
        """
        fallback = {
            "overall_sentiment_distribution": {},
            "top_positive_topics": [],
            "top_negative_topics": [],
            "recommended_actions": [
                "Unable to generate summary - check API connection and try again"
            ]
        }

        if error_reason:
            fallback["_error_reason"] = error_reason

        return fallback

    def __repr__(self) -> str:
        """Developer-friendly representation of the agent."""
        return f"SummarizerAgent(temperature={self.temperature})"