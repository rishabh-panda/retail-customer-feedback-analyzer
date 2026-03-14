from langchain.tools import tool
from typing import List, Dict

@tool
def load_sample_reviews() -> List[Dict[str, str]]:
    """Load sample retail customer reviews for analysis."""
    return [
        {"review": "The product arrived late and the quality was poor. Very disappointed.", "rating": 1},
        {"review": "Good value for money, but the packaging was damaged.", "rating": 3},
        {"review": "Absolutely love this! Fast shipping and excellent quality.", "rating": 5},
        {"review": "It's okay, does the job but nothing special.", "rating": 3},
        {"review": "Terrible customer service, will not buy again.", "rating": 1},
        {"review": "Perfect fit and great material. Will order again!", "rating": 5},
        {"review": "The size runs small, but customer service was helpful.", "rating": 4},
    ]