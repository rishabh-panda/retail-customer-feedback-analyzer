import random
from typing import List, Dict, Optional

from langchain.tools import tool


@tool
def load_sample_reviews(category: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load realistic sample customer reviews for retail product analysis.

    Args:
        category (Optional[str]): Filter reviews by product category.
                                 Options: 'electronics', 'clothing', 'home', 'beauty', None (all)
        limit (Optional[int]): Maximum number of reviews to return.

    Returns:
        List[Dict[str, str]]: List of review objects with keys 'review' and 'rating'.

    Raises:
        ValueError: If category is invalid or limit is less than 1.
    """
    valid_categories = {'electronics', 'clothing', 'home', 'beauty', None}

    if category is not None and category not in valid_categories:
        raise ValueError(f"Invalid category '{category}'. Choose from: electronics, clothing, home, beauty")

    if limit is not None and limit < 1:
        raise ValueError("Limit must be greater than 0")

    # Comprehensive review database with realistic entries
    review_pool = _build_review_pool()

    # Filter by category if specified
    if category:
        filtered_reviews = [r for r in review_pool if r.get('category') == category]
    else:
        filtered_reviews = review_pool

    # Remove internal category field before returning
    cleaned_reviews = [
        {'review': item['review'], 'rating': item['rating']}
        for item in filtered_reviews
    ]

    # Apply limit if specified
    if limit and limit < len(cleaned_reviews):
        # Shuffle a copy to avoid bias, then slice
        shuffled = cleaned_reviews.copy()
        random.shuffle(shuffled)
        return shuffled[:limit]

    return cleaned_reviews


def _build_review_pool() -> List[Dict[str, str]]:
    """Build the internal pool of realistic reviews across categories."""
    return [
        # Electronics reviews
        {
            "review": "Battery life is amazing, lasts almost 3 days with moderate use. Screen is crisp and bright. Only downside is the charger is too short.",
            "rating": 4,
            "category": "electronics"
        },
        # {
        #     "review": "Stopped working after 2 weeks. Customer support took forever to respond. Waste of money.",
        #     "rating": 1,
        #     "category": "electronics"
        # },
        # {
        #     "review": "Decent for the price. Setup was easy but the app crashes sometimes. Would recommend for casual users.",
        #     "rating": 3,
        #     "category": "electronics"
        # },
        # {
        #     "review": "Best purchase this year! Noise cancellation is incredible and the fit is perfect. Shipping was fast too.",
        #     "rating": 5,
        #     "category": "electronics"
        # },
        # {
        #     "review": "The interface is laggy and the instructions were poorly translated. Returning it tomorrow.",
        #     "rating": 2,
        #     "category": "electronics"
        # },
        # {
        #     "review": "Works exactly as described. Nothing fancy but reliable. Been using daily for 3 months now.",
        #     "rating": 4,
        #     "category": "electronics"
        # },

        # # Clothing reviews
        # {
        #     "review": "Fabric feels cheap and it shrank after first wash. Definitely not worth the $40 I paid.",
        #     "rating": 2,
        #     "category": "clothing"
        # },
        # {
        #     "review": "Super comfortable and looks great. I ordered a size up based on other reviews and it fits perfectly.",
        #     "rating": 5,
        #     "category": "clothing"
        # },
        # {
        #     "review": "Color is slightly different than the photo but still nice. Runs a bit small for athletic builds.",
        #     "rating": 3,
        #     "category": "clothing"
        # },
        # {
        #     "review": "Zipper broke on the third wear. Customer service offered a 10% discount on next purchase instead of refund. Frustrating.",
        #     "rating": 1,
        #     "category": "clothing"
        # },
        # {
        #     "review": "Good quality for fast fashion. Stitching is decent and it's very soft. Will buy other colors.",
        #     "rating": 4,
        #     "category": "clothing"
        # },
        # {
        #     "review": "Too sheer to wear in public without layers. Returning it, but return process was easy at least.",
        #     "rating": 2,
        #     "category": "clothing"
        # },

        # # Home & kitchen reviews
        # {
        #     "review": "This blender is a beast. Crushes ice perfectly and easy to clean. My morning smoothies are so much better now.",
        #     "rating": 5,
        #     "category": "home"
        # },
        # {
        #     "review": "Looks nice but the finish scratches if you just look at it wrong. Already showing wear after one week.",
        #     "rating": 2,
        #     "category": "home"
        # },
        # {
        #     "review": "Assembly took about an hour with two people. Instructions were okay but some holes didn't align perfectly. It's sturdy now that it's up.",
        #     "rating": 3,
        #     "category": "home"
        # },
        # {
        #     "review": "Absolute game changer for small apartments. Folds flat and stores under the bed. Would buy again.",
        #     "rating": 5,
        #     "category": "home"
        # },
        # {
        #     "review": "The non-stick coating started peeling after a month. Concerned about safety now. Requested refund.",
        #     "rating": 1,
        #     "category": "home"
        # },
        # {
        #     "review": "Average quality but arrived earlier than expected. Good enough for guest room use.",
        #     "rating": 3,
        #     "category": "home"
        # },

        # # Beauty & personal care reviews
        # {
        #     "review": "My skin has never looked better. This serum faded my dark spots within 3 weeks. Holy grail product.",
        #     "rating": 5,
        #     "category": "beauty"
        # },
        # {
        #     "review": "Broke me out badly. Maybe my skin is too sensitive for the ingredients. Wanted to love it.",
        #     "rating": 2,
        #     "category": "beauty"
        # },
        # {
        #     "review": "Nice scent but doesn't last more than 2 hours. Fine for gym or errands but not for work.",
        #     "rating": 3,
        #     "category": "beauty"
        # },
        # {
        #     "review": "The brush is so soft and applies foundation flawlessly. Machine washable and still looks new after 5 washes.",
        #     "rating": 5,
        #     "category": "beauty"
        # },
        # {
        #     "review": "Packaging leaked during shipping and half the product was wasted. Customer support sent a replacement quickly though.",
        #     "rating": 3,
        #     "category": "beauty"
        # },
        # {
        #     "review": "Made my hair greasy even with tiny amount. Not suitable for fine hair types. Returning.",
        #     "rating": 2,
        #     "category": "beauty"
        # },

        # # Mixed/general reviews (no category filter)
        # {
        #     "review": "The product arrived late and the quality was poor. Very disappointed.",
        #     "rating": 1,
        #     "category": None
        # },
        # {
        #     "review": "Good value for money, but the packaging was damaged.",
        #     "rating": 3,
        #     "category": None
        # },
        # {
        #     "review": "Absolutely love this! Fast shipping and excellent quality.",
        #     "rating": 5,
        #     "category": None
        # },
        # {
        #     "review": "It's okay, does the job but nothing special.",
        #     "rating": 3,
        #     "category": None
        # },
        # {
        #     "review": "Terrible customer service, will not buy again.",
        #     "rating": 1,
        #     "category": None
        # },
        # {
        #     "review": "Perfect fit and great material. Will order again!",
        #     "rating": 5,
        #     "category": None
        # },
        # {
        #     "review": "The size runs small, but customer service was helpful.",
        #     "rating": 4,
        #     "category": None
        # },
        # {
        #     "review": "Cheap price but you get what you pay for. It works but feels flimsy.",
        #     "rating": 3,
        #     "category": None
        # },
        # {
        #     "review": "Exceeded my expectations in every way. Highly recommend to friends and family.",
        #     "rating": 5,
        #     "category": None
        # },
        # {
        #     "review": "Missing parts in the box. Had to contact support twice to get them sent. Annoying process.",
        #     "rating": 2,
        #     "category": None
        # }
    ]


# Optional: Helper function to get review statistics
def get_review_summary(reviews: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Calculate basic statistics from review ratings.

    Args:
        reviews (List[Dict[str, str]]): List of reviews with 'rating' field.

    Returns:
        Dict[str, float]: Statistics including average rating, count, and distribution.

    Raises:
        ValueError: If reviews list is empty.
    """
    if not reviews:
        raise ValueError("Reviews list cannot be empty")

    ratings = []
    for review in reviews:
        try:
            rating = int(review.get('rating', 0))
            if 1 <= rating <= 5:
                ratings.append(rating)
        except (ValueError, TypeError):
            continue

    if not ratings:
        raise ValueError("No valid ratings found in reviews")

    return {
        "average_rating": round(sum(ratings) / len(ratings), 2),
        "total_reviews": len(ratings),
        "min_rating": min(ratings),
        "max_rating": max(ratings),
        "percent_positive": round((sum(1 for r in ratings if r >= 4) / len(ratings)) * 100, 1),
        "percent_negative": round((sum(1 for r in ratings if r <= 2) / len(ratings)) * 100, 1)
    }


def __repr__(self) -> str:
    return "load_sample_reviews(tool)"