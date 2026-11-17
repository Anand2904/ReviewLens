"""The fixed structure every analysed review must follow.

with_structured_output(ReviewAnalysis) forces the LLM to return exactly these
fields, so the rest of the app never parses free text.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]
Category = Literal["food_quality", "delivery", "price", "staff", "packaging", "app", "other"]
Urgency = Literal["low", "medium", "high"]

# The 5 standard columns every uploaded file is converted into (see ingest.py)
STANDARD_COLUMNS = ["review_id", "date", "platform", "rating", "review_text"]
ANALYSIS_COLUMNS = ["sentiment", "category", "urgency", "language", "summary", "suggested_reply"]


class ReviewAnalysis(BaseModel):
    """What the LLM must return for ONE review."""

    sentiment: Sentiment = Field(description="Overall customer mood.")
    category: Category = Field(description="Main topic the review is about.")
    urgency: Urgency = Field(
        description=(
            "high = needs a reply today: food poisoning or health risk, refund or "
            "money problem, rude or abusive staff, wrong order, very late or cold delivery. "
            "medium = clear complaint without those. low = praise, neutral or minor remark."
        )
    )
    language: str = Field(
        description="Language of the review, e.g. English, Tamil, Tanglish, Hindi, Hinglish."
    )
    summary: str = Field(
        description="ONE English sentence, whatever language the review is written in."
    )
    suggested_reply: str = Field(
        description="Polite, ready-to-post reply in English. Apologise only for complaints."
    )


class AnalysedReview(ReviewAnalysis):
    """LLM output + the standard columns (added in code, not by the LLM)."""

    review_id: str
    date: str
    platform: str
    rating: Optional[float] = None
    review_text: str