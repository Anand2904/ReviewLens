"""Per-review analysis chain (the 'map' half of map-then-reduce).

prompt | llm.with_structured_output(ReviewAnalysis)  ->  ReviewAnalysis object
Input is the standardised table made by ingest.normalise().
"""
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm
from schemas import ANALYSIS_COLUMNS, STANDARD_COLUMNS, AnalysedReview, ReviewAnalysis

SYSTEM_PROMPT = """You analyse customer reviews for restaurants and food-delivery businesses
(dine-in, Zomato, Swiggy, Google Maps, the brand's own app...).
Reviews may be in English, Tamil script, Tanglish (Tamil written in English letters, e.g.
"saapadu nalla irundhuchu but bill la extra charge pottanga"), Hindi, Hinglish or other
languages. Understand all of them. ALWAYS write summary and suggested_reply in English.

Rules:
- sentiment: use the text first; the star rating (if given) is a supporting signal only.
- category: pick the SINGLE main issue (or main praise).
  food_quality = taste, freshness, undercooked, hygiene, quantity of food.
  delivery = late, cold on arrival, rider behaviour. price = cost, extra charges on the bill.
  staff = rude or helpful people. packaging = leaks, spills, box. app = ordering app,
  payment failure, wrong order shown in app. other = anything else.
- language: name the language, e.g. English, Tamil, Tanglish, Hindi, Hinglish.
  Tamil = Tamil script. Tanglish = Tamil words in English letters (romba, nalla, irundhuchu,
  la, pottanga, semma, kammi, ah, venum). English mixed with those words is Tanglish.
- urgency: high if ANY of these is true:
    * food poisoning, sickness, or any health risk (including raw or undercooked food)
    * refund wanted, or money deducted / wrongly charged
    * rude, abusive or disrespectful staff (ALWAYS high)
    * wrong order delivered
    * delivery 45+ minutes late, or food arrived cold or spoiled
  medium = a clear complaint with none of the above (small extra charge, spill, small portion).
  low = praise or neutral remarks.
- summary: exactly one short English sentence. Keep concrete facts. If the review gives an
  order time and an arrival time, work out the delay (8pm to 9:10 = 70 minutes late).
- suggested_reply: 1-3 polite sentences, ready to post. For complaints, apologise and ask
  them to DM their order ID. For praise, thank them. Never promise a refund amount.
  Reply in English only."""

HUMAN_PROMPT = """Platform: {platform}
Rating: {rating_text}
Review: {review_text}"""


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    llm = get_llm().with_structured_output(ReviewAnalysis)
    return prompt | llm


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Table is missing columns: {', '.join(missing)}. "
                         "Run it through ingest.normalise() first.")


def _rating_text(v) -> str:
    return "not provided" if pd.isna(v) else f"{v:g}/5"


def analyse_reviews(df: pd.DataFrame, max_concurrency: int = 5, chain=None) -> pd.DataFrame:
    """Run the chain over every row; return standard cols + analysis + your original extras."""
    validate_columns(df)
    chain = chain or build_chain()
    df = df.reset_index(drop=True)

    inputs = [
        {"platform": r.platform, "rating_text": _rating_text(r.rating), "review_text": r.review_text}
        for r in df.itertuples()
    ]
    # return_exceptions=True: one bad review must not kill the whole batch
    results = chain.batch(inputs, config={"max_concurrency": max_concurrency},
                          return_exceptions=True)

    rows = []
    for res in results:
        if isinstance(res, Exception):
            # keep the row visible and flag it for a human instead of dropping it
            res = ReviewAnalysis(
                sentiment="neutral", category="other", urgency="medium", language="Unknown",
                summary=f"[Analysis failed: {type(res).__name__}]", suggested_reply="")
        rows.append(res.model_dump())
    analysis = pd.DataFrame(rows, columns=ANALYSIS_COLUMNS)
    extras = [c for c in df.columns if c not in STANDARD_COLUMNS]
    return pd.concat([df[STANDARD_COLUMNS], analysis, df[extras]], axis=1)