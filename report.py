"""Weekly insights report (the 'reduce' half of map-then-reduce).

Division of labour:
  * pandas  -> does the COUNTING (computers are exact at maths)
  * LLM     -> does the READING: groups reviews into themes and writes action items
The LLM returns review_ids per theme; WE count them, so numbers are never made up.
"""
from typing import Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm import get_llm


class Theme(BaseModel):
    title: str = Field(description="Short theme name, e.g. 'Late delivery'.")
    review_ids: list[str] = Field(description="IDs of every review that belongs to this theme.")


class WeeklyInsights(BaseModel):
    top_complaints: list[Theme] = Field(description="Up to 3 biggest complaint themes.")
    top_praises: list[Theme] = Field(description="Up to 2 biggest praise themes.")
    action_items: list[str] = Field(description="Exactly 3 specific, practical actions.")


SYSTEM_PROMPT = """You are an operations analyst for a restaurant / food-delivery business.
You get one line per analysed review: id | platform | [extra fields] | sentiment | category |
urgency | summary. Extra fields (such as outlet=..., city=...) appear only if the data has them.

Your job:
1. top_complaints: group the negative (and complaint-style neutral) reviews into at most
   3 themes. Pick the themes with the MOST reviews. Name each theme in 2-4 words.
2. top_praises: group the positive reviews into at most 2 themes the same way.
3. action_items: exactly 3 specific actions a manager can do this week. Each must fix a
   complaint theme above, mention the detail (time slot, platform, outlet, charge, etc.;
   use the extra fields when a pattern is clear) and be one sentence starting with a verb.

Rules:
- Put EVERY review whose main problem matches the theme in review_ids.
- Each review id may appear in ONLY ONE theme: choose its main problem, so the themes
  never add up to more reviews than exist.
- Only include a review if its summary clearly matches the theme. Do not stretch a theme.
- Use only ids that appear in the data. Never invent reviews or numbers."""

HUMAN_PROMPT = "Analysed reviews:\n{review_lines}"


def build_report_chain():
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt | get_llm().with_structured_output(WeeklyInsights)


def segment_hotspots(df: pd.DataFrame, segments: list, min_reviews: int = 3, top: int = 3) -> dict:
    """For each segment column (branch, city...), which values have the highest % negative."""
    out = {}
    for seg in segments or []:
        if seg not in df.columns:
            continue
        g = df.groupby(seg).agg(n=("sentiment", "size"),
                                neg=("sentiment", lambda s: int((s == "negative").sum())))
        g = g[(g["n"] >= min_reviews) & (g["neg"] > 0)]
        if g.empty:
            continue
        g["pct"] = (g["neg"] / g["n"] * 100).round().astype(int)
        g = g.sort_values(["pct", "n"], ascending=False).head(top)
        out[seg] = [(str(i), int(r.n), int(r.pct)) for i, r in g.iterrows()]
    return out


def compute_stats(df: pd.DataFrame, segments: Optional[list] = None) -> dict:
    """All the exact numbers, straight from pandas."""
    pct = (df["sentiment"].value_counts(normalize=True) * 100).round().astype(int)
    dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df else pd.Series(dtype="datetime64[ns]")
    rating = pd.to_numeric(df["rating"], errors="coerce") if "rating" in df else pd.Series(dtype=float)
    return {
        "total": len(df),
        "positive": int(pct.get("positive", 0)),
        "neutral": int(pct.get("neutral", 0)),
        "negative": int(pct.get("negative", 0)),
        "urgent": int((df["urgency"] == "high").sum()),
        "avg_rating": None if rating.notna().sum() == 0 else round(float(rating.mean()), 2),
        "by_category": df["category"].value_counts().to_dict(),
        "date_from": None if dates.notna().sum() == 0 else dates.min(),
        "date_to": None if dates.notna().sum() == 0 else dates.max(),
        "hotspots": segment_hotspots(df, segments),
    }


def _review_lines(df: pd.DataFrame, segments: Optional[list]) -> str:
    segments = [s for s in (segments or []) if s in df.columns]
    lines = []
    for _, r in df.iterrows():
        extra = (" | " + "; ".join(f"{s}={r[s]}" for s in segments)) if segments else ""
        lines.append(f"{r['review_id']} | {r['platform']}{extra} | {r['sentiment']} | "
                     f"{r['category']} | {r['urgency']} | {r['summary']}")
    return "\n".join(lines)


def _finish_theme(theme: Theme, df: pd.DataFrame) -> dict:
    """Keep only real ids, then count them ourselves and find the main platform."""
    real = set(df["review_id"])
    valid = [i for i in dict.fromkeys(theme.review_ids) if i in real]
    sub = df[df["review_id"].isin(valid)]
    top_platform = sub["platform"].mode().iloc[0] if len(sub) else ""
    return {"title": theme.title, "count": len(valid), "review_ids": valid,
            "top_platform": top_platform}


def generate_insights(df: pd.DataFrame, chain=None, segments: Optional[list] = None) -> dict:
    chain = chain or build_report_chain()
    raw: WeeklyInsights = chain.invoke({"review_lines": _review_lines(df, segments)})
    complaints = sorted((_finish_theme(t, df) for t in raw.top_complaints),
                        key=lambda t: -t["count"])[:3]
    praises = sorted((_finish_theme(t, df) for t in raw.top_praises),
                     key=lambda t: -t["count"])[:2]
    return {"complaints": [c for c in complaints if c["count"]],
            "praises": [p for p in praises if p["count"]],
            "actions": raw.action_items[:3]}


def _d(ts) -> str:
    return f"{ts.day} {ts:%b}"


def render_report(stats: dict, insights: dict) -> str:
    """Plain-text report in the same shape as 'Output B' of the brief."""
    s, L = stats, []
    if s["date_from"] is not None:
        weekly = (s["date_to"] - s["date_from"]).days <= 7
        L.append(f"{'WEEKLY ' if weekly else ''}FEEDBACK REPORT  |  "
                 f"{_d(s['date_from'])} to {_d(s['date_to'])} {s['date_to'].year}")
    else:
        L.append("FEEDBACK REPORT")
    line = (f"Total reviews: {s['total']}   Positive {s['positive']}%   "
            f"Neutral {s['neutral']}%   Negative {s['negative']}%")
    if s["avg_rating"] is not None:
        line += f"   Avg rating {s['avg_rating']}/5"
    L.append(line)
    L.append("\nTOP COMPLAINTS")
    for n, c in enumerate(insights["complaints"], 1):
        L.append(f"  {n}. {c['title']} ... {c['count']} review{'s' if c['count'] != 1 else ''}"
                 f"  (mostly {c['top_platform']})")
    L.append("\nTOP PRAISE")
    L.append("  " + (", ".join(f"{p['title']} ({p['count']} mention{'s' if p['count'] != 1 else ''})"
                               for p in insights["praises"]) or "None this period"))
    if s["hotspots"]:
        L.append("\nHOTSPOTS (highest % negative, 3+ reviews)")
        for seg, items in s["hotspots"].items():
            L.append(f"  {seg}: " + ", ".join(f"{v} ({p}% of {n})" for v, n, p in items))
    L.append(f"\nURGENT: {s['urgent']} reviews need a reply within 24 hours")
    L.append("\nACTION ITEMS")
    for n, a in enumerate(insights["actions"], 1):
        L.append(f"  {n}. {a}")
    return "\n".join(L)