"""Data Insights mode: for files with NO written feedback (orders, ratings, prices, delivery times).

pandas computes every number; the AI only writes the explanation from those numbers.
"""
from typing import Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ingest import _norm_rating, _parse_dates, _tokens
from llm import get_llm

CODE = r"^[A-Za-z]{0,3}[-_ #]?\d+$"
# lower tier = more useful to a restaurant manager, suggested first
DIM_TIERS = [
    {"restaurant", "outlet", "branch", "store", "city", "area", "location", "zone", "dish", "cuisine",
     "category", "platform", "segment"},
    {"payment", "method", "type", "day", "item"},
    {"gender", "age"},
]
SKIP_TOKENS = {"id", "phone", "pincode", "zip", "date", "time", "timestamp", "created", "posted", "signup"}


def numeric_columns(df, exclude=()):
    out = []
    for c in df.columns:
        if c in exclude or _tokens(c) & SKIP_TOKENS or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique() > 5:
            out.append(c)
    return out


def suggest_dims(df: pd.DataFrame, exclude=(), limit: int = 4) -> list:
    """Columns that split the data into a sensible number of groups (city, restaurant, dish...)."""
    scored = []
    for c in df.columns:
        s = df[c].dropna()
        if c in exclude or len(s) < 10 or pd.api.types.is_numeric_dtype(s) or _tokens(c) & SKIP_TOKENS:
            continue
        n = s.nunique()
        if not (2 <= n <= 30) or n / len(s) > 0.5 or s.astype(str).str.match(CODE).mean() >= 0.5:
            continue
        tier = next((i for i, hints in enumerate(DIM_TIERS) if _tokens(c) & hints), len(DIM_TIERS))
        scored.append((tier, n, c))
    return [c for *_, c in sorted(scored)[:limit]]


def _bin_label(iv) -> str:
    return f"{iv.left:.0f}-{iv.right:.0f}" if abs(iv.right) >= 10 else f"{iv.left:.1f}-{iv.right:.1f}"


def compute_tables(df: pd.DataFrame, rating_col: Optional[str], dims: list, metrics: list,
                   date_col: Optional[str]) -> dict:
    d = df.copy()
    d["_rating"] = _norm_rating(d[rating_col]) if rating_col else float("nan")
    has_rating = d["_rating"].notna().any()
    min_n = max(5, int(len(d) * 0.01))
    t = {"rows": len(d), "has_rating": has_rating, "min_n": min_n, "dims": {}, "metrics": {}, "trend": None,
         "overall": None, "low_pct": None, "rating_dist": None}
    if has_rating:
        r = d["_rating"].dropna()
        t["overall"] = round(float(r.mean()), 2)
        t["low_pct"] = round(float((r <= 2).mean() * 100), 1)
        t["rating_dist"] = r.round().astype(int).value_counts().sort_index()
    for dim in dims:
        g = d.groupby(dim).agg(n=(dim, "size"))
        if has_rating:
            g["avg"] = d.groupby(dim)["_rating"].mean().round(2)
            g["low_pct"] = (d.groupby(dim)["_rating"].apply(lambda s: (s <= 2).sum() / max(s.notna().sum(), 1)) * 100).round(1)
        g = g[g["n"] >= min_n]
        if not g.empty:
            t["dims"][dim] = g.sort_values("avg" if has_rating else "n", ascending=has_rating)
    if has_rating:
        for m in metrics:
            try:
                bins = pd.qcut(d[m], 4, duplicates="drop")
            except Exception:
                continue
            g = d.groupby(bins, observed=True).agg(n=("_rating", "size"), avg=("_rating", "mean"))
            g.index = [_bin_label(iv) for iv in g.index]
            g["avg"] = g["avg"].round(2)
            if len(g) >= 2:
                t["metrics"][m] = g
    if date_col:
        dt = pd.to_datetime(pd.Series(_parse_dates(d[date_col])), errors="coerce")
        tr = pd.DataFrame({"month": dt.dt.to_period("M").astype(str), "rating": d["_rating"].values})
        tr = tr[dt.notna().values]
        if tr["month"].nunique() >= 2:
            g = tr.groupby("month").agg(n=("month", "size"), avg=("rating", "mean"))
            g["avg"] = g["avg"].round(2)
            t["trend"] = g
    return t


def tables_to_text(t: dict) -> str:
    L = [f"Rows: {t['rows']}. Groups smaller than {t['min_n']} rows are hidden."]
    if t["has_rating"]:
        L.append(f"Overall average rating: {t['overall']}/5. Share of low ratings (2 or below): {t['low_pct']}%.")
        L.append(f"Rating counts: {t['rating_dist'].to_dict()}")
    else:
        L.append("No rating column: only counts are available.")
    for dim, g in t["dims"].items():
        L.append(f"\nBy {dim} (n, avg rating, % low):\n" + g.head(15).to_string())
    for m, g in t["metrics"].items():
        L.append(f"\nAverage rating by {m} quartile:\n" + g.to_string())
    if t["trend"] is not None:
        L.append("\nBy month:\n" + t["trend"].to_string())
    return "\n".join(L)


class DataInsights(BaseModel):
    headline: str = Field(description="One sentence summary of the most important finding.")
    findings: list[str] = Field(description="3 to 5 findings, each quoting numbers from the statistics.")
    action_items: list[str] = Field(description="Exactly 3 practical actions for a restaurant manager.")
    caveats: list[str] = Field(description="1 to 3 honest limits of this analysis.")


SYSTEM = """You analyse a restaurant / food-delivery dataset that has NO written customer reviews.
You receive precomputed statistics. Write plain-English insights for a manager.
Rules:
- Use ONLY numbers that appear in the statistics. Never invent or recompute figures.
- If the gap between the best and worst group is under 0.2 stars, say there is no meaningful
  difference between them. Do not make a story out of noise.
- Mention sample sizes (n) when a group is small.
- Describe patterns, never causes ("is associated with", not "is caused by").
- action_items: exactly 3, each starting with a verb and tied to a finding.
- caveats: always say that without written reviews the reasons behind low ratings are unknown."""


def generate_data_insights(t: dict, chain=None) -> DataInsights:
    chain = chain or (ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "Statistics:\n{stats}")])
                      | get_llm().with_structured_output(DataInsights))
    return chain.invoke({"stats": tables_to_text(t)})


# ---------------------------------------------- follow-ups for low-rated orders ----
class FollowUp(BaseModel):
    message: str = Field(description="The follow-up message to the customer: 2 to 3 sentences, English.")


FOLLOWUP_SYSTEM = """You write a short, warm follow-up message from a restaurant / food-delivery business to a
customer who gave a LOW rating but wrote no comment.
You are given ONLY structured facts (rating, restaurant, dish, city...). You do NOT know what went wrong.
Rules:
- NEVER guess or claim a specific problem (late, cold, rude, wrong item). Say you are sorry the experience
  fell short, in general terms.
- Mention the restaurant or dish only if it appears in the facts.
- Invite them to reply (or contact support with their id) and tell you what went wrong so you can fix it.
- Do not promise refunds, discounts or compensation.
- Never mention private details such as age or gender, or the customer's city unless it is a restaurant branch.
- 2 to 3 sentences, polite, no emojis, English only."""


def build_followup_chain():
    prompt = ChatPromptTemplate.from_messages([("system", FOLLOWUP_SYSTEM), ("human", "Facts:\n{facts}")])
    return prompt | get_llm(temperature=0.3).with_structured_output(FollowUp)


def count_low(df: pd.DataFrame, rating_col: str, max_rating: int) -> int:
    r = _norm_rating(df[rating_col])
    return int((r.notna() & (r <= max_rating)).sum())


def generate_followups(df: pd.DataFrame, rating_col: str, id_col: Optional[str], context_cols: list,
                       max_rating: int = 2, limit: int = 25, chain=None):
    """One follow-up per low-rated order (lowest ratings first). Returns (table, total_low_rated)."""
    r = _norm_rating(df[rating_col])
    low = df[r.notna() & (r <= max_rating)].copy()
    low["_r"] = r[low.index]
    total = len(low)
    low["_id"] = low[id_col].astype(str) if id_col else [f"row-{i + 1}" for i in low.index]
    low = low.sort_values("_r", kind="stable").head(limit).reset_index(drop=True)
    chain = chain or build_followup_chain()

    def facts(row):
        parts = [f"Customer/order id: {row['_id']}", f"Rating given: {row['_r']:g}/5"]
        for c in context_cols:
            v = row[c]
            parts.append(f"{c}: {'not provided' if pd.isna(v) else v}")
        return "\n".join(parts)

    res = chain.batch([{"facts": facts(row)} for _, row in low.iterrows()],
                      config={"max_concurrency": 5}, return_exceptions=True)
    out = pd.DataFrame({"id": low["_id"], "rating": low["_r"]})
    for c in context_cols:
        out[c] = low[c]
    out["follow_up_message"] = ["[failed, try again]" if isinstance(x, Exception) else x.message for x in res]
    return out, total


# ------------------------------------------------- weekly report (ratings only) ----
class ReportActions(BaseModel):
    action_items: list[str] = Field(description="Exactly 2 specific actions, each starting with a verb.")


ACTIONS_SYSTEM = """You write action items for a restaurant manager from rating statistics (there are NO written reviews).
Write exactly 2 action items. Use ONLY the groups and numbers given. Tie each action to a named problem area.
Describe what to check or fix, never claim a cause. If no problem area stands out, write one action to
collect written feedback (for example by adding a comment box to the rating screen) and one to keep monitoring.
Each action is one sentence starting with a verb."""


def build_data_report(df: pd.DataFrame, rating_col: str, dims: list, metrics: list,
                      dates: Optional[pd.Series] = None) -> dict:
    """All numbers for the report, counted by code. Rating bands: 4-5 positive, 3 neutral, below 3 negative.
    `dates` = review dates read ONCE from the full file (aligned to df's index), so a slice of the file
    can never be mis-read as day-first / month-first."""
    r = _norm_rating(df[rating_col])
    d = df[r.notna()].copy()
    d["_r"] = r[r.notna()]
    n = len(d)
    if n == 0:
        raise ValueError("The rating column has no usable numbers.")
    neg_all, pos_all = float((d["_r"] < 3).mean()), float((d["_r"] >= 4).mean())
    min_n = max(20, int(n * 0.01))
    groups = []

    def add(label, g):
        if len(g) >= min_n:
            neg, pos = int((g["_r"] < 3).sum()), int((g["_r"] >= 4).sum())
            groups.append({"label": label, "n": len(g), "neg": neg, "pos": pos,
                           "neg_pct": neg / len(g), "pos_pct": pos / len(g)})

    for dim in dims:
        for val, g in d.groupby(dim):
            add(f"{dim} = {val}", g)
    for m in metrics:
        try:
            bins = pd.qcut(d[m], 4, duplicates="drop")
        except Exception:
            continue
        for iv, g in d.groupby(bins, observed=True):
            add(f"{m} {_bin_label(iv)}", g)

    problems = sorted((g for g in groups if g["neg_pct"] >= neg_all + 0.05),
                      key=lambda g: -(g["neg"] - neg_all * g["n"]))[:3]
    praises = sorted((g for g in groups if g["pos_pct"] >= pos_all + 0.05),
                     key=lambda g: -(g["pos"] - pos_all * g["n"]))[:2]
    span = None
    if dates is not None:
        dt = dates.loc[d.index]
        if dt.notna().any():
            span = (dt.min(), dt.max())
    return {"total_rows": len(df), "rated": n, "avg": round(float(d["_r"].mean()), 2),
            "positive": round(pos_all * 100), "negative": round(neg_all * 100),
            "neutral": round(float(((d["_r"] >= 3) & (d["_r"] < 4)).mean()) * 100),
            "urgent": int((d["_r"] <= 1).sum()), "problems": problems, "praises": praises,
            "neg_all": neg_all, "dates": span}


def report_facts_text(st_: dict) -> str:
    L = [f"Overall: {st_['rated']} rated orders, average {st_['avg']}/5, {st_['negative']}% negative (1-2 stars)."]
    if st_["problems"]:
        L.append("Problem areas (more low ratings than average):")
        L += [f"- {p['label']}: {p['neg_pct']:.0%} low ratings of {p['n']} orders" for p in st_["problems"]]
    else:
        L.append("No group has clearly more low ratings than average.")
    if st_["praises"]:
        L.append("Best areas: " + "; ".join(f"{p['label']} ({p['pos_pct']:.0%} rated 4-5)" for p in st_["praises"]))
    return "\n".join(L)


def generate_report_actions(st_: dict, chain=None) -> list:
    chain = chain or (ChatPromptTemplate.from_messages([("system", ACTIONS_SYSTEM), ("human", "{facts}")])
                      | get_llm().with_structured_output(ReportActions))
    acts = chain.invoke({"facts": report_facts_text(st_)}).action_items[:2]
    acts.append(f"Send a personal follow-up to the {st_['urgent']} orders rated 1 star today "
                "(use the follow-up messages section below).")
    return acts


def _d(ts) -> str:
    return f"{ts.day} {ts:%b}"


def render_data_report(st_: dict, actions: list) -> str:
    L = []
    if st_["dates"]:
        a, b = st_["dates"]
        L.append(f"{'WEEKLY ' if (b - a).days <= 7 else ''}FEEDBACK REPORT  |  {_d(a)} to {_d(b)} {b.year}")
    else:
        L.append("FEEDBACK REPORT")
    L.append(f"Total orders: {st_['total_rows']} ({st_['rated']} rated)   Positive {st_['positive']}%   "
             f"Neutral {st_['neutral']}%   Negative {st_['negative']}%   Avg rating {st_['avg']}/5")
    L.append("\nTOP 3 PROBLEM AREAS  (most extra low ratings compared with the average)")
    if st_["problems"]:
        for i, p in enumerate(st_["problems"], 1):
            L.append(f"  {i}. {p['label']} ... {p['neg']} low ratings  "
                     f"({p['neg_pct']:.0%} of {p['n']}, overall {st_['neg_all']:.0%})")
    else:
        L.append("  No group stands out: low ratings are spread evenly.")
    L.append("\nTOP PRAISE")
    L.append("  " + (", ".join(f"{p['label']} ({p['pos_pct']:.0%} rated 4-5 stars)" for p in st_["praises"])
                     or "No group stands out."))
    L.append(f"\nURGENT: {st_['urgent']} orders rated 1 star need a personal follow-up within 24 hours")
    L.append("\nACTION ITEMS")
    for i, a in enumerate(actions, 1):
        L.append(f"  {i}. {a}")
    L.append("\nNOTE: this file has no written feedback, so the reasons behind low ratings are unknown. "
             "Positive = 4-5 stars, neutral = 3, negative = 1-2.")
    return "\n".join(L)
