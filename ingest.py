"""Universal CSV ingestion: accept ANY customer-feedback CSV, whatever its column names.

Pipeline:  read file -> AI looks at headers + sample rows -> ColumnMapping
           -> normalise() converts it to the 5 standard columns (+ keeps your other columns)

If the AI call fails (no key, network), a rule-based fallback guesses the mapping.
The user can always correct the mapping in the app before analysing.
"""
import re
from typing import Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm import get_llm
from schemas import ANALYSIS_COLUMNS, STANDARD_COLUMNS


class ColumnMapping(BaseModel):
    """Which of the file's own columns plays which role. Use EXACT column names or null."""

    review_text: Optional[str] = Field(
        default=None,
        description="Column holding the customer's written feedback (REQUIRED). If there are "
                    "both a title and a body, choose the longer body column.")
    rating: Optional[str] = Field(default=None, description="Numeric star/score column, or null.")
    date: Optional[str] = Field(default=None, description="Date or timestamp column, or null.")
    platform: Optional[str] = Field(
        default=None, description="Where the review was posted (Zomato, Swiggy, Google...), or null.")
    review_id: Optional[str] = Field(default=None, description="Unique review/order id column, or null.")
    segments: list[str] = Field(
        default_factory=list,
        description="Up to 2 columns useful for grouping complaints, such as branch, outlet, city, "
                    "restaurant, dish or delivery partner. Not free-text columns.")
    notes: str = Field(default="", description="One short sentence on how you decided.")


class MappingOut(BaseModel):
    """What the AI returns. No default values allowed here (OpenAI strict mode rejects them)."""

    review_text: Optional[str] = Field(
        description="Exact name of the column holding the customer's WRITTEN feedback (sentences). "
                    "REQUIRED if one exists; null if the file has no written feedback at all. If there "
                    "are both a title and a body, choose the longer body column.")
    rating: Optional[str] = Field(description="Numeric star/score column, or null.")
    date: Optional[str] = Field(description="Date or timestamp column of the review, or null.")
    platform: Optional[str] = Field(
        description="Where the review was posted (Zomato, Swiggy, Google...), or null.")
    review_id: Optional[str] = Field(description="Unique review/order id column, or null.")
    segments: list[str] = Field(
        description="Up to 2 columns useful for grouping complaints, such as branch, outlet, city, "
                    "restaurant, dish or delivery partner. Not free-text columns. Empty list if none.")
    notes: str = Field(description="One short sentence on how you decided.")


MAPPER_SYSTEM = """You are a data analyst. You receive the column names, types and sample values
of a CSV file that contains customer feedback for a restaurant or food-delivery business.
Decide which columns play which role. Use the EXACT column names given, or null when no
column fits. Never invent a column. The file may use any naming style or language."""

MAPPER_HUMAN = "Columns of the file:\n{profile}"


def build_mapper_chain():
    prompt = ChatPromptTemplate.from_messages([("system", MAPPER_SYSTEM), ("human", MAPPER_HUMAN)])
    return prompt | get_llm().with_structured_output(MappingOut)


# ---------------------------------------------------------------- reading ----
def read_csv_flexible(file) -> pd.DataFrame:
    """Read a CSV with unknown delimiter / encoding."""
    last = None
    for enc in ("utf-8-sig", "latin-1"):
        for sep in (None, ","):
            try:
                if hasattr(file, "seek"):
                    file.seek(0)
                df = pd.read_csv(file, sep=sep, engine="python", encoding=enc, on_bad_lines="skip")
                if len(df) and df.shape[1] >= 1:
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            except Exception as e:
                last = e
    raise ValueError(f"Could not read this file as a CSV ({last}).")


def profile_columns(df: pd.DataFrame, max_cols: int = 60, n_samples: int = 3) -> str:
    lines = []
    for col in list(df.columns)[:max_cols]:
        s = df[col].dropna()
        step = max(1, len(s) // n_samples)
        samples = [str(v).replace("\n", " ")[:100] for v in s.iloc[::step].head(n_samples)]
        lines.append(f"- {col!r} | type={df[col].dtype} | unique={s.nunique()} | samples={samples}")
    return "\n".join(lines)


# ------------------------------------------------------- fallback guesser ----
def _tokens(name: str) -> set:
    return set(re.split(r"[^a-z0-9]+", name.lower()))


def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)


def looks_like_feedback(series: pd.Series) -> bool:
    """True if the column holds written feedback (even short reviews), not ids, names or labels."""
    v = series.dropna().astype(str).str.strip().head(300)
    v = v[v != ""]
    if len(v) < 5:
        return False
    if v.str.match(r"^[A-Za-z]{0,3}[-_ #]?\d+$").mean() >= 0.5:           # codes: C5663, O-123, 4567
        return False
    if len(v) >= 30 and v.nunique() / len(v) < 0.15:                      # labels: Male, Cash, KFC...
        return False
    return (v.str.split().str.len() >= 3).mean() >= 0.35                  # real sentences


def heuristic_mapping(df: pd.DataFrame) -> ColumnMapping:
    def find(hints, exclude, numeric=None):
        for c in df.columns:
            if c in exclude:
                continue
            if numeric is not None and pd.api.types.is_numeric_dtype(df[c]) != numeric:
                continue
            if _tokens(c) & hints:
                return c
        return None

    used: set = set()
    scores = {}
    for c in df.columns:
        if _is_text(df[c]) and df[c].notna().any() and looks_like_feedback(df[c]):
            avg = df[c].dropna().astype(str).str.len().mean()
            hint = bool(_tokens(c) & {"review", "feedback", "comment", "comments", "text",
                                      "remarks", "content", "body", "message"})
            scores[c] = avg * (2.0 if hint else 1.0)
    text = max(scores, key=scores.get) if scores else None
    used.add(text)
    rating = find({"rating", "ratings", "stars", "star", "score"}, used)
    used.add(rating)
    date = find({"date", "time", "posted", "created", "timestamp", "datetime"}, used)
    used.add(date)
    platform = find({"platform", "source", "channel", "app", "site"}, used)
    used.add(platform)
    rid = find({"id", "ref", "reviewid", "orderid"}, used)
    used.add(rid)
    segs = []
    for c in df.columns:
        if c in used or not _is_text(df[c]) or len(segs) == 2:
            continue
        s = df[c].dropna().astype(str)
        if _tokens(c) & {"branch", "outlet", "restaurant", "city", "location", "store", "area",
                         "dish", "item", "partner", "zone", "region"} and 2 <= s.nunique() <= 60:
            segs.append(c)
    return ColumnMapping(review_text=text, rating=rating, date=date, platform=platform,
                         review_id=rid, segments=segs,
                         notes="Guessed from column names and text length (no AI).")


def _sanitise(m: ColumnMapping, df: pd.DataFrame) -> ColumnMapping:
    """Drop anything the AI named that is not really a column in the file."""
    cols, d, taken = set(df.columns), {}, set()
    for f in ("review_text", "review_id", "date", "rating", "platform"):   # priority order
        v = getattr(m, f)
        if f == "review_text" and v in cols and not looks_like_feedback(df[v]):
            v = None                                                     # ids / labels are not feedback
        d[f] = v if (v in cols and v not in taken) else None              # one column, one role
        if d[f]:
            taken.add(d[f])
    segs = []
    for c in m.segments:
        if c in cols and c not in taken and c not in segs:
            segs.append(c)
    return ColumnMapping(**d, segments=segs[:2], notes=m.notes)


def detect_mapping(df: pd.DataFrame, chain=None):
    """Returns (ColumnMapping, source) where source is 'ai' or 'fallback'."""
    why = ""
    try:
        chain = chain or build_mapper_chain()
        out = chain.invoke({"profile": profile_columns(df)})
        m = _sanitise(ColumnMapping(**out.model_dump()), df)
        if m.review_text:
            return m, "ai"
        why = "The AI found no column with written feedback."
    except Exception as e:
        why = f"AI mapping failed ({type(e).__name__}: {str(e)[:200]})."
    fb = _sanitise(heuristic_mapping(df), df)
    fb.notes = f"{why} Used rule-based guessing from column names and text instead."
    return fb, "fallback"


# ------------------------------------------------------------ normalising ----
def _norm_rating(series: pd.Series) -> pd.Series:
    nums = pd.to_numeric(series.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    mx = nums.max()
    if pd.isna(mx) or mx <= 5:
        return nums
    scale = 10 if mx <= 10 else 100 if mx <= 100 else mx
    return (nums / scale * 5).round(1)          # bring 1-10 / 0-100 scales to 5


def normalise(df: pd.DataFrame, m: ColumnMapping, max_rows: Optional[int] = None):
    """Convert any file to the standard columns. Returns (table, segment_column_names)."""
    if not m.review_text or m.review_text not in df.columns:
        raise ValueError("I could not find a column with customers' WRITTEN feedback. This looks like an "
                         "order or sales file (ids, dishes, prices), not a reviews file. Please upload a CSV "
                         "that has a review/comment/feedback text column, or pick that column above.")
    text = df[m.review_text].fillna("").astype(str).str.strip()
    work = df[text.str.len() >= 3]
    if work.empty:
        raise ValueError(f"The column '{m.review_text}' has no usable text.")
    if max_rows and len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=42).sort_index()
    return _build(m, work.reset_index(drop=True))


def _parse_dates(series: pd.Series):
    """ISO dates (2026-10-03) stay year-first; everything else is read day-first (03/10/2026)."""
    txt = series.astype(str)
    iso = txt.str.match(r"^\s*\d{4}-\d{1,2}-\d{1,2}")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(txt[iso], errors="coerce", dayfirst=False, format="mixed")
    if (~iso).any():
        rest = txt[~iso]
        parts = rest.str.extract(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}").astype(float)
        # 8/23/2023 has a "second part" above 12 -> it must be month-first. Otherwise assume day-first.
        month_first = (parts[1].max() > 12) and not (parts[0].max() > 12)
        out[~iso] = pd.to_datetime(rest, errors="coerce", dayfirst=not month_first, format="mixed")
    return out.dt.strftime("%Y-%m-%d").fillna("").values


def _build(m, work):
    n = len(work)
    out = pd.DataFrame(index=range(n))
    ids = work[m.review_id].astype(str).str.strip() if m.review_id else None
    if ids is None or ids.duplicated().any() or (ids == "").any():
        ids = pd.Series([f"R-{i + 1:04d}" for i in range(n)])
    out["review_id"] = ids.values
    if m.date:
        out["date"] = _parse_dates(work[m.date])
    else:
        out["date"] = ""
    out["platform"] = (work[m.platform].fillna("Unknown").astype(str).str.strip().values
                       if m.platform else "Unknown")
    out["rating"] = _norm_rating(work[m.rating]).values if m.rating else float("nan")
    out["review_text"] = work[m.review_text].astype(str).str.strip().values

    used = {c for c in (m.review_text, m.rating, m.date, m.platform, m.review_id) if c}
    reserved = set(STANDARD_COLUMNS) | set(ANALYSIS_COLUMNS)
    rename, extras = {}, work.drop(columns=list(used))
    for c in extras.columns:
        if c in reserved:
            rename[c] = f"{c}_original"
    # drop columns whose content is identical to the review text or id (pure duplicates)
    same = [c for c in extras.columns
            if extras[c].astype(str).str.strip().tolist() in
            (out["review_text"].tolist(), out["review_id"].tolist())]
    extras = extras.drop(columns=same)
    rename = {k: v for k, v in rename.items() if k not in same}
    extras = extras.rename(columns=rename).reset_index(drop=True)
    segments = [rename.get(s, s) for s in m.segments if s in work.columns and s not in same]
    return pd.concat([out, extras], axis=1), segments
